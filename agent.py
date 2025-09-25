import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Bernoulli
from typing import Dict
from config import Config
from models import ActorCritic
from utils import RunningMeanStd


class PPOAgent:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"成功找到 GPU！")
            print(f"设备名称: {torch.cuda.get_device_name(0)}")
            print(f"CUDA 版本: {torch.version.cuda}")

        self.network = ActorCritic(config).to(self.device)

        # Param groups: no weight_decay for log_std
        decay_params, no_decay_params = [], []
        for n, p in self.network.named_parameters():
            if 'log_std' in n:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        self.optimizer = optim.AdamW(
            [{'params': decay_params, 'weight_decay': 1e-5},
             {'params': no_decay_params, 'weight_decay': 0.0}],
            lr=config.lr_actor
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.lr_decay)

        self.buffer = []
        self.training_step = 0

        self.reward_stats = RunningMeanStd()
        self.advantage_stats = RunningMeanStd()

    def select_action(self, state: np.ndarray, training: bool=True):
        """
        state -> numpy array
        返回 (action: np.array([size, bps, post]), logprob: float or None)
        """
        # 防止 state 中出现 NaN/Inf
        state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        try:
            with torch.no_grad():
                if training:
                    action_t, logprob_t = self.network.get_action(s, training=True)
                    action_np = action_t.cpu().numpy()[0]
                    # logprob_t 可能是 tensor 标量
                    logprob_val = float(logprob_t.cpu().numpy()[0]) if isinstance(logprob_t, torch.Tensor) or hasattr(logprob_t, 'numpy') else float(logprob_t)
                    # 防止返回 inf/nan
                    if np.isnan(action_np).any() or not np.isfinite(logprob_val):
                        print("⚠️ select_action produced NaN/Inf. action/logprob:", action_np, logprob_val)
                        raise ValueError("NaN/Inf in select_action outputs")
                    return action_np, logprob_val
                else:
                    action_t, _ = self.network.get_action(s, training=False)
                    action_np = action_t.cpu().numpy()[0]
                    if np.isnan(action_np).any():
                        print("⚠️ deterministic select_action produced NaN.")
                        raise ValueError("NaN in deterministic action")
                    return action_np, None
        except ValueError as e:
            # 打印调试信息并把异常向上抛出（或按需返回一个保底动作）
            print("=== select_action exception ===")
            print(e)
            # 打印网络前向关键输出，帮助定位（在 no_grad 下可安全调用 forward）
            try:
                with torch.no_grad():
                    cont_mean, cont_std, post_logit, _ = self.network(s)
                    print("cont_mean:", cont_mean)
                    print("cont_std:", cont_std)
                    print("post_logit:", post_logit)
            except Exception as e2:
                print("Unable to extract debug values:", e2)
            raise

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        # 防止 NaN/Inf
        state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        next_state = np.nan_to_num(next_state, nan=0.0, posinf=1e6, neginf=-1e6)
        self.buffer.append({
            'state': state,
            'action': action,      
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.config.batch_size:
            return {}

        states = torch.tensor([e['state'] for e in self.buffer], dtype=torch.float32, device=self.device)
        actions = torch.tensor([e['action'] for e in self.buffer], dtype=torch.float32, device=self.device)  # [N,3]
        rewards = torch.tensor([e['reward'] for e in self.buffer], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([e['next_state'] for e in self.buffer], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e['done'] for e in self.buffer], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([e['log_prob'] for e in self.buffer], dtype=torch.float32, device=self.device)

        # Normalize rewards if enabled
        if self.config.use_reward_normalization:
            self.reward_stats.update(rewards.detach().cpu().numpy())
            rewards = (rewards - self.reward_stats.mean) / (self.reward_stats.std + 1e-8)

        # Compute values and next values
        with torch.no_grad():
            cont_mean, cont_std, post_logit, values = self.network(states)
            _, _, _, next_values = self.network(next_states)
            old_values = values.clone()

        # Compute returns and advantages using corrected GAE
        returns, advantages = self._compute_gae(rewards, values, next_values, dones)

        # Normalize advantages
        if self.config.use_advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to dataset
        dataset = {
            'states': states,
            'actions': actions,
            'returns': returns,
            'advantages': advantages,
            'old_log_probs': old_log_probs,
            'old_values': old_values
        }

        metrics = self._update_policy(dataset)
        
        # scheduler step after policy update
        self.scheduler.step()
        self.buffer.clear()
        self.training_step += 1

        return metrics

    def _compute_gae(self, rewards, values, next_values, dones):
        T = len(rewards)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        gamma, lam = self.config.gamma, self.config.gae_lambda
        
        next_v = torch.zeros_like(values)
        next_v[:-1] = values[1:]
        # next_values is shape [N], we want bootstrapped last-element
        next_v[-1] = next_values[-1] if next_values.numel() > 0 else 0.0

        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_v[t] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        return returns, advantages

    def _update_policy(self, dataset):
        actor_losses, critic_losses, entropies = [], [], []
        kl_divs, clipfracs, value_clipfracs = [], [], []

        N = len(dataset['states'])

        for epoch in range(self.config.n_epochs_per_update):
            indices = torch.randperm(N, device=self.device)

            for start in range(0, N, self.config.batch_size):
                end = min(start + self.config.batch_size, N)
                batch_indices = indices[start:end]

                batch_states = dataset['states'][batch_indices]
                batch_actions = dataset['actions'][batch_indices]
                batch_returns = dataset['returns'][batch_indices]
                batch_advantages = dataset['advantages'][batch_indices]
                batch_old_log_probs = dataset['old_log_probs'][batch_indices]
                batch_old_values = dataset['old_values'][batch_indices]

                # Forward pass
                cont_mean, cont_std, post_logit, values = self.network(batch_states)

                # --- 防止 NaN / Inf ---
                cont_mean = torch.nan_to_num(cont_mean, nan=0.0, posinf=1e6, neginf=-1e6)
                cont_std = torch.nan_to_num(cont_std, nan=1.0, posinf=1e6, neginf=1e-6)
                cont_std = torch.clamp(cont_std, min=1e-6, max=5.0)  # 限制 std 范围

                # construct distributions
                cont_dist = Normal(cont_mean, cont_std)
                actions_cont = batch_actions[:, :2]
                actions_cont = torch.nan_to_num(actions_cont, nan=0.0, posinf=1e6, neginf=-1e6)
                cont_log_prob = cont_dist.log_prob(actions_cont).sum(dim=-1)

                # binary
                bin_dist = Bernoulli(logits=post_logit)
                bin_actions = batch_actions[:, 2]
                bin_actions = torch.nan_to_num(bin_actions, nan=0.0, posinf=1.0, neginf=0.0)
                bin_log_prob = bin_dist.log_prob(bin_actions)

                new_log_probs = cont_log_prob + bin_log_prob

                # Protect against NaN/Inf in log_probs
                new_log_probs = torch.nan_to_num(new_log_probs, nan=-1e8, posinf=1e8, neginf=-1e8)
                batch_old_log_probs = torch.nan_to_num(batch_old_log_probs, nan=-1e8, posinf=1e8, neginf=-1e8)

                # PPO ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1e8, neginf=0.0)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # entropy bonus
                cont_entropy = cont_dist.entropy().sum(dim=-1).mean()
                bin_entropy = bin_dist.entropy().mean()
                entropy = cont_entropy + bin_entropy

                # value loss with optional clipping
                if getattr(self.config, "use_value_clipping", False):
                    values_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values,
                        -self.config.clip_ratio,
                        self.config.clip_ratio
                    )
                    value_loss_unclipped = F.mse_loss(values, batch_returns, reduction='none')
                    value_loss_clipped = F.mse_loss(values_clipped, batch_returns, reduction='none')
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    value_clipfrac = (value_loss_clipped > value_loss_unclipped).float().mean()
                    value_clipfracs.append(value_clipfrac.item())
                else:
                    value_loss = F.mse_loss(values, batch_returns)

                total_loss = (
                    actor_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_beta * entropy
                )

                # --- backward with grad checks ---
                self.optimizer.zero_grad()
                total_loss.backward()

                # 检查梯度是否异常
                skip_update = False
                for p in self.network.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            skip_update = True
                            print("⚠️ Detected NaN/Inf in grads, skipping optimizer.step() for this mini-batch.")
                            break

                if skip_update:
                    self.optimizer.zero_grad()
                    continue

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)

                # step
                self.optimizer.step()

                # logging
                actor_losses.append(actor_loss.item())
                critic_losses.append(value_loss.item())
                entropies.append(entropy.item())

                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean()
                    clipfrac = (torch.abs(ratio - 1.0) > self.config.clip_ratio).float().mean()
                    kl_divs.append(kl_div.item())
                    clipfracs.append(clipfrac.item())

        metrics = {
            'actor_loss': float(np.mean(actor_losses)) if actor_losses else 0.0,
            'critic_loss': float(np.mean(critic_losses)) if critic_losses else 0.0,
            'entropy': float(np.mean(entropies)) if entropies else 0.0,
            'kl_divergence': float(np.mean(kl_divs)) if kl_divs else 0.0,
            'clip_fraction': float(np.mean(clipfracs)) if clipfracs else 0.0,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

        if value_clipfracs:
            metrics['value_clip_fraction'] = float(np.mean(value_clipfracs))

        return metrics

    def save(self, path: str):
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'training_step': self.training_step,
            'reward_stats': {'mean': float(self.reward_stats.mean),
                             'std': float(self.reward_stats.std),
                             'count': float(self.reward_stats.count)}
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt['network_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.training_step = ckpt['training_step']
        if 'reward_stats' in ckpt:
            self.reward_stats.mean = ckpt['reward_stats']['mean']
            self.reward_stats.std = ckpt['reward_stats']['std']
            self.reward_stats.count = ckpt['reward_stats']['count']