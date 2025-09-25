import os
import torch
import numpy as np
import pandas as pd
import warnings
import argparse
from typing import Dict
from config import Config
from envs import MarketEnv
from agent import PPOAgent
from utils import simulate_twap_taker, simulate_is_baseline, argss
# Make sure your visualization script is named visualize.py or change the import
from visualize import create_visualization
import logging 
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_system(data_path: str = "data/AA_Comdty_cpu",
                 n_episodes: int = 500,
                 save_interval: int = 50,
                 side: str = 'buy') -> Dict:

    config = Config(side=args.side)
    if argss() is not None:
        args = argss()
        config.lr_actor = args.lr
        config.gamma = args.gamma
        config.n_epochs_per_update = args.n_epochs_per_update
        config.batch_size = args.batch_size
        config.buffer_size = args.buffer_size
        config.clip_ratio = args.clip_range
        config.value_loss_coef = args.vf_coef
        config.entropy_beta = args.ent_coef
        config.initial_inventory = args.initial_inventory
        config.time_horizon = args.time_horizon
        config.normalize_action = args.act_norm
        config.normalize_states = args.state_norm

    # --- MODIFICATION START ---
    # 1. Create a unique name for the experiment directory
    data_folder_name = os.path.basename(data_path)
    experiment_name = (f"{data_folder_name}_"
                       f"side_{config.side}_"
                       f"inv{config.initial_inventory}_"
                       f"th{config.time_horizon}")

    # 2. Define paths for the new directories
    base_dir = os.path.join('experiments_results', experiment_name)
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    results_dir = os.path.join(base_dir, 'output')
    result_dir=os.path.join(base_dir, 'resutls')

    # 3. Create the directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)


    log_file = os.path.join(result_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file), 
            logging.StreamHandler()        
        ]
    )
    logging.info(f"Experiment started: {experiment_name}")


    print(f"Experiment outputs will be saved in: '{base_dir}/'")
    # --- MODIFICATION END ---

    print(f"Loading data from {args.data_path}...")
    train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
    valid_data = pd.read_csv(os.path.join(data_path, "valid.csv"))
    test_data  = pd.read_csv(os.path.join(data_path, "test.csv"))
    print(f"Train: {len(train_data):,} | Valid: {len(valid_data):,} | Test: {len(test_data):,}")

    train_env = MarketEnv(train_data, config, mode='train', reward_type=args.reward_type)
    valid_env = MarketEnv(valid_data, config, mode='valid', reward_type=args.reward_type)
    test_env  = MarketEnv(test_data, config, mode='test', reward_type=args.reward_type)

    agent = PPOAgent(config)
    agent.network.to(device)
    agent.network.train()

    history = {
        'train_rewards': [], 'train_completion': [], 'train_costs': [],
        'valid_rewards': [], 'valid_completion': [], 'valid_costs': [],
        'best_valid_reward': -float('inf'), 'best_valid_completion': 0
    }

    patience_counter = 0
    best_valid_score = -float('inf')

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    for ep in range(args.n_episodes):
        state = train_env.reset()
        ep_reward = 0.0
        ep_costs = []

        for _ in range(config.time_horizon):
            action, logp = agent.select_action(state, training=True)
            next_state, reward, done, info = train_env.step(action)
            agent.store_transition(state, action, reward, next_state, done, logp)
            ep_reward += reward
            if info['trade_executed']:
                ep_costs.append(info['trade_cost_bps'])
            state = next_state
            if done: break

        history['train_rewards'].append(ep_reward)
        history['train_completion'].append(info['completion_rate'])
        history['train_costs'].append(np.mean(ep_costs) if ep_costs else 0.0)

        if len(agent.buffer) >= config.buffer_size:
            _ = agent.update()

        if (ep+1) % 25 == 0:
            agent.network.eval()
            with torch.no_grad():
                state = valid_env.reset()
                v_reward, v_costs = 0.0, []
                for _ in range(config.time_horizon):
                    action, _ = agent.select_action(state, training=False)
                    next_state, reward, done, vinfo = valid_env.step(action)
                    v_reward += reward
                    if vinfo['trade_executed']:
                        v_costs.append(vinfo['trade_cost_bps'])
                    state = next_state
                    if done: break

            history['valid_rewards'].append(v_reward)
            history['valid_completion'].append(vinfo['completion_rate'])
            history['valid_costs'].append(np.mean(v_costs) if v_costs else 0.0)

            valid_score = v_reward + 100*vinfo['completion_rate']
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                history['best_valid_reward'] = v_reward
                history['best_valid_completion'] = vinfo['completion_rate']
                patience_counter = 0
                # Use the new checkpoint_dir path
                agent.save(os.path.join(checkpoint_dir, 'best_model.pth'))
                print("---New best model saved!---")
            else:
                patience_counter += 1

            print(f"\nEpisode {ep+1}/{n_episodes}")
            print(f"  Train - Reward: {np.mean(history['train_rewards'][-25:]):.1f} | "
                  f"Completion: {np.mean(history['train_completion'][-25:])*100:.1f}% | "
                  f"Cost: {np.mean(history['train_costs'][-25:]):.1f} bps")
            print(f"  Valid - Reward: {v_reward:.1f} | Completion: {vinfo['completion_rate']*100:.1f}% | "
                  f"Cost: {np.mean(v_costs) if v_costs else 0:.1f} bps")
            print(f"  Learning - LR: {agent.scheduler.get_last_lr()[0]:.6f}")

            logging.info(f"\nEpisode {ep+1}/{n_episodes}")
            logging.info(f"  Train - Reward: {np.mean(history['train_rewards'][-25:]):.1f} | "
                  f"Completion: {np.mean(history['train_completion'][-25:])*100:.1f}% | "
                  f"Cost: {np.mean(history['train_costs'][-25:]):.1f} bps")
            logging.info(f"  Valid - Reward: {v_reward:.1f} | Completion: {vinfo['completion_rate']*100:.1f}% | "
                  f"Cost: {np.mean(v_costs) if v_costs else 0:.1f} bps")
            logging.info(f"  Learning - LR: {agent.scheduler.get_last_lr()[0]:.6f}")

            if patience_counter >= config.patience:
                print(f"\n⚠ Early stopping triggered (patience={config.patience})")
                break
            agent.network.train()

        if (ep+1) % args.save_interval == 0:
            # Use the new checkpoint_dir path
            agent.save(os.path.join(checkpoint_dir, f'checkpoint_ep{ep+1}.pth'))

    # ---------- Final testing ----------
    print("\n" + "="*60)
    print("FINAL TESTING")
    print("="*60)

    # Use the new checkpoint_dir path
    agent.load(os.path.join(checkpoint_dir, 'best_model.pth'))
    agent.network.eval()
    with torch.no_grad():
        state = test_env.reset()
        test_reward, test_costs, test_trades = 0.0, [], []
        for _ in range(config.time_horizon):
            action, _ = agent.select_action(state, training=False)
            next_state, reward, done, info = test_env.step(action)
            test_reward += reward
            if info['trade_executed']:
                test_costs.append(info['trade_cost_bps'])
                test_trades.append({'step': test_env.current_step,
                                    'size': info['order_size'],
                                    'filled': info['filled_qty'],
                                    'cost': info['trade_cost_bps']})
            state = next_state
            if done: break

    test_completion = info['completion_rate']
    avg_cost = np.mean(test_costs) if test_costs else 0.0

    # ... (The rest of the calculations remain the same) ...
    # Shortfall vs VWAP
    if test_env.executed_qty > 0:
        avg_px = test_env.executed_value / test_env.executed_qty
        vwap = test_env.vwap_numerator / max(1, test_env.vwap_denominator)
        if config.side == 'buy':
            shortfall_bps = 10000*(avg_px - vwap)/vwap
        else:
            shortfall_bps = 10000*(vwap - avg_px)/vwap
    else:
        shortfall_bps = 0.0

    # Portfolio Value vs final mid (for traded quantity)
    final_mid = test_env.processed_data.iloc[min(test_env.current_idx, len(test_env.processed_data)-1)]['mid']
    if config.side == 'buy':
        portfolio_value = -test_env.executed_value + test_env.executed_qty * final_mid
    else:
        portfolio_value = +test_env.executed_value - test_env.executed_qty * final_mid

    twap_baseline = simulate_twap_taker(test_env.processed_data, config)
    is_baseline = simulate_is_baseline(test_env.processed_data, config)
    print(f"\nFINAL TEST RESULTS (side={config.side.upper()}):")
    print(f"  RL  - Reward: {test_reward:.1f} | Completion: {test_completion*100:.2f}% | "
          f"Avg Cost: {avg_cost:.2f} bps | Shortfall(VWAP): {shortfall_bps:.2f} bps | "
          f"PV: {portfolio_value:.2f} | Trades: {len(test_env.trades)}")
    print(f"  TWAP- Completion: {twap_baseline['completion']*100:.2f}% | "
          f"Avg Cost: {twap_baseline['avg_cost_bps']:.2f} bps | Shortfall(VWAP): {twap_baseline['shortfall_bps']:.2f} bps | "
          f"PV: {twap_baseline['pv']:.2f} | Trades: {twap_baseline['num_trades']}")
    print(f"  IS  - Completion: {is_baseline['completion']*100:.2f}% | "
          f"IS Cost: {is_baseline['avg_cost_bps']:.2f} bps | "
          f"PV: {is_baseline['pv']:.2f} | Trades: {is_baseline['num_trades']}")
    
    test_results_df = pd.DataFrame([
        {'metric': 'Completion Rate', 'RL': test_completion, 'TWAP': twap_baseline['completion'], 'IS': is_baseline['completion']},
        {'metric': 'Avg Cost (bps)', 'RL': avg_cost, 'TWAP': twap_baseline['avg_cost_bps'], 'IS': is_baseline['avg_cost_bps']},
        {'metric': 'VWAP Shortfall (bps)', 'RL': shortfall_bps, 'TWAP': twap_baseline['shortfall_bps'], 'IS': is_baseline['shortfall_bps']},
        {'metric': 'Portfolio Value', 'RL': portfolio_value, 'TWAP': twap_baseline['pv'], 'IS': is_baseline['pv']},
        {'metric': 'Number of Trades', 'RL': len(test_env.trades), 'TWAP': twap_baseline['num_trades'], 'IS': is_baseline['num_trades']},
    ])
    results_csv_path = os.path.join(result_dir, 'final_test_summary.csv')
    test_results_df.to_csv(results_csv_path, index=False)
    print(f"\n[+] Final test summary saved to: {results_csv_path}")

    rl_trades_df = pd.DataFrame(test_env.trades)
    trades_csv_path = os.path.join(result_dir, 'rl_agent_trades.csv')
    rl_trades_df.to_csv(trades_csv_path, index=False)
    print(f"[+] RL agent trade log saved to: {trades_csv_path}")

    # --- MODIFICATION: Pass the new results_dir to the visualization function ---
    create_visualization(history, test_env, config, twap_baseline, is_baseline, save_dir=result_dir)

    return {
        'history': history,
        'test_reward': test_reward,
        'test_completion': test_completion,
        'test_cost': avg_cost,
        'test_shortfall': shortfall_bps,
        'test_trades': test_trades,
        'test_pv': portfolio_value,
        'is_baseline': is_baseline,
        'twap_baseline': twap_baseline
    }


if __name__ == "__main__":
    args = argss()
    results = train_system(args)
