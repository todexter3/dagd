import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Tuple
from config import Config
from utils import RunningMeanStd
from normalizer import ActionNormalizer

class MarketEnv:
    """
    Environment with:
    - Limit order semantics: orders can be non-marketable (post-only), partial/no fill with queue model
    - Two-sided (buy/sell)
    """
    def __init__(self, data: pd.DataFrame, config: Config, mode='train', reward_type='cpu'):
        self.data = data.reset_index(drop=True)
        self.config = config
        assert self.config.side in ('buy', 'sell')
        self.mode = mode
        self.reward_type = reward_type

        self._preprocess_data()

        # state and action normalizer
        self.state_normalizer = RunningMeanStd(shape=(config.state_dim,))
        self.action_normalizer = ActionNormalizer(max_price_offset_bps=config.max_price_offset_bps)

        #  State normalization settings
        self.normalize_states = getattr(config, 'normalize_states', True)
        self.update_state_stats = True  # Can be disabled during evaluation

        self.out_csv_name = "results/results"
        self.run = 0
        self.metrics = []
        self.step_index = 0

        self.price_history = deque(maxlen=config.history_len)
        self.volume_history = deque(maxlen=config.history_len)
        self.spread_history = deque(maxlen=config.history_len)

    def _preprocess_data(self):
        df = self.data.copy()
        df['mid'] = (df['bidPrice1'] + df['askPrice1']) / 2
        df['spread'] = df['askPrice1'] - df['bidPrice1']
        df['spread_bps'] = 10000 * df['spread'] / df['mid']

        # df['bid_depth'] = df[['bidVolume1', 'bidVolume2', 'bidVolume3']].sum(axis=1)
        df['bid_depth'] = df[['bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5']].sum(axis=1)
        df['ask_depth'] = df[['askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5']].sum(axis=1)
        df['total_depth'] = df['bid_depth'] + df['ask_depth']
        df['imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['total_depth'] + 1)

        df['bid_wavg'] = (
            df['bidPrice1'] * df['bidVolume1'] +
            df['bidPrice2'] * df['bidVolume2'] +
            df['bidPrice3'] * df['bidVolume3'] +
            df['bidPrice4'] * df['bidVolume4'] +
            df['bidPrice5'] * df['bidVolume5']
        ) / (df['bid_depth'] + 1)

        df['ask_wavg'] = (
            df['askPrice1'] * df['askVolume1'] +
            df['askPrice2'] * df['askVolume2'] +
            df['askPrice3'] * df['askVolume3'] +
            df['askPrice4'] * df['askVolume4'] +
            df['askPrice5'] * df['askVolume5']
        ) / (df['ask_depth'] + 1)

        for window in [5, 10, 20]:
            df[f'ret_{window}'] = df['mid'].pct_change(window)
            df[f'vol_{window}'] = df['mid'].pct_change().rolling(window).std()

        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_5'] + 1)

        df['bid_ask_spread_ratio'] = df['spread'] / df['mid']
        df['depth_imbalance_ratio'] = df['imbalance'].rolling(5).mean()

        df = df.fillna(0)
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['time', 'bidPrice1', 'askPrice1', 'mid']:
                df[col] = df[col].clip(
                    lower=df[col].quantile(0.01),
                    upper=df[col].quantile(0.99)
                )
        self.processed_data = df

    def reset(self):
        if self.mode == 'train':
            max_start = len(self.data) - self.config.time_horizon - 1
            self.start_idx = np.random.randint(0, max(1, max_start))
        else:
            self.start_idx = 0

        self.current_step = 0
        self.current_idx = self.start_idx

        self.remaining_inventory = self.config.initial_inventory
        self.executed_qty = 0
        self.executed_value = 0.0
        self.trades = []

        self.temp_impact = 0.0
        self.perm_impact = 0.0

        self.vwap_numerator = 0.0
        self.vwap_denominator = 0.0
        self.twap_target = self.config.initial_inventory / self.config.time_horizon

        self.consecutive_no_trades = 0
        self.last_trade_step = -10
        self.execution_curve = []

        self.price_history.clear()
        self.volume_history.clear()
        self.spread_history.clear()

        for i in range(min(self.config.history_len, self.current_idx)):
            idx = self.current_idx - self.config.history_len + i
            if idx >= 0:
                row = self.processed_data.iloc[idx]
                self.price_history.append(row['mid'])
                self.volume_history.append(row['volume'])
                self.spread_history.append(row['spread_bps'])

        return self._get_state()

    def _get_raw_state(self) -> np.ndarray:
        if self.current_idx >= len(self.processed_data):
            return np.zeros(self.config.state_dim, dtype=np.float32)

        #time
        row = self.processed_data.iloc[self.current_idx]
        time_pct = self.current_step / self.config.time_horizon
        time_remaining = 1 - time_pct
        time_urgency = np.exp(-5 * time_remaining) if time_remaining < 0.3 else 0

        # Inventory
        inv_pct = self.remaining_inventory / self.config.initial_inventory
        exec_pct = self.executed_qty / self.config.initial_inventory
        exec_deviation = exec_pct - time_pct   #comp with TWAP
        behind_schedule = max(0, -exec_deviation)
        ahead_schedule = max(0, exec_deviation)

        if len(self.price_history) > 1:
            price_trend = (self.price_history[-1] - self.price_history[0]) / (self.price_history[0] + 1e-8)
            vol_trend = np.mean(self.volume_history) if self.volume_history else 0
            spread_mean = np.mean(self.spread_history) if self.spread_history else row['spread_bps']
            price_volatility = np.std(self.price_history) / (np.mean(self.price_history) + 1e-8)
        else:
            price_trend = 0; vol_trend = 0; spread_mean = row['spread_bps']; price_volatility = 0

        features = [
            # Time (6)
            time_pct, time_pct ** 2, time_remaining, time_urgency,
            np.sin(2*np.pi*time_pct), np.cos(2*np.pi*time_pct),
            # Inventory (6)
            inv_pct, inv_pct ** 2, exec_pct, np.tanh(exec_deviation * 10),
            behind_schedule, ahead_schedule,
            # Market (12)
            np.tanh(row['spread_bps'] / 20), row['imbalance'],
            np.tanh(row['ret_5']*100) if not np.isnan(row['ret_5']) else 0,
            np.tanh(row['ret_10']*100) if not np.isnan(row['ret_10']) else 0,
            np.tanh(row['ret_20']*100) if not np.isnan(row['ret_20']) else 0,
            np.log1p(row['volume'])/10, np.log1p(row['bid_depth'])/10, np.log1p(row['ask_depth'])/10,
            row['depth_imbalance_ratio'],
            np.tanh(row['vol_5']*100) if not np.isnan(row['vol_5']) else 0,
            np.tanh(row['vol_10']*100) if not np.isnan(row['vol_10']) else 0,
            row['volume_ratio'],
            # Historical (6)
            price_trend, vol_trend/100, spread_mean/20, price_volatility,
            np.tanh((row['mid'] - (np.mean(self.price_history) if self.price_history else row['mid'])) /
                    ((np.std(self.price_history) if self.price_history else 1e-8) + 1e-8)),
            len(self.price_history)/self.config.history_len,
            # Execution history (6)
            min(self.consecutive_no_trades/20, 1),
            min((self.current_step-self.last_trade_step)/50, 1),
            len(self.trades)/(self.config.time_horizon/10),
            self.temp_impact*100, self.perm_impact*100,
            np.tanh((self.executed_value/max(1,self.executed_qty) - row['mid'])/row['mid']) if self.executed_qty>0 else 0,
            # Urgency (6)
            float(time_pct>0.7 and inv_pct>0.3),
            float(time_pct>0.9 and inv_pct>0.1),
            float(inv_pct>0.5 and time_pct>0.5),
            float(self.consecutive_no_trades>20 and inv_pct>0.2),
            float(exec_deviation<-0.1),
            float(exec_deviation>0.1),
        ]
        state = np.array(features[:self.config.state_dim], dtype=np.float32)
        if len(state) < self.config.state_dim:
            state = np.pad(state, (0, self.config.state_dim-len(state)), mode='constant')
        state = np.clip(state, -10, 10)
        return state

    def _get_state(self) -> np.ndarray:
        """ Get current state, normalize if configured """
        raw_state = self._get_raw_state()

        if not self.normalize_states:
            return raw_state

        # Update running statistics during training
        if self.update_state_stats and self.mode == 'train':
            self.state_normalizer.update(raw_state)

        # Normalize the state
        normalized_state = (raw_state - self.state_normalizer.mean) / (self.state_normalizer.std + 1e-8)

        # Clip to reasonable range to prevent extreme values
        normalized_state = np.clip(normalized_state, -5, 5)

        return normalized_state


    # ---------- helper execution logic ----------
    def _taker_fill(self, side: str, size: int, row) -> Tuple[int, float]:
        """ Fill against book L1-L3 as marketable; return (filled_qty, vwap_price) """
        if size <= 0: return 0, 0.0
        filled, cash = 0, 0.0
        if side == 'buy':
            for px, vol in [(row['askPrice1'], row['askVolume1']),
                            (row['askPrice2'], row['askVolume2']),
                            (row['askPrice3'], row['askVolume3']),
                            (row['askPrice4'], row['askVolume4']),
                            (row['askPrice5'], row['askVolume5'])]:
                take = int(min(size - filled, max(0, vol)))
                if take <= 0: continue
                filled += take
                cash += take * px
                if filled >= size: break
        else:
            for px, vol in [(row['bidPrice1'], row['bidVolume1']),
                            (row['bidPrice2'], row['bidVolume2']),
                            (row['bidPrice3'], row['bidVolume3']),
                            (row['bidPrice4'], row['bidVolume4']),
                            (row['bidPrice5'], row['bidVolume5'])]:
                take = int(min(size - filled, max(0, vol)))
                if take <= 0: continue
                filled += take
                cash += take * px
                if filled >= size: break
        vwap = cash/filled if filled>0 else 0.0
        return filled, vwap

    def _passive_fill(self, side: str, size: int, lim_price: float, row) -> Tuple[int, float, int]:
        """
        Passive (post-only) or non-marketable: snap to a level, queue model partial fill.
        Returns (filled_qty, px, picked_level_index) where level_index in {1,2,3}
        """
        if size <= 0: return 0, 0.0, 1
        # Choose level
        if side == 'buy':
            # snap to best bid (L1); if lower than bid2/bid3, snap accordingly
            levels = [(row['bidPrice1'], row['bidVolume1'], 1),
                      (row['bidPrice2'], row['bidVolume2'], 2),
                      (row['bidPrice3'], row['bidVolume3'], 3),
                      (row['bidPrice4'], row['bidVolume4'], 4),
                      (row['bidPrice5'], row['bidVolume5'], 5)]
            # pick the highest bid <= lim_price (can't place above ask; we enforce outside crossing before)
            candidate = None
            for px, vol, idx in levels:
                if lim_price >= px:
                    candidate = (px, vol, idx); break
            if candidate is None:  # lim price below bid3 -> snap to bid3
                candidate = levels[-1]
        else:
            levels = [(row['askPrice1'], row['askVolume1'], 1),
                      (row['askPrice2'], row['askVolume2'], 2),
                      (row['askPrice3'], row['askVolume3'], 3),
                      (row['askPrice4'], row['askVolume4'], 4),
                      (row['askPrice5'], row['askVolume5'], 5)]
            candidate = None
            for px, vol, idx in levels:
                if lim_price <= px:
                    candidate = (px, vol, idx); break
            if candidate is None:  # lim price above ask3 -> snap to ask3
                candidate = levels[-1]

        px, depth, level_idx = candidate

        # Queue model: portion of opposite-side trade volume hits our level
        total_vol = max(100.0, float(row['volume']))  # safeguard
        # assume half flow hits our side, exponential decay by level distance
        hit = 0.5 * total_vol * (0.6 ** (level_idx-1))
        # our share in queue (append at back)
        queue_share = size / (size + max(1.0, float(depth)))
        exp_fill = hit * queue_share
        filled = int(min(size, max(0.0, exp_fill)))
        return filled, float(px), level_idx

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        action = [size_ratio in (0,1), price_offset_bps in [-max,max], post_only in {0,1}]
        """
        self.step_index += 1
        # check if normalize action true from config then
        if getattr(self.config, 'normalize_action', True):
            normalized_action = self.action_normalizer.normalize_action(action)

            size_ratio = normalized_action[0]      # [0, 1]
            price_offset_bps = normalized_action[1]    # [-max_bps, max_bps]
            post_only = normalized_action[2]       # {0, 1}
        else:
            # Parse action
            size_ratio = float(np.clip(action[0], 0.0, 1.0))
            price_offset_bps = float(np.clip(action[1], -self.config.max_price_offset_bps, self.config.max_price_offset_bps))
            post_only = 1 if action[2] >= 0.5 else 0

        time_pct = self.current_step / self.config.time_horizon
        inv_pct = self.remaining_inventory / self.config.initial_inventory

        # Adaptive sizing
        if time_pct < self.config.urgency_threshold:
            base_size = self.config.base_order_pct * self.remaining_inventory
            urgency_multiplier = 1.0
        else:
            progress = (time_pct - self.config.urgency_threshold) / (1 - self.config.urgency_threshold)
            urgency_multiplier = 1 + 3 * (progress ** 2)
            base_size = self.config.base_order_pct * self.remaining_inventory * urgency_multiplier

        max_size = min(self.config.max_order_pct * self.remaining_inventory * urgency_multiplier,
                       self.remaining_inventory)
        order_size = int(base_size + size_ratio * (max_size - base_size))
        if 0 < order_size < self.config.min_order_units:
            order_size = min(self.config.min_order_units, self.remaining_inventory)
        order_size = min(order_size, self.remaining_inventory)

        # Execute
        trade_executed = False
        trade_cost_bps = 0.0
        execution_price = 0.0
        filled_qty = 0

        if order_size > 0 and self.current_idx < len(self.processed_data):
            row = self.processed_data.iloc[self.current_idx]
            self.price_history.append(row['mid'])
            self.volume_history.append(row['volume'])
            self.spread_history.append(row['spread_bps'])

            bid1, ask1 = row['bidPrice1'], row['askPrice1']
            mid, spread = row['mid'], row['spread']
            # proposed limit price: positive offset increases aggressiveness for both sides
            price_adj = (price_offset_bps/10000.0) * mid
            if self.config.side == 'buy':
                lim_price = mid + price_adj
                # if not post-only and lim crosses, act as taker
                if (not post_only) and lim_price >= ask1:
                    filled_qty, execution_price = self._taker_fill('buy', order_size, row)
                else:
                    # ensure not marketable
                    lim_price = min(lim_price, ask1 - 1e-8)
                    filled_qty, px, _ = self._passive_fill('buy', order_size, lim_price, row)
                    execution_price = px if filled_qty>0 else 0.0
            else:
                lim_price = mid - price_adj
                if (not post_only) and lim_price <= bid1:
                    filled_qty, execution_price = self._taker_fill('sell', order_size, row)
                else:
                    lim_price = max(lim_price, bid1 + 1e-8)
                    filled_qty, px, _ = self._passive_fill('sell', order_size, lim_price, row)
                    execution_price = px if filled_qty>0 else 0.0

            # Market impact (small)
            volume = max(100, row['volume'])
            vol_ratio = (filled_qty / volume) if volume>0 else 0.0
            self.temp_impact = (self.temp_impact * self.config.impact_decay +
                                self.config.temp_impact_coef * np.sqrt(max(0.0, vol_ratio)))
            self.perm_impact += self.config.perm_impact_coef * max(0.0, vol_ratio)

            if filled_qty > 0:
                # Apply impact to exec price (mild)
                execution_price *= (1 + self.temp_impact + self.perm_impact)

                # Update state
                self.remaining_inventory -= filled_qty
                self.executed_qty += filled_qty
                self.executed_value += filled_qty * execution_price

                # VWAP benchmark uses mid (neutral)
                self.vwap_numerator += filled_qty * mid
                self.vwap_denominator += filled_qty

                # Cost bps sign: positive = worse than mid (slippage)
                if mid <= 1e-8 or execution_price <= 0:
                    trade_cost_bps = 0.0
                else:
                    if self.config.side == 'buy':
                        trade_cost_bps = 10000.0 * (execution_price - mid) / mid
                    else:
                        trade_cost_bps = 10000.0 * (mid - execution_price) / mid

                trade_executed = True
                self.last_trade_step = self.current_step
                self.consecutive_no_trades = 0
                self.trades.append({
                    'step': self.current_step,
                    'time_pct': time_pct,
                    'size': filled_qty,
                    'price': execution_price,
                    'mid': mid,
                    'cost_bps': trade_cost_bps,
                    'post_only': post_only,
                    'price_offset_bps': price_offset_bps,
                })
            else:
                self.consecutive_no_trades += 1
        else:
            self.consecutive_no_trades += 1
            if self.current_idx < len(self.processed_data):
                row = self.processed_data.iloc[self.current_idx]
                self.price_history.append(row['mid'])
                self.volume_history.append(row['volume'])
                self.spread_history.append(row['spread_bps'])

        self.execution_curve.append(self.executed_qty / self.config.initial_inventory)

        # Advance time
        self.current_step += 1
        self.current_idx += 1

        done = (
            self.current_step >= self.config.time_horizon or
            self.remaining_inventory == 0 or
            self.current_idx >= len(self.data) - 1
        )

        reward = self._compute_reward(order_size, filled_qty, trade_cost_bps, trade_executed, done)

        next_state = self._get_state() if not done else np.zeros(self.config.state_dim, dtype=np.float32)

        info = {
            'current_step': self.step_index,
            'reward': reward,
            'side': self.config.side,
            'remaining_inventory': self.remaining_inventory,
            'executed_qty': self.executed_qty,
            'executed_value': self.executed_value,
            'completion_rate': self.executed_qty / self.config.initial_inventory,
            'num_trades': len(self.trades),
            'order_size': order_size,
            'filled_qty': filled_qty,
            'trade_cost_bps': trade_cost_bps,
            'execution_price': execution_price,
            'trade_executed': trade_executed,
            'time_pct': time_pct,
            'inv_pct': inv_pct
        }
        self.metrics.append(info)

        if self.step_index > 5000:
            self.save_csv(self.out_csv_name, self.run)
            self.run += 1
            self.metrics = []
            self.step_index = 0

        return next_state, reward, done, info

    def _compute_reward(self, order_size, filled_qty, trade_cost_bps, trade_executed, done):
        return self._calc_reward(order_size, filled_qty, trade_cost_bps, trade_executed, done)

    def save_csv(self, out_csv_name, run):
        """
        Save the logged metrics to CSV for later analysis.
        """
        if out_csv_name is not None and len(self.metrics) > 0:
            df = pd.DataFrame(self.metrics)
            df.to_csv(f"{out_csv_name}_run{run}.csv", index=False)

    def _execution_quality_reward(self, trade_cost_bps: float) -> float:
        """Smooth reward for execution quality based on trade cost"""
        # Sigmoid-like function: negative cost (better than mid) gets positive reward
        # Positive cost (worse than mid) gets negative reward with exponential decay
        if trade_cost_bps <= 0:
            return 2.0 * (1 - np.exp(trade_cost_bps / 5.0))  # Positive for negative costs
        else:
            return -np.tanh(trade_cost_bps / 10.0)  # Smooth negative for positive costs

    def _size_appropriateness_reward(self, filled_qty: int, order_size: int,
                                exec_pct: float) -> float:
        """Reward for appropriate order sizing relative to remaining time"""
        target_size = ((1 - exec_pct) * self.config.initial_inventory /
                    max(1, self.config.time_horizon - self.current_step))
        actual_size = filled_qty if filled_qty > 0 else order_size
        size_ratio = actual_size / max(1, target_size)

        # Bell curve centered at 1.0 (optimal sizing)
        return np.exp(-((size_ratio - 1.0) ** 2) / 0.2)

    def _tracking_reward(self, exec_pct: float, time_pct: float) -> float:
        """Smooth reward for tracking the execution schedule"""
        tracking_error = abs(exec_pct - time_pct)
        # Exponential decay with tracking error
        return 2.0 * np.exp(-tracking_error * 20.0)

    def _urgency_reward(self, trade_executed: bool, time_pct: float,
                    exec_pct: float) -> float:
        """Reward for urgency when behind schedule"""
        if time_pct <= 0.5:
            return 0.0

        behind_schedule = max(0, time_pct - exec_pct - 0.1)
        urgency_factor = np.tanh(behind_schedule * 10.0)

        if trade_executed:
            return urgency_factor  # Positive reward for trading when urgent
        else:
            return -urgency_factor  # Negative reward for not trading when urgent

    def _inaction_penalty(self, inv_pct: float) -> float:
        """Smooth penalty for consecutive periods without trading"""
        if inv_pct <= 0.2:
            return 0.0

        inaction_factor = min(self.consecutive_no_trades / 20.0, 1.0)
        return -inv_pct * np.tanh(inaction_factor * 2.0)

    def _completion_reward(self, completion_rate: float) -> float:
        """Smooth completion reward with exponential growth near 100%"""
        if completion_rate >= 1.0:
            return 1.0
        else:
            # Exponential growth as we approach 100% completion
            return np.tanh((completion_rate - 0.8) * 10.0)

    def _vwap_performance_reward(self) -> float:
        """Reward for VWAP performance"""
        if self.executed_qty == 0:
            return 0.0

        avg_px = self.executed_value / self.executed_qty
        vwap = self.vwap_numerator / max(1, self.vwap_denominator)

        if self.config.side == 'buy':
            shortfall_bps = 10000 * (avg_px - vwap) / vwap
        else:
            shortfall_bps = 10000 * (vwap - avg_px) / vwap

        # Sigmoid function: negative shortfall (outperformance) gives positive reward
        return np.tanh(-shortfall_bps / 10.0)

    def _trading_frequency_reward(self) -> float:
        """Reward for appropriate trading frequency"""
        trades_per_100 = len(self.trades) / max(1, self.current_step / 100)
        optimal_freq = 10.0  # Target frequency

        # Bell curve centered at optimal frequency
        freq_deviation = abs(trades_per_100 - optimal_freq)
        return np.exp(-(freq_deviation ** 2) / 50.0)

    def _pv_reward(self) -> float:
        """P&L reward based on mark-to-market"""
        if self.executed_qty == 0:
            return 0.0

        final_mid = self.processed_data.iloc[min(self.current_idx, len(self.processed_data)-1)]['mid']

        if self.config.side == 'buy':
            pv = -self.executed_value + self.executed_qty * final_mid
        else:
            pv = self.executed_value - self.executed_qty * final_mid

        pv_ratio = pv / max(1.0, self.config.initial_inventory * final_mid)
        return np.tanh(pv_ratio * 100.0)

    def _calc_reward(self, order_size: int, filled_qty: int, trade_cost_bps: float,
                    trade_executed: bool, done: bool) -> float:
        """
        Improved reward function with smooth components and convex combination
        """
        time_pct = self.current_step / self.config.time_horizon
        exec_pct = self.executed_qty / self.config.initial_inventory
        inv_pct = self.remaining_inventory / self.config.initial_inventory

        # Base reward components
        rewards = {}

        # 1. Execution quality (when trade occurs)
        if trade_executed:
            rewards['execution_quality'] = self._execution_quality_reward(trade_cost_bps)
            rewards['size_appropriateness'] = self._size_appropriateness_reward(
                filled_qty, order_size, exec_pct)
        else:
            rewards['execution_quality'] = 0.0
            rewards['size_appropriateness'] = 0.0

        # 2. Schedule tracking (always active)
        rewards['tracking'] = self._tracking_reward(exec_pct, time_pct)

        # 3. Urgency handling
        rewards['urgency'] = self._urgency_reward(trade_executed, time_pct, exec_pct)

        # 4. Inaction penalty
        rewards['inaction'] = self._inaction_penalty(inv_pct)

        # 5. Terminal rewards (only when done)
        if done:
            completion_rate = self.executed_qty / self.config.initial_inventory
            rewards['completion'] = self._completion_reward(completion_rate)
            rewards['vwap_performance'] = self._vwap_performance_reward()
            rewards['trading_frequency'] = self._trading_frequency_reward()
            rewards['pv'] = self._pv_reward()
        else:
            rewards['completion'] = 0.0
            rewards['vwap_performance'] = 0.0
            rewards['trading_frequency'] = 0.0
            rewards['pv'] = 0.0

        # Convex combination weights (should sum to 1.0)
        weights = {
            'execution_quality': 0.5,
            'size_appropriateness': 0.02,
            'tracking': 0.10,
            'urgency': 0.23,
            'inaction': 0.0,
            'completion': 0.40,
            'vwap_performance': 0.00,
            'trading_frequency': 0.05,
            'pv': 0.00
        }

        # Apply time-varying weights for terminal rewards
        if done:
            # Boost terminal reward weights
            weights['completion'] *= 45.0
            weights['vwap_performance'] *= 2.0
            weights['trading_frequency'] *= 2.0
            weights['pv'] *= 2.0

            # Renormalize
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate final reward as weighted combination
        final_reward = sum(weights[component] * reward_value
                        for component, reward_value in rewards.items())

        # Scale by base reward and add small base
        final_reward = self.config.base_reward + final_reward * 2.0

        return float(final_reward)
