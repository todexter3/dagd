import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict
from config import Config
import os

def create_visualization(history: Dict, test_env, config: Config, twap_baseline, is_baseline: Dict, save_dir: str = '.'):
    fig = plt.figure(figsize=(17, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.32, wspace=0.28)

    # 1. Training rewards
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['train_rewards'], alpha=0.3, label='Train')
    ax1.plot(pd.Series(history['train_rewards']).rolling(25).mean(), linewidth=2, label='Train MA(25)')
    if history['valid_rewards']:
        val_x = range(24, len(history['train_rewards']), 25)[:len(history['valid_rewards'])]
        ax1.plot(val_x, history['valid_rewards'], 'ro-', linewidth=2, label='Valid')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_title('Learning Progress (Reward)')
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Total Reward'); ax1.grid(True, alpha=0.3); ax1.legend()

    # 2. Completion
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(np.array(history['train_completion'])*100, alpha=0.3, label='Train')
    ax2.plot(pd.Series(history['train_completion']).rolling(25).mean()*100, linewidth=2, label='Train MA(25)')
    if history['valid_completion']:
        val_x = range(24, len(history['train_completion']), 25)[:len(history['valid_completion'])]
        ax2.plot(val_x, np.array(history['valid_completion'])*100, 'ro-', linewidth=2, label='Valid')
    ax2.axhline(y=99, color='gold', linestyle='--', alpha=0.5, label='99% Target')
    ax2.set_ylim([85, 101])
    ax2.set_title('Order Completion')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Completion (%)'); ax2.grid(True, alpha=0.3); ax2.legend()

    # 3. Costs
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history['train_costs'], alpha=0.3, label='Train')
    ax3.plot(pd.Series(history['train_costs']).rolling(25).mean(), linewidth=2, label='Train MA(25)')
    if history['valid_costs']:
        val_x = range(24, len(history['train_costs']), 25)[:len(history['valid_costs'])]
        ax3.plot(val_x, history['valid_costs'], 'ro-', linewidth=2, label='Valid')
    ax3.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='10 bps Threshold')
    ax3.set_title('Execution Cost (bps)'); ax3.set_xlabel('Episode'); ax3.set_ylabel('Avg Cost (bps)'); ax3.grid(True, alpha=0.3); ax3.legend()

    # 4. Execution curve (test)
    ax4 = fig.add_subplot(gs[1, :2])
    execution_curve = test_env.execution_curve
    if len(execution_curve) > 0:
        t = np.arange(len(execution_curve))/len(execution_curve)
        ec = np.array(execution_curve)
        ax4.plot(t, ec, linewidth=2, label='RL')
        ax4.plot([0,1],[0,1],'k--',alpha=0.5,label='Linear Target')
        ax4.fill_between(t, ec, t, where=(ec>=t), alpha=0.25, label='Ahead')
        ax4.fill_between(t, ec, t, where=(ec<t), alpha=0.25, label='Behind')
    ax4.set_title(f'Test Execution Trajectory (side={config.side.upper()})')
    ax4.set_xlabel('Time'); ax4.set_ylabel('Execution Progress'); ax4.grid(True, alpha=0.3); ax4.legend()

    # 5. RL vs TWAP cost bar
    ax5 = fig.add_subplot(gs[1, 2])
    rl_cost = np.mean([t['cost_bps'] for t in test_env.trades]) if test_env.trades else 0.0
    bar_x = ['RL', 'TWAP', 'IS']
    bar_y = [rl_cost, twap_baseline['avg_cost_bps'], is_baseline['avg_cost_bps']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax5.bar(bar_x, bar_y, color=colors)
    ax5.bar_label(bars, fmt='{:.2f}')
    ax5.axhline(y=0, color='k', lw=1)
    ax5.set_title('Avg Cost vs Baselines (bps)'); ax5.grid(True, axis='y', alpha=0.3)

    # 6. Summary panel
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    if test_env.executed_qty > 0:
        avg_px = test_env.executed_value / test_env.executed_qty
        vwap = test_env.vwap_numerator / max(1, test_env.vwap_denominator)
        if config.side == 'buy':
            shortfall_bps = 10000*(avg_px - vwap)/vwap
        else:
            shortfall_bps = 10000*(vwap - avg_px)/vwap
    else:
        shortfall_bps = 0.0
    final_mid = test_env.processed_data.iloc[min(test_env.current_idx, len(test_env.processed_data)-1)]['mid']
    if config.side == 'buy':
        pv = -test_env.executed_value + test_env.executed_qty * final_mid
    else:
        pv = +test_env.executed_value - test_env.executed_qty * final_mid

    text = f"""
    PERFORMANCE SUMMARY (side={config.side.upper()})
    {'='*75}
    TRAINING:
      Best Valid Reward:      {history['best_valid_reward']:.1f}
      Best Valid Completion:  {history['best_valid_completion']*100:.2f}%
      Final Train Reward:     {history['train_rewards'][-1]:.1f}
      Final Train Completion: {history['train_completion'][-1]*100:.2f}%

    TEST (RL):
      Completion Rate:        {test_env.executed_qty/config.initial_inventory*100:.2f}%
      Avg Trade Cost:         {rl_cost:.2f} bps
      VWAP Shortfall:         {shortfall_bps:.2f} bps
      Portfolio Value:        {pv:.2f}
      Number of Trades:       {len(test_env.trades)}

    BASELINE (TWAP Taker):
      Completion Rate:        {twap_baseline['completion']*100:.2f}%
      Avg Trade Cost:         {twap_baseline['avg_cost_bps']:.2f} bps
      VWAP Shortfall:         {twap_baseline['shortfall_bps']:.2f} bps
      Portfolio Value:        {twap_baseline['pv']:.2f}
      Number of Trades:       {twap_baseline['num_trades']}

    BASELINE (Implementation Shortfall):
      Completion Rate:        {is_baseline['completion']*100:.2f}%
      Implementation Cost:    {is_baseline['avg_cost_bps']:.2f} bps
      Portfolio Value:        {is_baseline['pv']:.2f}
      Number of Trades:       {is_baseline['num_trades']}
    """
    ax6.text(0.02, 0.98, text, fontsize=10, family='monospace', va='top')

    plt.suptitle('RL Order Execution - Performance & Baseline Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 构建完整的保存路径
    output_path = os.path.join(save_dir, 'rl_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[+] Visualization saved to: {output_path}")
