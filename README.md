# RL Order Execution System

A reinforcement learning-based system for optimal order execution in financial markets using PPO (Proximal Policy Optimization).

## Overview

This project implements an intelligent trading agent that learns to execute large orders in financial markets 
while minimizing transaction costs and market impact. The system uses deep reinforcement learning with PPO algorithm 
to develop adaptive execution strategies that outperform traditional methods like TWAP. By processing real-time limit 
order book data and learning from market microstructure, the agent makes sophisticated decisions about order timing, 
sizing, and pricing to achieve optimal execution quality.

### Key Features

- **PPO Algorithm**: Stable on-policy RL algorithm for continuous and discrete action spaces
- **Realistic Market Simulation**: Includes limit order semantics, partial fills, and market impact
- **Mixed Action Space**: Handles continuous (size, price) and discrete (order type) decisions
- **Adaptive Execution**: Increases urgency as time deadline approaches
- **Comprehensive Benchmarking**: Compares against TWAP baseline strategy

## Project Structure

```
.
├── config.py         # Configuration parameters
├── envs.py           # Market environment simulation
├── models.py         # Neural network architectures
├── agents.py         # PPO agent implementation
├── utils.py          # Utility functions
├── visualize.py      # Visualization tools
├── train.py          # Main training script
├── predict.py        # Script for running a trained model
└── README.md         # This file
```

## Requirements

```bash
pip install numpy pandas torch matplotlib
```

## Data Format

The system expects CSV files with the following columns:
- Time series data with bid/ask prices and volumes (L1-L5)
- Required columns: `bidPrice1-5`, `askPrice1-5`, `bidVolume1-5`, `askVolume1-5`, `volume`

Data should be organized as:
```
data/
├── <data_name1>/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── <data_name2>/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
```

## How to Start


### 1. Training

Use the `train.py` script to train a new model. You can specify all parameters via the command line.

**Example:**
```bash
python train.py \
    --data_path data/2021 \
    --side sell \
    --initial_inventory 2000 \
    --time_horizon 300 \
    --n_episodes 2000 \
    --lr 0.0001
```


### 2. Predict

**Example:**
```bash
python predict.py \
    --model_path experiments/2021_inv2000_th300_side_sell/checkpoints/best_model.pth \
    --data_path data/2022/test.csv \
    --output_dir prediction_results/run1 \
    --side sell \
    --initial_inventory 2000 \
    --time_horizon 300
```

### Custom Configuration

You can see all available options by running:
python train.py --help

## Output
```
experiments/
└── <experiment_name>/          # e.g., 2021_inv1000_th500_side_buy
    ├── checkpoints/
    │   ├── best_model.pth
    │   └── checkpoint_ep50.pth
    └── results/
        ├── rl_results.png
        ├── final_test_summary.csv
        └── rl_agent_trades.csv
```

## Performance Metrics

- **Completion Rate**: Percentage of target volume executed
- **Average Cost**: Execution cost in basis points vs mid-price
- **VWAP Shortfall**: Performance vs volume-weighted average price
- **Portfolio Value**: Final P&L considering market movements


