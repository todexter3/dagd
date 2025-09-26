import os
import torch
import numpy as np
import pandas as pd
import argparse
from config import Config
from envs import MarketEnv
from agent import PPOAgent
from utils import simulate_twap_taker, simulate_is_baseline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(args):
    """
    Load a trained model and run it on test data.
    """

    if args.model_path is None:
        data_folder_name = os.path.basename(os.path.normpath(args.data_path_for_model))
        experiment_name = (f"{data_folder_name}_"
                           f"inv{args.initial_inventory}_"
                           f"th{args.time_horizon}_"
                           f"side_{args.side}")
        args.model_path = os.path.join('experiments', experiment_name, 'checkpoints', 'best_model.pth')
        print(f"INFO: --model_path not provided. Using default path based on parameters:")
        print(f"      -> {args.model_path}")

    print(f"\nLoading model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}\n"
                                f"Please ensure the training parameters match or provide the correct path using --model_path.")

    # 回测数据
    print(f"Loading backtest data from: {args.backtest_data_path}")
    if not os.path.exists(args.backtest_data_path):
        raise FileNotFoundError(f"Data file not found at {args.backtest_data_path}")
    test_data = pd.read_csv(args.backtest_data_path)

    # 配置
    config = Config(side=args.side)
    config.initial_inventory = args.initial_inventory
    config.time_horizon = args.time_horizon
    
    # 初始化环境和 Agent
    test_env = MarketEnv(test_data, config, mode='test')
    agent = PPOAgent(config)
    agent.load(args.model_path) 
    agent.network.to(device)
    agent.network.eval() 

    print("\n" + "="*60)
    print("STARTING PREDICTION / BACKTEST")
    print("="*60)

    # 运行回测
    with torch.no_grad():
        state = test_env.reset()
        for _ in range(config.time_horizon):
            action, _ = agent.select_action(state, training=False)
            next_state, _, done, _ = test_env.step(action)
            state = next_state
            if done:
                break
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存交易记录
    trades_df = pd.DataFrame(test_env.trades)
    trades_csv_path = os.path.join(output_dir, "prediction_trades.csv")
    trades_df.to_csv(trades_csv_path, index=False)
    print(f"Prediction trade log saved to: {trades_csv_path}")

    # ... (计算性能指标) ...
    print("\nPrediction finished.")
    print(f"  Completion Rate: {test_env.executed_qty / config.initial_inventory * 100:.2f}%")
    print(f"  Number of Trades: {len(test_env.trades)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained RL agent for order execution.")
    

    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model checkpoint (.pth file). If not provided, it will be inferred from other parameters.")

    parser.add_argument("--backtest_data_path", type=str, required=True, help="Path to the new data CSV file for backtesting.")
    parser.add_argument("--data_path_for_model", type=str, default="data/2021", help="The original data path used for training the model (to infer the model's location).")
    parser.add_argument("--side", type=str, default="buy", choices=['buy', 'sell'], help="Trading side (must match the trained model).")
    parser.add_argument('--initial_inventory', type=int, default=1000, help='Initial inventory (must match the trained model).')
    parser.add_argument('--time_horizon', type=int, default=500, help='Time horizon (must match the trained model).')
    parser.add_argument("--output_dir", type=str, default="prediction_results", help="Directory to save prediction results.")
    
    cli_args = parser.parse_args()
    predict(cli_args)