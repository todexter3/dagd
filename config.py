from dataclasses import dataclass


@dataclass
class Config:
    # --- Environment ---
    initial_inventory: int = 300
    time_horizon: int = 500
    side: str = 'buy'   # 'buy' or 'sell'

    # Order sizing
    base_order_pct: float = 0.01
    max_order_pct: float = 0.20
    min_order_units: int = 10
    urgency_threshold: float = 0.7

    # Market impact (small but kept)
    temp_impact_coef: float = 0.00002
    perm_impact_coef: float = 0.000008
    impact_decay: float = 0.98

    # Features
    state_dim: int = 42
    action_dim: int = 3           # [size_ratio, price_offset_bps, post_only]
    history_len: int = 10

    # Network
    hidden_dim: int = 256
    n_layers: int = 3
    dropout: float = 0.1
    activation: str = 'leaky_relu'

    # PPO
    lr_actor: float = 3e-5
    lr_decay: float = 0.9995
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_beta: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_value_clipping: bool = True
    normalize_states: bool = True
    normalize_action: bool = True

    # Training
    batch_size: int = 128
    n_epochs_per_update: int = 4
    buffer_size: int = 2048

    # Exploration (continuous)
    initial_std_size: float = 0.25   # std for size_ratio
    initial_std_bps: float = 10.0    # std for price_offset_bps (in bps)
    min_std: float = 0.05

    # Price control
    max_price_offset_bps: float = 50.0   # |offset| cap

    # Reward shaping
    base_reward: float = 1.0
    completion_bonus: float = 40.0
    vwap_bonus: float = 20.0
    cost_penalty_scale: float = 0.5
    pv_weight: float = 0.0   # keep 0 for stability; set small (e.g., 0.01) if you want PV in reward

    # Normalization
    use_reward_normalization: bool = True
    use_advantage_normalization: bool = True

    # Early stopping
    patience: int = 50
    min_improvement: float = 0.01
