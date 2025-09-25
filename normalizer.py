import numpy as np
import torch
from typing import Union, Tuple, Dict, Any


class ActionNormalizer:
    """
    Normalizes actions from agent's output space [-1, 1] to environment's native action space.
    
    For your trading environment:
    - Continuous dim 0: size_ratio ∈ [0, 1] 
    - Continuous dim 1: price_offset_bps ∈ [-max_bps, max_bps]
    - Discrete dim 2: post_only ∈ {0, 1}
    """
    
    def __init__(self, max_price_offset_bps: float = 100.0):
        self.max_price_offset_bps = max_price_offset_bps
        
        # Action space bounds for continuous dimensions
        self.continuous_low = np.array([0.0, -max_price_offset_bps], dtype=np.float32)
        self.continuous_high = np.array([1.0, max_price_offset_bps], dtype=np.float32)
        
        # For discrete dimension, we'll use sigmoid threshold
        self.discrete_threshold = 0.0  # sigmoid(0) = 0.5
    
    def normalize_action(self, raw_action: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert agent's raw action [-1, 1] to environment's action space.
        
        Args:
            raw_action: [batch_size, 3] or [3] - raw actions from agent in [-1, 1]
                       - raw_action[..., 0]: size ratio (will be mapped to [0, 1])
                       - raw_action[..., 1]: price offset (will be mapped to [-max_bps, max_bps])  
                       - raw_action[..., 2]: post_only logit (will be mapped to {0, 1})
        
        Returns:
            normalized_action: Same shape as input, normalized to environment space
        """
        is_torch = isinstance(raw_action, torch.Tensor)
        
        if is_torch:
            action = raw_action.clone()
            sigmoid = torch.sigmoid
            tanh = torch.tanh
        else:
            action = raw_action.copy()
            sigmoid = self._sigmoid_np
            tanh = np.tanh
        
        # Continuous dimensions: map [-1, 1] to target ranges
        # Size ratio: [-1, 1] → [0, 1] using sigmoid-like transformation
        action[..., 0] = (tanh(action[..., 0]) + 1.0) / 2.0  # [-1,1] → [0,1]
        
        # Price offset: [-1, 1] → [-max_bps, max_bps] using tanh (already in [-1,1])
        action[..., 1] = tanh(action[..., 1]) * self.max_price_offset_bps
        
        # Discrete dimension: convert logit to binary
        action[..., 2] = (action[..., 2] > self.discrete_threshold).float() if is_torch else \
                        (action[..., 2] > self.discrete_threshold).astype(np.float32)
        
        return action
    
    def denormalize_action(self, normalized_action: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert environment's action back to agent's [-1, 1] space.
        Useful for logging or analysis.
        
        Args:
            normalized_action: Action in environment's native space
            
        Returns:
            raw_action: Action in agent's [-1, 1] space
        """
        is_torch = isinstance(normalized_action, torch.Tensor)
        
        if is_torch:
            action = normalized_action.clone()
            atanh = torch.atanh
            clamp = torch.clamp
        else:
            action = normalized_action.copy() 
            atanh = np.arctanh
            clamp = np.clip
        
        # Reverse the transformations
        # Size ratio: [0, 1] → [-1, 1]
        size_norm = clamp(action[..., 0] * 2.0 - 1.0, -0.999, 0.999)  # Avoid atanh singularities
        action[..., 0] = atanh(size_norm)
        
        # Price offset: [-max_bps, max_bps] → [-1, 1]  
        price_norm = clamp(action[..., 1] / self.max_price_offset_bps, -0.999, 0.999)
        action[..., 1] = atanh(price_norm)
        
        # Discrete: {0, 1} → logit space (approximate)
        # Map 0 → -1, 1 → +1 
        action[..., 2] = 2.0 * action[..., 2] - 1.0
        
        return action
    
    @staticmethod
    def _sigmoid_np(x):
        """Numerically stable sigmoid for numpy."""
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Return information about the normalized action space."""
        return {
            'continuous_dims': 2,
            'discrete_dims': 1,
            'continuous_low': self.continuous_low.tolist(),
            'continuous_high': self.continuous_high.tolist(),
            'discrete_values': [0, 1],
            'agent_action_space': '[-1, 1] for all dimensions'
        }
    
    def clip_actions(self, actions: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Clip raw agent actions to [-1, 1] range before normalization.
        """
        if isinstance(actions, torch.Tensor):
            return torch.clamp(actions, -1.0, 1.0)
        else:
            return np.clip(actions, -1.0, 1.0)


# Example usage and testing
if __name__ == "__main__":
    normalizer = ActionNormalizer(max_price_offset_bps=100.0)
    
    # Test with numpy
    raw_actions = np.array([
        [0.0, 0.0, 0.5],    # neutral actions  
        [-1.0, -1.0, -1.0], # extreme negative
        [1.0, 1.0, 1.0],    # extreme positive
    ])
    
    print("Raw actions (agent output):")
    print(raw_actions)
    
    normalized = normalizer.normalize_action(raw_actions)
    print("\nNormalized actions (environment input):")
    print(normalized)
    
    denormalized = normalizer.denormalize_action(normalized)
    print("\nDenormalized actions (back to agent space):")
    print(denormalized)
    
    print("\nAction space info:")
    print(normalizer.get_action_space_info())