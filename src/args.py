"""
    Params:
        Reads parameter file and returns arguments in class format.
"""
import torch as th
from src.utils import parse_args

class Args():
    
    def __init__(self, file):
        # ===== Get the configuration from file =====
        self.config  = parse_args(file)
        
        # ===== METADATA =====
        self.exp_name = self.config.get('exp_name', "dqn")

        # ===== FILE HANDLING =====
        self.log_dir =  self.config.get("log_dir", "./logs")
        
        # ===== MODEL =====
        self.model = self.config.get("model", "lstm")
        self.hidden_size = self.config.get("hidden_size", 42)
        self.device = th.device('cuda' if th.cuda.is_available() and self.config.get('device') in ('auto', 'cuda') else 'cpu')
        
        # ===== EXPERIMENT =====
        self.seed = self.config.get("seed", 1)
        self.n_actions_per_sequence = self.config.get('n_actions_per_sequence', 1)
        self.max_episode_length = self.config.get('max_episode_length', 100)
        

        # ===== EXPLORATION =====
        
        # ===== ENVIRONMENT ==== 
        self.maze_id =  self.config.get("maze_id", 1)
        
        # ===== TRAJECTORY BUFFER =====
     
        
        # ===== LEARNING ===== 
        self.train_n_episodes = self.config.get('train_n_episodes', 100)
        self.train_open_loop_probability = self.config.get('train_open_loop_probability', 0.0)
        
        # ===== EVALUATION =====
        self.eval_n_episodes = self.config.get('eval_n_episodes', 1)
        self.eval_episodes_interval = self.config.get('eval_episodes_interval', 10)
        self.eval_mask_regions = self.config.get('eval_mask_regions', False)
        self.eval_mask_indices = self.config.get('eval_mask_indices', [])
        
        # ===== PLOTTING =====
        self.plot_info = self.config.get('plot_info', True)