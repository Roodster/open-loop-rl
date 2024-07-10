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
        self.save_model_frequency = self.config.get('save_model_frequency', 100)
        
        # ===== MODEL =====
        self.model = self.config.get("model", "lstm")
        self.hidden_size = self.config.get("hidden_size", 42)
        self.device = th.device('cuda' if th.cuda.is_available() and self.config.get('device') in ('auto', 'cuda') else 'cpu')
        
        # ===== EXPERIMENT =====
        self.seed = self.config.get("seed", 1)
        

        # ===== EXPLORATION =====
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_finish = self.config.get('epsilon_finish', 0.1)
        self.anneal_time = self.config.get('anneal_time', 100)
        self.exploration_fraction = 0.6
        
        # ===== ENVIRONMENT ==== 
        self.maze_id =  self.config.get("maze_id", 1)
        self.n_envs = self.config.get("n_envs", 1)    
        
        # ===== TRAJECTORY BUFFER =====
        self.buffer_size = self.config.get('buffer_size', 1000)
        self.use_last_episode = self.config.get('use_last_episode', True)
        self.batch_size = self.config.get('batch_size', 128)
        self.trajectory_length  = self.config.get('trajectory_length', 5)
        
        # ===== LEARNING ===== 
        self.train_max_episode_length = self.config.get('train_max_episode_length', 150)
        self.train_n_episodes = self.config.get('train_n_episodes', 100)
        self.train_n_environmental_steps = self.config.get('train_n_environmental_steps', 50000)
        self.train_open_loop_probability = self.config.get('train_open_loop_probability', 0.0)
        self.train_learning_starts = self.config.get('train_learning_starts', 1000)
        self.gamma = self.config.get('gamma', 0.99)
        self.learning_rate = self.config.get('learning_rate', 1e-2)
        self.target_model = self.config.get('target_model', True)
        self.target_update = self.config.get('target_update', 'copy')
        self.target_update_interval = self.config.get('target_update_interval', 100)
        self.soft_target_update_factor = self.config.get('soft_target_update_factor', 0.2)
        self.grad_updates = self.config.get('grad_updates', 1)
        self.train_frequency = self.config.get('train_frequency', 10)
        self.train_start = self.config.get('train_start', 1000)
        
        # ===== EVALUATION =====
        self.eval_n_episodes = self.config.get('eval_n_episodes', 1)
        self.eval_episodes_interval = self.config.get('eval_episodes_interval', 10)
        self.eval_mask_regions = self.config.get('eval_mask_regions', False)
        self.eval_mask_indices = self.config.get('eval_mask_indices', [])
        self.eval_max_episode_length = self.config.get('eval_max_episode_length', 100)
        
        # ===== PLOTTING =====
        self.plot_info = self.config.get('plot_info', True)
        self.plot_animation = self.config.get('plot_animation', False)
        
    def set_env(self, env):
        self.action_space = env.action_space
        self.n_actions = env.action_space.n
        self.observation_space = env.observation_space
    
    def default(self):
        return self.__dict__