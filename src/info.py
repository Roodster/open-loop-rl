import pandas as pd
import numpy as np






























class Info:
    
    def __init__(self, file=None):
        
        if file is not None:
            self._parse_file(file)    
        else: 
            self._episode_lengths = []
            self._episode_rewards = []
            self._episode_losses = []
            self._episode_losses_rew = []
            self._episode_losses_val = []
            self._env_steps = []
    
    def _parse_file(self, file):
        pass
    
    def get(self):
        
        
        info = {
            'rewards': self._episode_rewards,
            'lengths': self._episode_lengths,
            'steps': self._env_steps
        }
        
        if len(self._episode_losses) > 0:
            info['losses'] = self._episode_losses
            info['loss_rew'] = self._episode_losses_rew
            info['loss_val'] = self._episode_losses_val
            
        return pd.DataFrame(info)
        
    def add_step(self, step):
        self._env_steps.append(step)
    
    @property
    def episode_lengths(self):
        return np.array(self._episode_lengths)
    
    @episode_lengths.setter
    def episode_lengths(self, value):
        self._episode_lengths.append(value)


    @property
    def episode_rewards(self):
        return np.array(self._episode_rewards)
    
    @episode_rewards.setter
    def episode_rewards(self, value):
        self._episode_rewards.append(value)

    @property
    def env_steps(self):
        return np.array(self._env_steps)

    @env_steps.setter
    def env_steps(self, value):
        self._env_steps.append(value)

    @property
    def episode_losses(self):
        return np.array(self._episode_losses)


    @episode_losses.setter
    def episode_losses(self, value):
        self._episode_losses.append(value)


    @property
    def episode_losses_rew(self):
        return np.array(self._episode_losses_rew)

    @episode_losses_rew.setter
    def episode_losses_rew(self, value):
        self._episode_losses_rew.append(value)

    
    @property
    def episode_losses_val(self):
        return np.array(self._episode_losses_val)

    @episode_losses_val.setter
    def episode_losses_val(self, value):
        self._episode_losses_val.append(value)
