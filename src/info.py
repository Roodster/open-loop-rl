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
            self._episode_optimal_steps_count = []
    
    def _parse_file(self, file):
        pass
    
    def get(self):
        
        
        info = {}
        
        print('len(self._episode_rewards)', len(self._episode_rewards))
        if len(self._episode_rewards) > 0:
            info['rewards'] = self._episode_rewards,
        
        print('len(self._episode_lengths)', len(self._episode_lengths))
        if len(self._episode_lengths) > 0:
            info['lengths'] = self._episode_lengths,
        
        
        print('len(self._env_steps)', len(self._env_steps))
        if len(self._env_steps) > 0:
            info['steps'] = self._env_steps        


        print('len(self._episode_losses)', len(self._episode_losses))
        print('len(self._episode_losses_rew)', len(self._episode_losses_rew))
        print('len(self._episode_losses_val)', len(self._episode_losses_val))
        if len(self._episode_losses) > 0:
            info['losses'] = self._episode_losses
            info['loss_rew'] = self._episode_losses_rew
            info['loss_val'] = self._episode_losses_val

        print('len(self._episode_optimal_steps_count)', len(self._episode_optimal_steps_count))
        if len(self._episode_optimal_steps_count) > 0:
            info['optimal_steps'] = self._episode_optimal_steps_count
            
        print(info)
            
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

    @property
    def episode_optimal_steps_count(self):
        return np.array(self._episode_optimal_steps_count)

    @episode_optimal_steps_count.setter
    def episode_optimal_steps_count(self, value):
        self._episode_optimal_steps_count.append(value)
