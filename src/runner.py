import numpy as np


from src.buffer import Episode


class BaseRunner:
    
    
    def __init__(self, env, controller, replay_buffer, params):
        
        # ===== DEPENDENCIES =====
        self.env = env
        self.controller = controller
        self.replay_buffer = replay_buffer
        
        
        # ===== VARIABLES ====
        self.state = self.env.reset()
        self.max_episode_length = params.max_episode_length
        
        # ===== STATISTICS =====
        self.time = 0
        self.sum_rewards = 0
        

    def close(self):
        """ Closes the underlying environment. Should always when ending an experiment. """
        self.env.close()
    
    def run(self):
        pass


class Runner(BaseRunner):
    
    
    def __init__(self, env, controller, replay_buffer, params=None):
        assert params is not None, "CLASS: Runner. MESSAGE: No parameters to set values with."
        super().__init__(env=env, 
                         controller=controller,
                         replay_buffer=replay_buffer,
                         params=params
                         )
        
    
    
    def run(self, n_steps, return_dict=None):
        
        episode_start, episode_lengths, episode_rewards = 0, [], []
        max_episode_length = n_steps if n_steps > 0 else self.max_episode_length
        
        episode = Episode()
        
        for step in range(max_episode_length):
            
            # Choose an action based on current state.
            action = self.controller.choose()
            
            next_state, reward, done, truncated, info = self._run_step(action=action)
            
            episode.add(state=self.state,
                        action=action,
                        reward=reward,
                        done=done)
            

            if done or step == (max_episode_length - 1):
                self.replay_buffer.add_episode(episode.get_episode_and_reset())
                episode_start = step + 1
                episode_lengths.append(self.time + 1)
                episode_rewards.append(self.sum_rewards)
    
    
    def run_episode(self):
        
        
        
        return { 
                'episode_reward': np.random.uniform(low=-1, high=1),
                'episode_length': np.random.uniform(low=-1, high=1),
                'env_steps': 1}