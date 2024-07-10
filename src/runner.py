import numpy as np


from src.buffer import Episode


class BaseRunner:
    
    
    def __init__(self, env, controller, replay_buffer, args):
        
        # ===== DEPENDENCIES =====
        self.env = env
        self.controller = controller
        self.replay_buffer = replay_buffer
        
        
        # ===== VARIABLES ====
        self.state = self.env.reset()[0]
        self.max_episode_length = args.train_max_episode_length
        self.eval_max_episode_length = args.eval_max_episode_length
        
        # ===== STATISTICS =====
        self.time = 0
        self.sum_rewards = 0
        self.total_num_steps = 0
        

    def close(self):
        """ Closes the underlying environment. Should always when ending an experiment. """
        self.env.close()
        
        
    def _next_state(self):
        pass
    
    def run(self):
        pass


class Runner(BaseRunner):
    
    
    def __init__(self, env, controller, replay_buffer, args=None):
        assert args is not None, "CLASS: Runner. MESSAGE: No parameters to set values with."
        super().__init__(env=env, 
                         controller=controller,
                         replay_buffer=replay_buffer,
                         args=args
                         )
        self.plot_animation = args.plot_animation
        
    def next_state(self, terminated, next_state):
        self.time = 0 if terminated else self.time + 1
        self.total_num_steps += 1
        if terminated:
            self.sum_rewards = 0
            self.state, _ = self.env.reset()
        else:
            self.state = next_state
    
    def run_step(self, action):
        """ Make a step in the environment (and update internal bookeeping) """
        next_state, reward, terminated, truncated, info = self.env.step(action=action)
        self.sum_rewards += reward
        return next_state, reward, terminated, truncated, info  # reward, next state, terminal, done
    
    
    def run(self, n_steps, mode='train', return_dict=None):
        
        self.controller.model.train() if mode == 'train' else self.controller.model.eval()

        episode = Episode()    
        episodes = []    
        time, episode_lengths, episode_rewards = 0, [], []
        
        if mode == 'train':
            max_episode_length = n_steps if n_steps > 0 else self.max_episode_length
        else:
            max_episode_length = self.eval_max_episode_length
            
        increase_counter = True if mode == 'train' else False
            
        for t in range(max_episode_length):

            if self.plot_animation and mode != 'train':
                self.env.render()
                        
            # Choose an action based on current state.
            action = self.controller.choose(observations=self.state, explore_type='mixed', increase_counter=increase_counter)
            next_state, reward, terminated, truncated, info = self._run_step(action=action)
    
            
            episode.add(state=self.state,
                        action=action,
                        reward=reward,
                        terminated=terminated)
            
            if t == (max_episode_length - 1): terminated = True


            if terminated:
                data = episode.get_episode_and_reset()
                episodes.append(data)
                self.replay_buffer.add_episode(data)
                episode_lengths.append(self.time + 1)
                episode_rewards.append(self.sum_rewards)
            
            self._next_state(terminated=terminated, next_state=next_state)

            time += 1
            if terminated:
                break
        
        return_dict = { 
                'episodes': episodes,
                'episode_reward': episode_rewards[0],
                'episode_length': episode_lengths[0],
                'env_steps': time}
        
        
        return return_dict
            
    
    
    def run_episode(self, mode='', return_dict=None):
        return self.run(0, mode=mode, return_dict=return_dict)