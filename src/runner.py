import numpy as np


from src.buffer import Episode


class BaseRunner:
    
    
    def __init__(self, env, controller, replay_buffer, args):
        
        # ===== DEPENDENCIES =====
        self.env = env
        self.controller = controller
        self.replay_buffer = replay_buffer
        self.prev_actions = []

        
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
        
    def set_next_state(self, terminated, next_state, is_masked=False, action=None,  mode='eval', reset_mode='initial'):
        self.time = 0 if terminated else self.time + 1
        self.total_num_steps += 1
        if terminated:
            self.sum_rewards = 0
            self.state, _ = self.env.reset(options=reset_mode)
        else:
            self.state = next_state
    
    def run_step(self, action):
        """ Make a step in the environment (and update internal bookeeping) """
        next_state, reward, terminated, truncated, info = self.env.step(action=action)
        self.sum_rewards += reward
        return next_state, reward, terminated, truncated, info  # reward, next state, terminal, done
    
    

class OpenLoopRunner(Runner):
    
    
    def __init__(self, env, controller, replay_buffer, args=None):
        assert args is not None, "CLASS: Runner. MESSAGE: No parameters to set values with."
        super().__init__(env=env, 
                         controller=controller,
                         replay_buffer=replay_buffer,
                         args=args
                         )
        self.open_loop_trajectory_probability = args.train_open_loop_probability
        self.trajectory_length = args.trajectory_length
        self.num_blind_steps = 0
        self.total_open_steps = 0
        self.total_closed_steps = 0
        
    def set_next_state(self, terminated, next_state, is_masked=False, action=None, mode='eval', reset_mode='initial'):
        self.time = 0 if terminated else self.time + 1
        self.total_num_steps += 1
        if mode == 'default':
            if terminated:
                self.sum_rewards = 0
                self.prev_actions = []
                self.state, _ = self.env.reset(options=reset_mode)
            else:
                if self.open_loop_trajectory_probability > 0.0:
                    if self.num_blind_steps > 0:
                        self.prev_actions.append(action)
                        self.num_blind_steps -= 1
                        self.total_open_steps += 1
                    elif np.random.uniform(0, 1) < (self.open_loop_trajectory_probability * (1 / self.trajectory_length)):
                        self.prev_actions.append(action)
                        self.num_blind_steps = (self.trajectory_length - 1)
                        self.total_open_steps += 1
                    else:
                        self.state = next_state
                        self.prev_actions = []
                        self.total_closed_steps += 1
                else:
                    self.state = next_state
                
        elif mode == 'eval':
            
            if terminated:
                self.sum_rewards = 0
                self.prev_actions = []
                self.state, _ = self.env.reset(options=reset_mode)
            else:
                if self.open_loop_trajectory_probability > 0.0:
                    if is_masked and action is not None:
                        self.prev_actions.append(action)
                        self.total_open_steps += 1
                    else:
                        self.prev_actions = []
                        self.state = next_state
                        self.total_closed_steps += 1
    
