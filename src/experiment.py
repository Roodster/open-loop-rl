from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np 
from time import sleep

from src.utils import now, set_seed, min_max_normalize, count_consecutive_equal_vectorized
from src.info import Info
from src.buffer import Episode



class BaseExperiment:
    
    
    def __init__(self):
        pass
    
    def run(self):
        pass
    
   
    
class Experiment(BaseExperiment):


    def __init__(self, args, controller, runner, runner_eval, learner, writer, solver=None):
        
        # ===== DEPENDENCIES =====
        self.args = args
        self.controller = controller
        self.runner = runner
        self.runner_eval = runner_eval
        self.learner = learner
        self.writer = writer
        self.solver = solver
        
        # ===== STATISTICS =====
        self.total_run_time = 0.0
        self.train_info = Info()
        self.eval_info = Info()       
        
        # ===== TRAIN VARIABLES =====
        self.train_n_episodes = args.train_n_episodes
        self.train_n_environmental_steps = args.train_n_environmental_steps
        self.train_max_episode_length = args.train_max_episode_length
        self.grad_updates = args.grad_updates
        self.train_frequency = args.train_frequency
        self.initial_collection_steps = 128
        
        # ==== EVAL VARIABLES =====
        self.eval_n_episodes = args.eval_n_episodes
        self.eval_episodes_interval = args.eval_episodes_interval
        self.eval_max_episode_length = args.eval_max_episode_length
        
        # ===== PLOT VARIABLES =====
        self.plot_info = args.plot_info
        self.plot_animation = args.plot_animation
        
        # ===== FILE HANDLING =====
        self.save_model_frequency = args.save_model_frequency
                
        if solver is not None:
            self.optimal_actions = self.solver.get_actions()
            
        self.writer.save_hyperparameters(self.args)
            

    def close(self):
        """ Overrides Experiment.close(). """
        self.runner.close()
            
    def _learn_from_episode(self):
        batch = self.runner.replay_buffer.sample_trajectories()
        
        loss, loss_rew, loss_val = self.learner.train(batch)
                
        return loss, loss_rew, loss_val
        
    
    def run(self):
        """
            METHOD: 
                run
            DESCRIPTION:
                Runs over specified environmental steps:
                - takes a step
                - adds info to trajectory buffer
                - learns from step
                    - samples from trajectory buffer
                - evaluates
        """
        
        episode = Episode()
        env_steps = 0 if len(self.train_info.env_steps) == 0 else self.train_info.env_steps[-1]
        pbar = tqdm(range(env_steps, self.train_n_environmental_steps))
        
        
        # Collect trajectories using random policy.
        for step in range(1, self.initial_collection_steps + 1):
 
            # choose an action
            action = self.controller.choose(observations=self.runner.state, explore_type='all')
            # take a step
            next_state, reward, terminated, truncated, info = self.runner.run_step(action=action)
            
            
            # add info to episode
            episode.add(state=self.runner.state,
                        action=action,
                        reward=reward,
                        terminated=terminated
                        )
            
            if step == self.initial_collection_steps: terminated = True
            
            
            # if last step add episode to trajectory buffer and reset episode
            if terminated:
                data = episode.get_episode_and_reset()
                self.runner.replay_buffer.add_episode(data)
                
            self.runner.next_state(terminated=terminated, next_state=next_state)
                    
        last_episode = 0

        for step in pbar:

           
            # if self.plot_animation:
            #     self.runner.env.render()
            
            # choose an action
            action = self.controller.choose(observations=self.runner.state, explore_type='all', increase_counter=True)
            # take a step
            next_state, reward, terminated, truncated, info = self.runner.run_step(action=action)

            # add info to episode
            episode.add(state=self.runner.state,
                        action=action,
                        reward=reward,
                        terminated=terminated
                        )
            
            if last_episode + self.train_max_episode_length <= step:
                last_episode = step
                terminated = True
                       
            # if last step add episode to trajectory buffer and reset episode
            if terminated:
                data = episode.get_episode_and_reset()
                self.runner.replay_buffer.add_episode(data)
        
            # learn from the experience
            if (step+1) % self.train_frequency == 0:
                loss, loss_rew, loss_val = self._learn_from_episode()
                
        
                # every so often evaluate the model
                if (step+1) % self.eval_episodes_interval == 0:
                    self.train_info.env_steps = step
                    self.train_info.episode_losses = loss
                    self.train_info.episode_losses_rew = loss_rew
                    self.train_info.episode_losses_val = loss_val
                    self.train_info.episode_rewards = np.sum(data['rewards'])
                    n_optimal_steps = self.evaluate()
                    self.eval_info.env_steps = step
                    self.eval_info.episode_optimal_steps_count = n_optimal_steps

                    
                    pbar.set_description(f'loss={loss} loss_rew={loss_rew}, loss_val={loss_val} optimal_steps={n_optimal_steps}')
                    if self.plot_info:
                        self.plot()
                                        
            self.runner.next_state(terminated=terminated, next_state=next_state)


    def evaluate(self):
        
        for step in range(self.eval_max_episode_length):

            # choose an action
            if self.runner_eval.env is not None and self.plot_animation == True:
                self.runner_eval.env.render()

            action = self.controller.choose(observations=self.runner_eval.state, explore_type='exploit')
            
            next_state, reward, terminated, truncated, info = self.runner_eval.run_step(action=action)
            
            if self.optimal_actions[step] != action:
                terminated = True
                
            self.runner_eval.next_state(terminated=terminated, next_state=next_state)

            if terminated == True:
                break

        self.writer.save_video(self.runner_eval.env.get_video())

        return step
        
    
    def plot(self, update=True):
        """ Plots logged training results. Use "update=True" if the plot is continuously updated
            or use "update=False" if this is the final call (otherwise there will be double plotting). """
        # Smooth curves

        window = max(int(len(self.train_info.env_steps) / 50), 10)
        

        if len(self.train_info.env_steps) < window + 2: return
        rewards = np.convolve(self.train_info.episode_rewards, np.ones(window)/window, 'valid')
        losses = np.convolve(self.train_info.episode_losses, np.ones(window)/window, 'valid')
        losses_rew = np.convolve(self.train_info.episode_losses_rew, np.ones(window)/window, 'valid')
        losses_val = np.convolve(self.train_info.episode_losses_val, np.ones(window)/window, 'valid')

        env_steps = np.convolve(self.train_info.env_steps, np.ones(window)/window, 'valid')
        optimal_steps_count = np.convolve(self.eval_info.episode_optimal_steps_count, np.ones(window)/window, 'valid')

        # Determine x-axis based on samples or episodes
        # Create plot
        colors = ['b', 'g', 'r']


        n_plots = 3
        fig = plt.gcf()
        fig.set_size_inches(16, 4)
        plt.clf()

        # Plot the losses in the left subplot
        plt.subplot(1, n_plots, 1)
        plt.title(label="Train Reward ")
        plt.step(env_steps, rewards, colors[0])
        plt.xlabel('environment steps')
        plt.ylabel('reward')
        # ax.plot(env_steps, lengths, colors[0])
        # ax.set_xlabel('environment steps' if self.plot_train_samples else 'episodes')
        # ax.set_ylabel('episode length')
        # # Plot the losses in the right subplot
        ax = plt.subplot(1, n_plots, 2)
        plt.title(label="Train loss")
        ax.plot(env_steps, losses, colors[0])
        ax.plot(env_steps, losses_rew, colors[0])
        ax.plot(env_steps, losses_val, colors[0])

        ax.set_xlabel('environment steps')
        ax.set_ylabel('loss')

        # Plot the losses in the left subplot
        plt.subplot(1, n_plots, 3)
        plt.title(label="Eval optimal steps")
        plt.step(env_steps, optimal_steps_count, colors[0])
        plt.xlabel('environment steps')
        plt.ylabel('steps')
        # # Plot the episode lengths in the middle subplot
        # ax = plt.subplot(1, 3, 2)
        # plt.title(label="Length over environmental steps")



        if update:
            plt.pause(1e-3)
        
        else:
            # save figure
            self.writer.save_plots(fig, "train_ret_rew_loss")
            plt.show()

