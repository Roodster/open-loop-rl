










































from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np 

from src.utils import now, set_seed, min_max_normalize
from src.info import Info



class BaseExperiment:
    
    
    def __init__(self):
        pass
    
    def run(self):
        pass
    
   
    
class Experiment(BaseExperiment):


    def __init__(self, args, controller, runner, learner, writer, solver=None):
        
        # ===== DEPENDENCIES =====
        self.args = args
        self.controller = controller
        self.runner = runner
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
        self.grad_updates = args.grad_updates
        self.train_frequency = args.train_frequency
        # ==== EVAL VARIABLES =====
        self.eval_n_episodes = args.eval_n_episodes
        self.eval_episodes_interval = args.eval_episodes_interval
        
        # ===== PLOT VARIABLES =====
        self.plot_info = args.plot_info
        
        # ===== FILE HANDLING =====
        self.save_model_frequency = args.save_model_frequency
                
        if solver is not None:
            self.optimal_actions = self.solver.get_actions()
            

    def close(self):
        """ Overrides Experiment.close(). """
        self.runner.close()
        
    
    def _learn_from_episode(self):
        
        losses = 0
        losses_reward = 0
        losses_value = 0
        for i in range(self.grad_updates):
            batch = self.runner.replay_buffer.sample_trajectories()
        
            loss, loss_rew, loss_val = self.learner.train(batch)
            losses += loss
            losses_reward += loss_rew
            losses_value += loss_val
            
        losses_reward = loss_rew / self.grad_updates
        losses_value = losses_value / self.grad_updates
        losses = losses / self.grad_updates
        
        return losses, losses_reward, losses_value
    
    
    
    def run_environmental_steps(self):
        
        self.writer.save_hyperparameters(self.args)
        
        env_steps = 0 if len(self.train_info.env_steps) == 0 else self.train_info.env_steps[-1]
        pbar = tqdm(range(env_steps, self.train_n_environmental_steps, 1))
        for t in pbar:
            begin_time = now()
            # Run an episode.
            info = self.runner.run(n_steps=1, mode='train')
            # Log the info
            env_steps += info['env_steps']
            
            if (t+1) % self.train_frequency == 0:
            # Update the model.
                loss, loss_rew, loss_val = self._learn_from_episode()
                self.total_run_time += (now() - begin_time).total_seconds()
                pbar.set_description(f"time={self.total_run_time:.5f} steps={env_steps} loss={loss:.5f} epsilon={self.runner.controller.epsilon():.5f}")
            

            # Evaluate the model.
            if (t + 1) % self.eval_episodes_interval == 0:
                self.train_info.episode_lengths = info['episode_length']
                self.train_info.episode_rewards = info['episode_reward']
                self.train_info.env_steps = env_steps
                
                if loss is not None: 
                    self.train_info.episode_losses = loss
                    self.train_info.episode_losses_rew = loss_rew
                    self.train_info.episode_losses_val = loss_val                 
                
                # Evaluate the model.
                self.evaluate()
                
                # Plot the statistics.
                if self.plot_info:
                    self.plot()

            # Save the model to file
            if (t + 1) % self.save_model_frequency == 0:
                self.writer.save_model(self.controller.parameters(), episode=t+1)
                self.writer.save_video(images=self.controller.env.get_video())

            
            

                
        # Save information to file
        self.writer.save_info(self.train_info.get(), name='train')
        self.writer.save_info(self.eval_info.get(), name='eval')
        
        
    
    def run(self):
        
        self.writer.save_hyperparameters(self.args)
        
        env_steps = 0 if len(self.train_info.env_steps) == 0 else self.train_info.env_steps[-1]
        pbar = tqdm(range(self.train_n_episodes))
        for e in pbar:
            begin_time = now()
            # Run an episode.
            info = self.runner.run_episode(mode='train')
            # Log the info
            env_steps += info['env_steps']
            # Update the model.
            loss, loss_rew, loss_val = self._learn_from_episode()
            
            self.total_run_time += (now() - begin_time).total_seconds()

            # Evaluate the model.
            if (e + 1) % self.eval_episodes_interval == 0:
                self.train_info.episode_lengths = info['episode_length']
                self.train_info.episode_rewards = info['episode_reward']
                self.train_info.env_steps = env_steps
                
                if loss is not None: 
                    self.train_info.episode_losses = loss
                    self.train_info.episode_losses_rew = loss_rew
                    self.train_info.episode_losses_val = loss_val                 
                
                # Evaluate the model.
                self.evaluate()
                
                # Plot the statistics.
                if self.plot_info:
                    self.plot()

            # Save the model to file
            if (e + 1) % self.save_model_frequency == 0:
                self.writer.save_model(self.controller.parameters(), episode=e+1)
                self.writer.save_video(images=self.controller.env.get_video())

            
            pbar.set_description(f"time={self.total_run_time:.5f} steps={env_steps} reward={info['episode_reward']:.5f} length={info['episode_length']} loss={loss:.5f} epsilon={self.runner.controller.epsilon():.5f}")

                
        # Save information to file
        self.writer.save_info(self.train_info.get(), name='train')
        self.writer.save_info(self.eval_info.get(), name='eval')
        
    def run(self, mode='steps'):
        
        if mode == 'steps':
            self.run_environmental_steps()
        elif mode == 'episodes':
            self.run_episodes()
        else:
            raise "Experiment.run(): Unknown mode {mode}. Choose between <steps|episodes>."
                        
    def evaluate(self):
        
        if self.solver is not None:
            eval_env_steps = 0 if len(self.eval_info.env_steps) == 0 else self.eval_info.env_steps[-1]
            for e in range(self.eval_n_episodes):
                # Run an episode.
                info = self.runner.run_episode(mode='eval')                
                taken_actions = info['episodes'][0]['actions']

                correct_actions = [1 if taken_actions[i] == [a] else 0 for i, a in enumerate(self.optimal_actions)]


                n_correct_actions = sum(correct_actions)                
                # Log the info
                eval_env_steps += info['env_steps']
                self.eval_info.episode_lengths = n_correct_actions
                self.eval_info.episode_rewards = info['episode_reward']
                self.eval_info.env_steps = eval_env_steps            
            
        else:
            
            eval_env_steps = 0 if len(self.eval_info.env_steps) == 0 else self.eval_info.env_steps[-1]
            for e in range(self.eval_n_episodes):
                # Run an episode.
                info = self.runner.run_episode(mode='eval')
                
                # Log the info
                eval_env_steps += info['env_steps']
                self.eval_info.episode_lengths = info['episode_length']
                self.eval_info.episode_rewards = info['episode_reward']
                self.eval_info.env_steps = eval_env_steps

    def plot(self, update=True):
        """ Plots logged training results. Use "update=True" if the plot is continuously updated
            or use "update=False" if this is the final call (otherwise there will be double plotting). """
        # Smooth curves
        window = max(int(len(self.eval_info.episode_lengths) / 50), 1)       
        
        if len(self.eval_info.episode_lengths) < window + 2: return
        train_returns = np.convolve(self.train_info.episode_rewards, np.ones(window)/window, 'valid')
        train_lengths = np.convolve(self.train_info.episode_lengths, np.ones(window)/window, 'valid')
        train_losses = np.convolve(min_max_normalize(self.train_info.episode_losses), np.ones(window)/window, 'valid')
        train_losses_rew = np.convolve(min_max_normalize(self.train_info.episode_losses_rew), np.ones(window)/window, 'valid')
        train_losses_val = np.convolve(min_max_normalize(self.train_info.episode_losses_val), np.ones(window)/window, 'valid')
        train_env_steps = np.convolve(self.train_info.env_steps, np.ones(window)/window, 'valid')
        
        eval_returns = np.convolve(self.eval_info.episode_rewards, np.ones(window)/window, 'valid')
        eval_lengths = np.convolve(self.eval_info.episode_lengths, np.ones(window)/window, 'valid')
        eval_env_steps = np.convolve(self.eval_info.env_steps, np.ones(window)/window, 'valid')

        # Determine x-axis based on samples or episodes
        # x_losses = train_env_steps[(len(train_env_steps) - len(train_losses)):]
        x_losses = train_env_steps
        # Create plot
        colors = ['b', 'g', 'r']

        n_plots = 4
        fig = plt.gcf()
        fig.set_size_inches(16, 4)
        plt.clf()

        # Plot the eval returns in the subplot

        ax = plt.subplot(1, n_plots, 1)
        plt.title(label="Training Returns")
        ax.plot(train_env_steps, train_returns, colors[0])
        ax.set_xlabel('environment steps')
        ax.set_ylabel('returns')

        # Plot the eval lengths in the subplot

        # ax = plt.subplot(1, n_plots, 2)
        # plt.title(label="Train Lengths")
        # ax.plot(train_env_steps, train_lengths, colors[0])
        # ax.set_xlabel('environment steps')
        # ax.set_ylabel('lengths')
        
        ax = plt.subplot(1, n_plots, 2)
        plt.title(label="Training Loss")
        ax.plot(x_losses, train_losses_rew, colors[1], label='loss_rew')
        ax.plot(x_losses, train_losses_val, colors[2], label='loss_val')
        ax.plot(x_losses, train_losses, colors[0], label='loss')
        ax.legend(loc='upper right')
        ax.set_xlabel('environment steps')
        ax.set_ylabel('loss')


        # Plot the eval lengths in the subplot

        ax = plt.subplot(1, n_plots, 3)
        plt.title(label="Eval Returns")
        ax.plot(eval_env_steps, eval_returns, colors[0])
        ax.set_xlabel('environment steps')
        ax.set_ylabel('returns')
        
            # Plot the eval lengths in the subplot

        ax = plt.subplot(1, n_plots, 4)
        plt.title(label="Eval Lengths")
        ax.plot(eval_env_steps, eval_lengths, colors[0])
        ax.set_xlabel('environment steps')
        ax.set_ylabel('lengths')


        if update:
            plt.pause(1e-16)
        
        else:
            # save figure
            self.writer.save_plots(fig, "train_ret_rew_loss")
            plt.show()
        
        
        
        
        