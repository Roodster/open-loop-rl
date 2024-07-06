from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np 

from src.utils import now
from src.info import Info



class BaseExperiment:
    
    
    def __init__(self):
        pass
    
    def run(self):
        pass
    
   
    
class Experiment(BaseExperiment):


    def __init__(self, args, controller, runner, learner, writer):
        
        # ===== DEPENDENCIES =====
        self.controller = controller
        self.runner = runner
        self.learner = learner
        self.writer = writer
        
        # ===== STATISTICS =====
        self.total_run_time = 0.0
        self.train_info = Info()
        self.eval_info = Info()       
        
        # ===== TRAIN VARIABLES =====
        self.train_n_episodes = args.train_n_episodes
        
        
        # ==== EVAL VARIABLES =====
        self.eval_n_episodes = args.eval_n_episodes
        self.eval_episodes_interval = args.eval_episodes_interval
        
        # ===== PLOT VARIABLES =====
        self.plot_info = args.plot_info
        

    def close(self):
        """ Overrides Experiment.close(). """
        self.runner.close()
        
        
    
    def _learn_from_episode(self):
        
        batch = self.runner.trajectory_buffer.sample_trajectories()
        loss = self.learner.train(batch)
        
        return loss
    
    def run(self):
        
        env_steps = 0 if len(self.train_info.env_steps) == 0 else self.train_info.env_steps[-1]
        
        episodes = tqdm(range(self.train_n_episodes))
        for e in episodes:
            begin_time = now()
            # Run an episode.
            info = self.runner.run_episode()
            
            # Log the info
            env_steps += info['env_steps']

            # Update the model.
            loss = self._learn_from_episode()
            self.total_run_time += (now() - begin_time).total_seconds()

            # Evaluate the model.
            if (e + 1) % self.eval_episodes_interval == 0:
                self.train_info.add_length(info['episode_length'])
                self.train_info.add_reward(info['episode_reward'])
                self.train_info.add_step(env_steps)
                if loss is not None: self.train_info.add_loss(loss)
            
                # Evaluate the model.
                self.evaluate()
                # Save the model to file
                # self.writer.save_model(self.controller.parameters(), episode=e+1)
                # Plot the statistics.
                if self.plot_info:
                    self.plot()
                        
    def evaluate(self):
        
        eval_env_steps = 0 if len(self.eval_info.env_steps) == 0 else self.eval_info.env_steps[-1]
        
        for e in range(self.eval_n_episodes):
            # Run an episode.
            info = self.runner.run_episode()
            
            # Log the info
            eval_env_steps += info['env_steps']
            self.eval_info.add_length(info['episode_length'])
            self.eval_info.add_reward(info['episode_reward'])
            self.eval_info.add_step(eval_env_steps)

    def plot(self, update=True):
        """ Plots logged training results. Use "update=True" if the plot is continuously updated
            or use "update=False" if this is the final call (otherwise there will be double plotting). """
        # Smooth curves

        window = max(int(len(self.train_info.episode_losses) / 50), 10)
        
        if len(self.train_info.episode_losses) < window + 2: return
        train_returns = np.convolve(self.train_info.episode_rewards, np.ones(window)/window, 'valid')
        train_lengths = np.convolve(self.train_info.episode_lengths, np.ones(window)/window, 'valid')
        train_losses = np.convolve(self.train_info.episode_losses, np.ones(window)/window, 'valid')
        train_env_steps = np.convolve(self.train_info.env_steps, np.ones(window)/window, 'valid')
        
        eval_returns = np.convolve(self.eval_info.episode_rewards, np.ones(window)/window, 'valid')
        eval_lengths = np.convolve(self.eval_info.episode_lengths, np.ones(window)/window, 'valid')
        eval_env_steps = np.convolve(self.eval_info.env_steps, np.ones(window)/window, 'valid')

        # Determine x-axis based on samples or episodes
        x_losses = train_env_steps[(len(train_env_steps) - len(train_losses)):]

        # Create plot
        colors = ['b', 'g', 'r']

        fig = plt.gcf()
        fig.set_size_inches(16, 4)
        plt.clf()

        ax = plt.subplot(1, 3, 1)
        plt.title(label="Loss over environmental steps")
        ax.plot(x_losses, train_losses, colors[0])
        ax.set_xlabel('environment steps')
        ax.set_ylabel('loss')

        # Plot the eval returns in the subplot

        ax = plt.subplot(1, 3, 2)
        plt.title(label="Eval Return over environmental steps")
        ax.plot(eval_env_steps, eval_returns, colors[0])
        ax.set_xlabel('environment steps')
        ax.set_ylabel('returns')

        # Plot the eval lengths in the subplot

        ax = plt.subplot(1, 3, 3)
        plt.title(label="Eval Lengths over environmental steps")
        ax.plot(eval_env_steps, eval_lengths, colors[0])
        ax.set_xlabel('environment steps')
        ax.set_ylabel('lengths')

        if update:
            plt.pause(1e-16)
        
        else:
            # save figure
            self.writer.save_plots(fig, "train_ret_rew_loss")
            plt.show()
        
        
        
        
        