"""
    Writer:
        Writes data to files.
"""
import os
import torch as th
import matplotlib.pyplot as plt
import json
import datetime as dt
import yaml

class Writer():
    """
        Based on folder structure:
            ./logs
                /run_<datetime>_<exp_name>_<model>_<maze_id>_<actionstep>_<open_loop_probability>
                    /seed_<seed>
                    ...
                    /seed_<seed+x>
                        /models
                            exp_<expname>_model_<model>_maze_<maze_id>_<actionstep>_<episode>.pickle
                        stats.pickle
                        plot_<exp_name>_<model>_<maze_id>_<attribute>.png
                        render.gif
                    hyperparameters.txt
                    /evaluation
                        /eval_<eval_name>_<datetime>
                            stats.pickle
                            plot.png
                            parameters.txt


    """
        
    def __init__(self, args):
        
        self.args = args
        self.datetime = str(dt.datetime.now().strftime("%d%m%Y%H%M"))
        self.root = args.log_dir
        self.base_dir = self.root + f"/run_{self.datetime}_{args.exp_name}_{args.model}_{args.maze_id}_{args.n_actions_per_sequence}_{args.train_open_loop_probability}"
        self.train_dir = self.base_dir + f"/seed_{args.seed}"
        self.eval_dir = self.base_dir + f"/evaluation_{args.exp_name}"
        self.model_dir = self.train_dir + "/models"
        
        self._create_directories(self.base_dir)
        self._create_directories(self.train_dir)
        self._create_directories(self.eval_dir)
        self._create_directories(self.model_dir)

    def _create_directories(self, path):
        os.makedirs(path, exist_ok=True)
        
    
    def save_model(self, model, episode):
        
        _dir = os.path.join(self.model_dir)
        file = f"/model_{self.datetime}_{self.args.exp_name}_{self.args.model}_{self.args.maze_id}_{self.args.n_actions_per_sequence}_{self.args.train_open_loop_probability}_{episode}.pickle"

        full_path = os.path.join(_dir, file)
        
        th.save(model.state_dict(), full_path)
    
    def save_plots(self, plot, attribute):
        filepath = f"/plot_{self.args.exp_name}_{self.args.model}_maze_{self.args.maze_id}_{attribute}.png"
        
        plot_path = os.path.join(self.train_dir, filepath)
        
        plot.savefig(plot_path)
        pass
    
    def save_render(self, render_object, attribute):
        filepath = f"/render_{self.args.exp_name}_{self.args.model}_maze_{self.args.maze_id}_{attribute}.gif"
        
        render_path = os.path.join(self.train_dir, filepath)
        
        render_object.save(render_path, save_all=True, append_images=render_object[1:], duration=100, loop=0)
    
    
    def save_statistics(self, statistics):
        filepath = f"/stats_{self.args.exp_name}_{self.args.model}_maze_{self.args.maze_id}.json"
        stats_path = os.path.join(self.train_dir, filepath)
        with open(stats_path, 'w') as f:
            json.dump(statistics, f)
            
    def save_hyperparameters(self, hyperparameters):
        filepath = f"/hyperparameters_{self.args.exp_name}_{self.args.model}_maze_{self.args.maze_id}.yaml"

        hyperparams_path = os.path.join(self.base_dir, filepath)
        with open(hyperparams_path, 'w') as f:
            yaml.dump(hyperparameters.__dict__, f)