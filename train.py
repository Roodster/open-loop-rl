import numpy as np
import torch as th

from src.utils import set_seed
from src.args import Args
from src.controller import EpsilonGreedyController, OpenLoopController
from src.writer import Writer
from src.learner import LearnerSoftTarget
from src.runner import Runner, OpenLoopRunner
from src.experiment import Experiment
from src.models.model_manager import get_model
from src.buffer import TrajectoryBuffer

from gym_env.make_environment import make_environment
from gym_env.envs.astar import AstarSolver

import warnings
warnings.filterwarnings('ignore')

def main(args):
    
    
    env = make_environment(maze_id=args.maze_id, 
                                 seed=args.seed, 
                                 max_steps=args.train_max_episode_length,
                                 mask_regions=None,
                                 mask_idxs=None
                           )
    env_eval = make_environment(maze_id=args.maze_id, 
                                 seed=args.seed, 
                                 max_steps=args.eval_max_episode_length,
                                 mask_regions=None,
                                 mask_idxs=None
                           )
    
    args.set_env(env=env)
    
    model = get_model(args=args, env=env)
    trajectory_buffer = TrajectoryBuffer(args)
    controller = OpenLoopController(model=model, env=env, args=args)
    learner = LearnerSoftTarget(model=model, args=args)
    runner = OpenLoopRunner(env=env, controller=controller, replay_buffer=trajectory_buffer, args=args)
    runner_eval = Runner(env=env_eval, controller=controller, replay_buffer=trajectory_buffer, args=args)

    writer = Writer(args=args)
    solver = AstarSolver(env=env, goal=env.goal_states)
    
    
    experiment = Experiment(args=args,
                            controller=controller,
                            runner=runner,
                            runner_eval=runner_eval,
                            learner=learner,
                            solver=solver,
                            writer=writer
                            )
    
    # ========== RUN EXPERIMENT ==========
    try:
        experiment.run()
    except KeyboardInterrupt:
        experiment.close()
    experiment.plot(update=False)
       
    import matplotlib.pyplot as plt 
    epsilons = controller.epsilons
    fig = plt.gcf()
    plt.clf()
    plt.plot(range(0, len(epsilons)), epsilons)
    writer.save_plots(plot=fig, attribute="epsilon_schedule")
    plt.show()
        
    
        
if __name__ == "__main__":
    args = Args(file="./data/configs/default_mha.yaml")

    main(args=args)

    
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('ncalls')
    # stats.print_stats()