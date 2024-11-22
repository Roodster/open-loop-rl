import numpy as np
import torch as th
from torchsummary import summary


from src.utils import set_seed
from src.args import Args
from src.controller import EpsilonGreedyController, OpenLoopController
from src.writer import Writer
from src.learner import LearnerSoftTarget
from src.runner import Runner, OpenLoopRunner
from src.experiment import Experiment
from src.models.model_manager import get_model
from src.buffer import TrajectoryBuffer
from src.utils import get_mask_config_indices


from gym_env.make_environment import make_environment
from gym_env.envs.astar import AstarSolver


import warnings
warnings.filterwarnings('ignore')

def main(args):
    
        
    """
    
        get masks
        Get models
        get metadata of models
        set parameters
        
        
        for each mask
            evaluate()
    """
    
    
    model_files = [
        '.\\logs\\run_dqn_multi-mha_3_12_0.0\seed_3_270720241905\models\model_270720241905_dqn_multi-mha_3_12_0.0_100000.pickle',
    ]

    mask_configurations = get_mask_config_indices(min_val=0, max_val=17)
    
    for model_path in model_files:
        env = make_environment(maze_id=args.maze_id, 
                                seed=args.seed, 
                                max_steps=args.train_max_episode_length,
                                mask_regions=None,
                                mask_idxs=None
                        )
        args.set_env(env=env)
        model = get_model(args=args, env=env)
        
        # print(summary(model, input_size=[(3072, 63), (12288, 1, 4)], batch_size=1))
        model.load_state_dict(th.load(model_path, map_location=args.device))
    
        
        trajectory_buffer = TrajectoryBuffer(args)
        controller = OpenLoopController(model=model, env=env, args=args)
        learner = LearnerSoftTarget(model=model, args=args)
        runner = Runner(env=env, controller=controller, replay_buffer=trajectory_buffer, args=args)

        writer = Writer(args=args)
        solver = AstarSolver(env=env, goal=env.goal_states)
        print('mask_configurations', mask_configurations)
        for mask_index_config in mask_configurations:
    
            print(f"current config", mask_index_config )

            env_eval = make_environment(maze_id=args.maze_id, 
                                        seed=args.seed, 
                                        max_steps=args.eval_max_episode_length,
                                        mask_regions=True,
                                        mask_idxs=mask_index_config
                           )            
            runner_eval = OpenLoopRunner(env=env_eval, 
                                         controller=controller, 
                                         replay_buffer=trajectory_buffer, 
                                         args=args)
    
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
                steps = experiment.evaluate()
                print(f"Number of optimal steps={steps} at mask length={len(mask_index_config)-1} ")
                
                # if steps < len(experiment.optimal_actions)-1:
                #     break 
            except KeyboardInterrupt:
                experiment.close()
            experiment.plot(update=False)
        
if __name__ == "__main__":
    args = Args(file="./data/configs/default_mha.yaml")

    main(args=args)

    
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('ncalls')
    # stats.print_stats()