import numpy as np


from src.args import Args
from src.controller import Controller
from src.writer import Writer
from src.learner import Learner
from src.runner import Runner
from src.experiment import Experiment
from src.model import get_model
from src.buffer import TrajectoryBuffer

from gym_env.make_environment import make_environment

def main(args):
    
    
    env = make_environment(maze_id=args.maze_id, 
                                 seed=args.seed, 
                                 mask_regions=None,
                                 mask_idxs=None
                           )
    
    model = get_model(args=args, env=env)

    trajectory_buffer = TrajectoryBuffer()
    controller = Controller(model=model)
    learner = Learner(model=model)
    runner = Runner(env=env, controller=controller, replay_buffer=trajectory_buffer)
    writer = Writer(args=args)
    
    experiment = Experiment(args=args,
                            controller=controller,
                            runner=runner,
                            learner=learner,
                            writer=writer
                            )
    
    # ========== RUN EXPERIMENT ==========
    # Re-executing this code-block picks up the experiment where you left off
    try:
        experiment.run()
    except KeyboardInterrupt:
        experiment.close()
    experiment.plot(update=False)

        
    
        
if __name__ == "__main__":
    args = Args(file="./data/configs/default_lstm.yaml")
    main(args=args)