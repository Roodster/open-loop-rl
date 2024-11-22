from src.models.lstm import LSTMNetwork
from src.models.mha import MHANetwork, NormalizedMHANetwork, MultiMHANetwork, MultiNormalizedMHANetwork
import numpy as np

def get_model(args, env=None):
    """
    
    Get the specified model
    
    Current models:
        - LSTMNetwork: "lstm"
        - MHANetwork: "mha"

    Args:
        model_name (_type_, optional): _description_. Defaults to None.
        env: environment of the RL problem.
    """
    assert env is not None, "Error: No environment."
    
    in_features = env.action_space.n
    in_representation = np.array(env.observation_space.shape).prod()
    hidden_features = args.hidden_size
    
    if args.model == "lstm":
        return LSTMNetwork(in_features=in_features, in_representation=in_representation, hidden_features=hidden_features, device=args.device)
    elif args.model == "mha":
        return MHANetwork(in_features=in_features, in_representation=in_representation, hidden_features=hidden_features, device=args.device)
    elif args.model == "norm-mha":
        return NormalizedMHANetwork(in_features=in_features, in_representation=in_representation, hidden_features=hidden_features, device=args.device)
    elif args.model == "multi-mha":
        return MultiMHANetwork(in_features=in_features, in_representation=in_representation, hidden_features=hidden_features, device=args.device)
    elif args.model == "multi-norm-mha":
        return MultiNormalizedMHANetwork(in_features=in_features, in_representation=in_representation, hidden_features=hidden_features, device=args.device)
    else:
        raise "ERROR: model not implemented."