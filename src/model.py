import torch.nn as nn
import numpy as np
import torch as th

HIDDEN_SIZE = 42

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
    
    in_features = np.array(env.observation_space.shape).prod()
    in_representation = env.action_space.n
    hidden_features = args.hidden_size
    
    if args.model == "lstm":
        return LSTMNetwork(in_features=in_features, in_representation=in_representation, hidden_features=hidden_features, device=args.device)
    else:
        raise "ERROR: model not implemented."

class BaseModel(nn.Module):
    def __init__(self, in_features, in_representation, hidden_features, device):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.in_representation = in_representation
        self.hidden_features = hidden_features
        
        
    def forward(self):
        pass


class LSTMNetwork(BaseModel):
    def __init__(self, in_features, in_representation, hidden_features, device):
        super().__init__(in_features, in_representation, hidden_features, device)
        self.hidden_rep = nn.Linear(self.in_representation, self.hidden_features).to(self.device)

        self.lstm = nn.LSTM(input_size=self.in_features, hidden_size=self.hidden_features, batch_first=True).to(self.device)
        self.reward = nn.Linear(self.hidden_features, 1).to(self.device)
        self.value = nn.Linear(self.hidden_features, 1).to(self.device)

    def forward(self, observations, actions, action_space):
        hidden = self.hidden_rep(observations).repeat(1, action_space, 1)
        h_0 = th.zeros(hidden.shape, device=self.device)
        # here they input the whole sequences, the output shape should be
        # e*n*42 (e is the batch size, n is trajectory size, 42 is the hidden size)
        hidden_after_actions, _ = self.lstm(actions, (h_0, hidden))
        state_rewards = self.reward(hidden_after_actions)
        state_values = self.value(hidden_after_actions)
        return state_rewards, state_values
    
    