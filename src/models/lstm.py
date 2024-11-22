import torch as th
import torch.nn as nn
from src.models.base import BaseModel


class LSTMNetwork(BaseModel):
    def __init__(self, in_features, in_representation, hidden_features, device):
        super().__init__(in_features, in_representation, hidden_features, device)
        self.hidden_rep = nn.Linear(self.in_representation, self.hidden_features).to(self.device)

        self.lstm = nn.LSTM(input_size=self.in_features, hidden_size=self.hidden_features, batch_first=True).to(self.device)


    def forward(self, observations, actions, action_space):        
        # print(f"OBS=\n{observations.numpy()}\n")
        hidden = self.hidden_rep(observations).repeat(1, action_space, 1)
        h_0 = th.zeros(hidden.shape, device=self.device)
        # here they input the whole sequences, the output shape should be
        # e*n*42 (e is the batch size, n is trajectory size, 42 is the hidden size)
        hidden_after_actions, _ = self.lstm(actions, (h_0, hidden))
        state_rewards = self.reward(hidden_after_actions)
        state_values = self.value(hidden_after_actions)
        return state_rewards, state_values