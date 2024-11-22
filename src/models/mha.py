import torch.nn as nn
import numpy as np
import torch as th
import math

from .base import BaseModel

    
class ResGate(nn.Module):
    """Residual skip connection"""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return x + y

class MHANetwork(BaseModel):
    def __init__(self, in_features, in_representation, hidden_features, device):
        super().__init__(in_features, in_representation, hidden_features, device)
        self.hidden_rep = nn.Linear(self.in_representation, self.hidden_features).to(self.device)
        self.action_rep = nn.Linear(self.in_features, self.hidden_features).to(self.device)

        self.layernorm1 = nn.LayerNorm(self.hidden_features).to(self.device)
        self.layernorm2 = nn.LayerNorm(self.hidden_features).to(self.device)

        self.mha = nn.MultiheadAttention(embed_dim=self.hidden_features, num_heads=7, batch_first=True).to(self.device)

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_features, self.in_features * self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.in_features * self.hidden_features, self.hidden_features)
        ).to(self.device)
        self.attn_gate = ResGate().to(self.device)
        self.mlp_gate = ResGate().to(self.device)

    def position_encoding(self, batch_size, length_n, actions_num):
        # here I use the sinusoidal positional encoding same as the one used in transformer
        # shape [length_n, 1, 1]
        position = th.arange(length_n, device=self.device).unsqueeze(1).unsqueeze(2).to(self.device)
        # get shape[actions_num/2]
        div_term = th.exp(th.arange(0, actions_num, 2) * -(math.log(10000.0) / actions_num)).to(self.device)
        # get shape: [1, 1, actions_num/2]
        div_term = div_term.unsqueeze(0).unsqueeze(1)
        pe = th.zeros(length_n, 1, actions_num, device=self.device)
        pe[:, :, 0::2] = th.sin(position * div_term)
        pe[:, :, 1::2] = th.cos(position * div_term)
        # return [1, length_n, actions_num]
        return pe.permute(1, 0, 2)
    
    def hidden(self, mha_input):

        # Causal masking for attention
        # history_len:        The maximum number of actions to take in.
        history_len = mha_input.size(1) + 1

        # The mask is additive meaning that with mask:
        # [0, -inf, -inf, ..., -inf]
        # [0,    0, -inf, ..., -inf]
        # ...
        # [0,    0,    0, ...,    0]
        # the mask values will be added to the attention weight.        
        # I.o.w. 0 means that timestep is allowed to attend.
        # So the first timestep can attend only to the first timestep
        # and the last timestep can attend to all observations.
        attn_mask = th.triu(th.ones(history_len, history_len), diagonal=1).to(self.device)
        attn_mask[attn_mask.bool()] = -float("inf")

        # print(hidden.shape, action_rep.shape) # for debugging
        hidden_after_actions, _ = self.mha(mha_input,
                                           mha_input,
                                           mha_input,
                                           attn_mask=attn_mask[: mha_input.size(1), : mha_input.size(1)])

        hidden_after_actions = self.attn_gate(hidden_after_actions, mha_input)
        hidden_after_actions = self.layernorm1(hidden_after_actions)
        ffn = self.ffn(hidden_after_actions)
        # Skip connection then LayerNorm
        hidden_after_actions = self.mlp_gate(hidden_after_actions, ffn)
        hidden_after_actions = self.layernorm2(hidden_after_actions)
        
        return hidden_after_actions

    def forward(self, observations, actions, action_space):
        # for openloop:
        # actions: [batch_size, length_n, actions_num]
        # observations: [batch_size, space_num]
        hidden = self.hidden_rep(observations).repeat(1, action_space, 1).transpose(0, 1).to(self.device)

        action_rep = self.action_rep(actions)
        mha_input = th.cat((hidden, action_rep), dim=1).to(self.device)
        mha_input = mha_input + self.position_encoding(mha_input.shape[0], mha_input.shape[1], mha_input.shape[2]).to(self.device)
        mha_input = mha_input.to(self.device)

        # here they input the whole sequences, the output shape should be
        # e*n*42 (e is the batch size, n is trajectory size, 42 is the hidden size)
        # return shape [batch_size, length_n, self.hidden_features]
        
        hidden = self.hidden(mha_input=mha_input)
        hidden = hidden[:, 1:, :]

        state_rewards = self.reward(hidden)
        state_values = self.value(hidden)
        return state_rewards, state_values
    
    
class NormalizedMHANetwork(BaseModel):
    def __init__(self, in_features, in_representation, hidden_features, device):
        super().__init__(in_features, in_representation, hidden_features, device)
        self.device = device
        self.hidden_rep = nn.Linear(self.in_representation, self.hidden_features).to(self.device)
        self.action_rep = nn.Linear(self.in_features, self.hidden_features).to(self.device)

        self.layernorm1 = nn.LayerNorm(self.hidden_features).to(self.device)
        self.layernorm2 = nn.LayerNorm(self.hidden_features).to(self.device)

        self.mha = nn.MultiheadAttention(embed_dim=self.hidden_features, num_heads=7, batch_first=True).to(self.device)

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_features, self.in_features * self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.in_features * self.hidden_features, self.hidden_features)
        ).to(self.device)
        self.attn_gate = ResGate().to(self.device)
        self.mlp_gate = ResGate().to(self.device)

    def position_encoding(self, batch_size, length_n, actions_num):
        # here I use the sinusoidal positional encoding same as the one used in transformer
        # shape [length_n, 1, 1]
        position = th.arange(length_n, device=self.device).unsqueeze(1).unsqueeze(2).to(self.device)
        # get shape[actions_num/2]
        div_term = th.exp(th.arange(0, actions_num, 2) * -(math.log(10000.0) / actions_num)).to(self.device)
        # get shape: [1, 1, actions_num/2]
        div_term = div_term.unsqueeze(0).unsqueeze(1)
        pe = th.zeros(length_n, 1, actions_num, device=self.device)
        pe[:, :, 0::2] = th.sin(position * div_term)
        pe[:, :, 1::2] = th.cos(position * div_term)
        # return [1, length_n, actions_num]
        return pe.permute(1, 0, 2)

    def hidden(self, mha_input):

        # here they input the whole sequences, the output shape should be
        # e*n*42 (e is the batch size, n is trajectory size, 42 is the hidden size)
        # return shape [batch_size, length_n, hidden_size]

        # Causal masking for attention
        # history_len:        The maximum number of actions to take in.
        history_len = mha_input.size(1) + 1

        # The mask is additive meaning that with mask:
        # [0, -inf, -inf, ..., -inf]
        # [0,    0, -inf, ..., -inf]
        # ...
        # [0,    0,    0, ...,    0]
        # the mask values will be added to the attention weight.        
        # I.o.w. 0 means that timestep is allowed to attend.
        # So the first timestep can attend only to the first timestep
        # and the last timestep can attend to all observations.
        attn_mask = th.triu(th.ones(history_len, history_len), diagonal=1)
        attn_mask[attn_mask.bool()] = -float("inf")
        
        
        normalized_mha_input = self.layernorm1(mha_input)

        # print(hidden.shape, action_rep.shape) # for debugging
        hidden_after_actions, _ = self.mha(normalized_mha_input,
                                        normalized_mha_input,
                                        normalized_mha_input,
                                        attn_mask=attn_mask[: mha_input.size(1), : mha_input.size(1)])

        hidden_after_actions = self.attn_gate(hidden_after_actions, mha_input)
        normalized_hidden_after_actions = self.layernorm2(hidden_after_actions)
        ffn = self.ffn(normalized_hidden_after_actions)
        # Skip connection then LayerNorm
        hidden_after_actions = self.mlp_gate(hidden_after_actions, ffn)
            
        return hidden_after_actions

    def forward(self, observations, actions, action_space):

        # for openloop:
        # actions: [batch_size, length_n, actions_num]
        # observations: [batch_size, space_num]
        hidden = self.hidden_rep(observations).repeat(1, action_space, 1).transpose(0, 1)

        action_rep = self.action_rep(actions)
        mha_input = th.cat((hidden, action_rep), dim=1)
        mha_input = mha_input + self.position_encoding(mha_input.shape[0], mha_input.shape[1], mha_input.shape[2])

        hidden = self.hidden(mha_input=mha_input)
        # hidden_after_actions, _ = self.lstm(actions, (h_0, hidden))
        hidden = hidden[:, 1:, :]
        
        state_rewards = self.reward(hidden)
        state_values = self.value(hidden)
                        
        return state_rewards, state_values
    


class MultiNormalizedMHANetwork(BaseModel):
    def __init__(self, in_features, in_representation, hidden_features, device):
        super().__init__(in_features, in_representation, hidden_features, device)
        
        self.hidden_rep = nn.Linear(self.in_representation, self.hidden_features).to(self.device)
        self.action_rep = nn.Linear(self.in_features, self.hidden_features).to(self.device)
        
        self.encoder1 = NormalizedMHANetwork(in_features, in_representation, hidden_features, device).to(self.device)
        self.relu = nn.ReLU()
        self.encoder2 = NormalizedMHANetwork(in_features, in_representation, hidden_features, device).to(self.device)
        self.encoder3 = NormalizedMHANetwork(in_features, in_representation, hidden_features, device).to(self.device)
        
        
    def position_encoding(self, batch_size, length_n, actions_num):
        # here I use the sinusoidal positional encoding same as the one used in transformer
        # shape [length_n, 1, 1]
        position = th.arange(length_n, device=self.device).unsqueeze(1).unsqueeze(2).to(self.device)
        # get shape[actions_num/2]
        div_term = th.exp(th.arange(0, actions_num, 2) * -(math.log(10000.0) / actions_num)).to(self.device)
        # get shape: [1, 1, actions_num/2]
        div_term = div_term.unsqueeze(0).unsqueeze(1)
        pe = th.zeros(length_n, 1, actions_num, device=self.device)
        pe[:, :, 0::2] = th.sin(position * div_term)
        pe[:, :, 1::2] = th.cos(position * div_term)
        # return [1, length_n, actions_num]
        return pe.permute(1, 0, 2)


    def forward(self, observations, actions, action_space):
        hidden = self.hidden_rep(observations).repeat(1, action_space, 1).transpose(0, 1)

        action_rep = self.action_rep(actions)
        mha_input = th.cat((hidden, action_rep), dim=1)
        mha_input = mha_input + self.position_encoding(mha_input.shape[0], mha_input.shape[1], mha_input.shape[2])
                
        mha_input = self.encoder1.hidden(mha_input=mha_input)
        mha_input = self.relu(mha_input)

        mha_input= self.encoder2.hidden(mha_input=mha_input)
        mha_input = self.relu(mha_input)

        mha_output = self.encoder3.hidden(mha_input=mha_input)
        
        hidden_after_actions = mha_output[:, 1:, :]
        state_rewards = self.reward(hidden_after_actions)
        state_values = self.value(hidden_after_actions)
        
        return state_rewards, state_values
    
    
class MultiMHANetwork(BaseModel):
    def __init__(self, in_features, in_representation, hidden_features, device):
        super().__init__(in_features, in_representation, hidden_features, device)
        
        self.hidden_rep = nn.Linear(self.in_representation, self.hidden_features).to(self.device)
        self.action_rep = nn.Linear(self.in_features, self.hidden_features).to(self.device)
        
        self.encoder1 = MHANetwork(in_features, in_representation, hidden_features, device).to(self.device)
        self.relu = nn.ReLU()
        self.encoder2 = MHANetwork(in_features, in_representation, hidden_features, device).to(self.device)
        self.encoder3 = MHANetwork(in_features, in_representation, hidden_features, device).to(self.device)
        
        
    def position_encoding(self, batch_size, length_n, actions_num):
        # here I use the sinusoidal positional encoding same as the one used in transformer
        # shape [length_n, 1, 1]
        position = th.arange(length_n, device=self.device).unsqueeze(1).unsqueeze(2).to(self.device)
        # get shape[actions_num/2]
        div_term = th.exp(th.arange(0, actions_num, 2) * -(math.log(10000.0) / actions_num)).to(self.device)
        # get shape: [1, 1, actions_num/2]
        div_term = div_term.unsqueeze(0).unsqueeze(1)
        pe = th.zeros(length_n, 1, actions_num, device=self.device)
        pe[:, :, 0::2] = th.sin(position * div_term)
        pe[:, :, 1::2] = th.cos(position * div_term)
        # return [1, length_n, actions_num]
        return pe.permute(1, 0, 2)


    def forward(self, observations, actions, action_space):
        print(f"OBS=\n{observations.numpy()}\n")
        hidden = self.hidden_rep(observations).repeat(1, action_space, 1).transpose(0, 1)

        action_rep = self.action_rep(actions)
        
        mha_input = th.cat((hidden, action_rep), dim=1)
        mha_input = mha_input + self.position_encoding(mha_input.shape[0], mha_input.shape[1], mha_input.shape[2])
                
        mha_input = self.encoder1.hidden(mha_input=mha_input)
        mha_input = self.relu(mha_input)

        mha_input= self.encoder2.hidden(mha_input=mha_input)
        mha_input = self.relu(mha_input)

        hidden = self.encoder3.hidden(mha_input=mha_input)
        
        hidden = hidden[:, 1:, :]
        state_rewards = self.reward(hidden)
        state_values = self.value(hidden)
        
        return state_rewards, state_values
    