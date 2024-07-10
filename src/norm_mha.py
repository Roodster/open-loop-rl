import math
import numpy as np

import torch.nn as nn
import torch as th

HIDDEN_SIZE = 42
N_ACTION_SPACE = 4

class ResGate(nn.Module):
    """Residual skip connection"""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return x + y

class NormalizedMHANetwork(nn.Module):
    def __init__(self, env, device="cpu"):
        super().__init__()
        self.device = device
        self.hidden_rep = nn.Linear(np.array(env.single_observation_space.shape).prod(), HIDDEN_SIZE).to(self.device)
        self.action_rep = nn.Linear(N_ACTION_SPACE, HIDDEN_SIZE).to(self.device)

        self.layernorm1 = nn.LayerNorm(HIDDEN_SIZE).to(self.device)
        self.layernorm2 = nn.LayerNorm(HIDDEN_SIZE).to(self.device)

        self.mha = nn.MultiheadAttention(embed_dim=HIDDEN_SIZE, num_heads=7, batch_first=True).to(self.device)

        self.ffn = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, N_ACTION_SPACE * HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(N_ACTION_SPACE * HIDDEN_SIZE, HIDDEN_SIZE)
        ).to(self.device)
        self.attn_gate = ResGate().to(self.device)
        self.mlp_gate = ResGate().to(self.device)

        self.reward = nn.Linear(HIDDEN_SIZE, 1).to(self.device)
        self.value = nn.Linear(HIDDEN_SIZE, 1).to(self.device)

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
        # for openloop:
        # actions: [batch_size, length_n, actions_num]
        # observations: [batch_size, space_num]
        hidden = self.hidden_rep(observations).repeat(1, action_space, 1).transpose(0, 1)

        action_rep = self.action_rep(actions)
        mha_input = th.cat((hidden, action_rep), dim=1)
        mha_input = mha_input + self.position_encoding(mha_input.shape[0], mha_input.shape[1], mha_input.shape[2])

        # here they input the whole sequences, the output shape should be
        # e*n*42 (e is the batch size, n is trajectory size, 42 is the hidden size)
        # return shape [batch_size, length_n, hidden_size]

        # Causal masking for attention
        # history_len:        The maximum number of actions to take in.
        history_len = actions.size(1) + 1

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

        # hidden_after_actions, _ = self.lstm(actions, (h_0, hidden))
        hidden_after_actions = hidden_after_actions[:, 1:, :]

        # print(hidden_after_actions.shape) # for debugging
        state_rewards = self.reward(hidden_after_actions)
        state_values = self.value(hidden_after_actions)
                

        
        return state_rewards, state_values
    
