import numpy as np
import torch as th
import torch.nn.functional as F
from copy import deepcopy


class BaseLearner:
    
    
    def __init__(self, model, args):
        self.model = model
        self.target_model = None
        self.batch_size = args.batch_size
        self.device = args.device
        
    def train(self):
        pass
        
        

class Learner(BaseLearner):
    
    
    def __init__(self, model, args):
        super().__init__(model, args)
        self.target_update = args.target_update 
        self.target_update_interval = args.target_update_interval
        self.trajectory_length = args.trajectory_length
        self.all_parameters = list(model.parameters())
        self.optimizer = th.optim.Adam(self.all_parameters, lr=args.learning_rate)
        self.criterion = th.nn.MSELoss()
        self.gamma = args.gamma
        if args.target_model:
            self.target_model = deepcopy(model)
            for p in self.target_model.parameters():
                p.requires_grad = False
        
    def target_model_update(self):
        """ This function updates the target network with a hard update. """
        if self.target_model is not None:
            # Target network update by copying it every so often
            if self.target_update == 'copy':
                self.target_update_calls = (self.target_update_calls + 1) % self.target_update_interval
                if self.target_update_calls == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                    
    def train(self, batch):
        # Compute TD-loss
        
        
        with th.no_grad():

            possible_actions = th.tensor(np.arange(0, self.model.in_features))
            
            possible_actions = th.repeat_interleave(input=possible_actions,
                                                        repeats=self.batch_size * self.trajectory_length,
                                                        dim=0).unsqueeze(1).unsqueeze(1)

            possible_actions = th.nn.functional.one_hot(possible_actions.long(), num_classes=self.model.in_features).squeeze(
                dim=1).float().to(self.device)

            reshaped_obs = batch.next_observations.reshape(self.batch_size * self.trajectory_length,
                                                            self.model.in_representation)


            target_rewards, target_values = self.target_model(observations=reshaped_obs,
                                                            actions=possible_actions,
                                                            action_space=self.model.in_features)
         
            target_rewards = target_rewards.reshape(self.model.in_features,
                                                    self.batch_size,
                                                    self.trajectory_length)
            target_values = target_values.reshape(self.model.in_features,
                                                    self.batch_size,
                                                    self.trajectory_length)
           
            target_max, _ = (target_rewards + self.gamma * target_values).max(dim=0)  # y(s)
            td_target = target_max * (1 - batch.terminated.squeeze(2))  # y(s)
 

        input_actions = th.nn.functional.one_hot(batch.actions.type(th.int64).unsqueeze(1),
                                                    num_classes=self.model.in_features).float().squeeze(dim=1).squeeze(2).to(self.device)

        old_rewards, old_values = self.model(observations=batch.observations[:, 0],
                                            actions=input_actions,
                                            action_space=1)

        # Final loss calculation
        loss_val = F.mse_loss(old_values.squeeze(2), td_target)

    
        loss_rew = F.mse_loss(old_rewards.flatten(), batch.rewards.flatten())
        
        loss = loss_val + loss_rew
    
    
        # Backpropagate loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update target network (if specified) and return loss
        self.target_model_update()
        
        return loss.item(), loss_rew.item(), loss_val.item()
    
    
class LearnerSoftTarget(Learner):
    def __init__(self, model, args={}):
        super().__init__(model, args)
        self.target_update_calls = 0
        self.soft_target_update_factor = args.soft_target_update_factor
        if args.target_model:
            self.target_model = deepcopy(model)
            for p in self.target_model.parameters():
                p.requires_grad = False
        assert self.target_model is None or self.target_update == 'soft' or self.target_update == 'copy',\
            'If a target model is specified, it needs to be updated using the "soft" or "copy" options.'
        
    def target_model_update(self):
        """ This function updates the target network. """
        if self.target_model is not None:
            # Target network update by copying it every so often
            if self.target_update == 'copy':
                super().target_model_update()
            elif self.target_update == 'soft':
                    self.target_update_calls = (self.target_update_calls + 1) % self.target_update_interval
                    if self.target_update_calls == 0:                           
                        for tp, mp in zip(self.target_model.parameters(), self.model.parameters()):
                            tp.data.copy_(
                                self.soft_target_update_factor * mp.data + (1.0 - self.soft_target_update_factor) * tp.data
                            )