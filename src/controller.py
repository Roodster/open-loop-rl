import torch as th
import numpy as np
import matplotlib.pyplot as plt


from src.utils import set_seed





class BaseController:
    def __init__(self, model, env):
        self.model = model
        self.env = env
        set_seed(1)
    
    def parameters(self):
        """ Returns a generator of the underlying model parameters. """
        return self.model
    
    def choose(self):
        pass
        
class Controller(BaseController):
    def __init__(self, model, env, args):
        super().__init__(model=model, env=env)
        self.model = model
        self.num_actions = args.n_actions
        self.gamma = args.gamma
        
    
    def extended_q_values(self, observations, actions):

        rewards, values = self.model(
            observations=observations,
            actions=actions,
            action_space=self.num_actions
        )
        rewards = rewards[:, -1]
        values = values[:, -1]

        q_values = rewards + self.gamma * values
        
        return q_values
    
    def choose(self, observations, actions):
        q_values = self.extended_q_values(observations=observations, actions=actions)
        actions = th.argmax(q_values, dim=0).flatten().cpu().numpy()[0]
        return actions
        
        
class EpsilonGreedyController(Controller):
    def __init__(self, model, env, args):
        super().__init__(model=model, env=env, args=args)
        self.num_decisions = 0
        self.max_epsilon = args.epsilon_start
        self.min_epsilon = args.epsilon_finish
        self.anneal_time = args.anneal_time
        self.exploration_fraction = args.exploration_fraction
        self.n_envs = args.n_envs
        self.epsilons = []
        
    def epsilon(self, ):
        """ Returns current epsilon. """
        # slope = (self.min_epsilon - self.max_epsilon) / self.duration
        # return max(slope * self.num_decisions + self.max_epsilon, self.min_epsilon)

        epsilon = max(1 - self.num_decisions / (self.anneal_time - 1), 0) \
                * (self.max_epsilon - self.min_epsilon) + self.min_epsilon
        
        self.epsilons.append(epsilon)
        
        return epsilon
                
    def exploit(self, observations, increase_counter=True):
        
  
        actions = th.tensor(np.arange(0, self.num_actions)).unsqueeze(1)
        actions = actions.unsqueeze(1)
        
        actions = th.nn.functional.one_hot(actions, num_classes=self.num_actions).squeeze(1).float()

        observations = th.Tensor(observations)
        actions = super().choose(observations=observations, actions=actions) 
        
        return actions
    
    def mixed_exploration(self, observations):
        eps = self.epsilon()
    
        if np.random.rand() < eps:

            p = max(0.3, eps**2)

            if p > 0.5:
                actions = np.random.choice(self.env.get_valid_explore_actions())
            else:
                actions = np.random.choice(self.env.get_all_explore_actions())
        else:
            actions = self.exploit(observations=observations, increase_counter=False)
            
        return actions
    
    def valid_exploration(self, observations):
        eps = self.epsilon()
    
        if np.random.rand() < eps:

            actions = np.random.choice(self.env.get_valid_explore_actions())

        else:
            actions = self.exploit(observations=observations, increase_counter=False)
            
        return actions
                
    def greedy_exploration(self, observations):
        eps = self.epsilon()
    
        if np.random.rand() < eps:
            actions = np.random.choice(self.env.get_all_explore_actions())
        else:
            actions = self.exploit(observations=observations, increase_counter=False)
            
        return actions    
            
    def choose(self, observations, explore_type='mixed',  increase_counter=True):
        
        if increase_counter: 
            self.num_decisions += 1
            
        if explore_type == 'mixed':
            actions = self.mixed_exploration(observations)
        elif explore_type == 'valid':
            actions = self.valid_exploration(observations)
        elif explore_type == 'all':    
            actions = self.greedy_exploration(observations)
        else:
            actions= self.exploit(observations)
        
        
        return actions
        
        