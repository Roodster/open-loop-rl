



class Info:
    
    def __init__(self, file=None):
        
        if file is not None:
            self._parse_file(file)    
        else: 
            self.episode_lengths = []
            self.episode_rewards = []
            self.episode_losses = []
            self.env_steps = []
    
    def _parse_file(self, file):
        pass
    
    def add_length(self, length):
        self.episode_lengths.append(length)

    def add_reward(self, reward):
        self.episode_rewards.append(reward)

    def add_loss(self, loss):
        self.episode_losses.append(loss)
        
    def add_step(self, step):
        self.env_steps.append(step)
    