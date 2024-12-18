'''
model : 4 -> 1 action
DQN :
        predicts = torch.gather(self.model(history), 1, actions)
        max_next_q = self.target_model(next_history).detach().max(1)[0].unsqueeze(1)
        targets = rewards + (1 - dones) * self.discount_factor * max_next_q

        loss = nn.SmoothL1Loss()(predicts, targets).to(self.device)
'''       

env = gym.make("CartPole-v1", render_mode="rgb_array")
obs = (4,     
       


class MyModel
    nn 

class MyAgent:

