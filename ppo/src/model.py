import torch.nn as nn
import torch.nn.functional as F


class PPO(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PPO, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)  # (32, 42,42) 
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # (32, 21, 21)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # (32, 11, 11)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # (32, 6, 6)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))  # orthogonal base
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)   # underline means update val in-place

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.linear(x.view(x.size(0), -1))  # flattern
        return self.actor_linear(x), self.critic_linear(x)
