import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self,
                 environment,
                 device,
                 in_channels,
                 hidden_filters,
                 start_epsilon,
                 max_decay,
                 decay_steps, ):
        super().__init__()

        self.start_epsilon = start_epsilon
        self.epsilon = start_epsilon
        self.max_decay = max_decay
        self.decay_steps = decay_steps

        self.environment = environment
        self.num_actions = environment.action_space.n
        self.device = device

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_filters[0], kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(hidden_filters[0], hidden_filters[1], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(hidden_filters[1] * 9 * 9, 512),  # for 84x84 input
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.apply(self._init_weights)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device, dtype=torch.float32)
        return self.conv_layers(x / 255.0)

    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return torch.tensor([self.environment.action_space.sample()], device=self.device)

        with torch.no_grad():

            if len(state.shape) == 3:
                state = state.unsqueeze(0)

            q_values = self(state)
            action = torch.argmax(q_values, dim=1)

        return action

        # Decays epsilon to shift from exploration → exploitation over time (more for adventure, less for pong)

    def epsilon_decay(self, step):
        fraction = max(0.0, 1 - step / self.decay_steps)
        self.epsilon = self.max_decay + (self.start_epsilon - self.max_decay) * fraction

    # Initialise Linear/Conv layers with Kaiming init for ReLU, set biases to 0
    # Improves training stability and gradient flow
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

