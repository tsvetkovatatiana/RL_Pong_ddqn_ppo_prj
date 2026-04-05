import torch
import torch.nn as nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

class DQN(nn.Module):
    def __init__(self,
                 environment,
                 device=device,
                 in_channels=4,
                 hidden_filters=[32, 64],
                 start_epsilon=1.0,
                 max_decay=0.05,
                 decay_steps=300_000,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            nn.Linear(hidden_filters[1] * 9 * 9, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.apply(self._init)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device).float()
        return self.conv_layers(x / 255.0)

    def epsilon_greedy(self, state, dim=1):
        if np.random.random() < self.epsilon:
            return torch.tensor(self.environment.action_space.sample(), device=self.device)

        with torch.no_grad():
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            q_values = self(state)

        action = torch.argmax(q_values, dim=dim)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        return action

    def epsilon_decay(self, step):
        self.epsilon = self.max_decay + (self.start_epsilon - self.max_decay) * \
                       max(0, (self.decay_steps - step) / self.decay_steps)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)