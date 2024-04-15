import torch

import torch.nn.functional as F

from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int=1) -> None:
        super(Block, self).__init__()
        self.conv1  = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1    = nn.BatchNorm2d(out_channels)
        self.conv2  = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2    = nn.BatchNorm2d(out_channels)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clone the input tensor.
        y = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Add the residual.
        x = x + y

        x = self.relu(x)

        return x


class Attention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, bias: bool=False):
        super(Attention, self).__init__()
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.padding        = padding

        self.rel_h          = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w          = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)
        self.query          = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.key            = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value          = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.softmax        = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Set defualt datatypes.
        query   : torch.Tensor
        key     : torch.Tensor
        value   : torch.Tensor

        # Get the shape of x.
        batch, channels, height, width = x.size()

        # Pad x.
        x_pad = F.pad(input=x, pad=(self.padding, self.padding, self.padding, self.padding), value=0)

        # Calculate query, key and value.
        query   = self.query(x)
        key     = self.key(x_pad)
        value   = self.value(x_pad)

        query   = query.view(batch, 1, self.out_channels, height, width, 1)

        key     = key.unfold(2, self.kernel_size, 1)
        value   = value.unfold(2, self.kernel_size, 1)

        key     = key.unfold(3, self.kernel_size, 1)
        value   = value.unfold(3, self.kernel_size, 1)

        key_h, key_w = key.split(self.out_channels // 2, dim=1)
        key     = torch.cat((key_h + self.rel_h, key_w + self.rel_w), dim=1)

        key     = key.reshape(batch, 1, self.out_channels, height, width, -1)
        value   = value.reshape(batch, 1, self.out_channels, height, width, -1)

        output = query * key
        output = self.softmax(output)

        output = torch.einsum('...wk, ...wk -> ...w', output, value)
        output = output.view(batch, -1, height, width)

        return output


class H_GO(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_dim: int=128) -> None:
        super(H_GO, self).__init__()
        self.conv1  = nn.Conv2d(in_channels=input_size, out_channels=hidden_dim, kernel_size=3)
        self.bn1    = nn.BatchNorm2d(64)
        self.tanh   = nn.Tanh()

        self.layers = nn.Sequential(
            Block(hidden_dim, hidden_dim, 3, 1),
            Block(hidden_dim, hidden_dim, 3, 1),
            Block(hidden_dim, hidden_dim, 3, 1),
            Block(hidden_dim, hidden_dim, 3, 1),
            Block(hidden_dim, hidden_dim, 3, 1),

            Block(hidden_dim, hidden_dim, 3, 1),
            Block(hidden_dim, hidden_dim, 3, 1),
            Block(hidden_dim, hidden_dim, 3, 1),
            Block(hidden_dim, hidden_dim, 3, 1),
            Block(hidden_dim, hidden_dim, 3, 1),
        )

        self.attention = Attention(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        self.policy = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(578, output_size),
        )

        self.value = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(289, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.tanh(x)

        x = self.layers(x)

        # Attention network.
        x = self.attention(x)

        # Policy network.
        policy = self.policy(x)

        # Value network.
        value = self.value(x)

        return policy, value