import torch

import numpy as np

from network import H_GO
from torch import nn, optim
from dataReader import DataReader


class SL_Trainer:
    def __init__(self, epochs: int, model: H_GO, device: torch.device, optimizer: optim.RAdam, loss_fu_policy: nn.CrossEntropyLoss, loss_fu_value: nn.BCELoss) -> None:
        self.epochs         = epochs
        self.model          = model
        self.device         = device
        self.optimizer      = optimizer
        self.loss_fu_policy = loss_fu_policy
        self.loss_fu_value  = loss_fu_value

    def train(self, data_reader: DataReader, data_num: int, batch_size: int) -> None:
        # Set model to training mode.
        self.model.train()

        for epoch in range(self.epochs):
            # Set default value.
            total_loss          = 0
            total_acc_policy    = 0
            total_acc_value     = 0

            for _ in range(data_num // batch_size):
                # Set default datatypes.
                policy: torch.Tensor
                value:  torch.Tensor
                loss:   torch.Tensor

                # Get training data batch.
                training_batch = data_reader.get_training_batch(batch_size=batch_size, shuffle=True)

                # Convert data to numpy.array.
                game_data   = np.array([x[0] for x in training_batch])
                step        = np.array([x[1] for x in training_batch])
                winner      = np.array([x[2] for x in training_batch])

                # Convert data to tensor.
                game_data   = torch.tensor(game_data, dtype=torch.float).to(device=self.device)
                step        = torch.tensor(step, dtype=torch.long).to(device=self.device)
                winner      = torch.tensor(winner, dtype=torch.float).to(device=self.device)

                # Get model output.
                policy, value = self.model(game_data)

                # Calculate loss.
                policy_loss = self.loss_fu_policy(policy, step)
                value_loss  = self.loss_fu_value(value, winner)
                loss        = policy_loss + value_loss

                # Calculate loss.
                total_loss += loss.item()

                # Calculate accuracy.
                total_acc_policy += torch.sum(policy.argmax(1) == step).item()
                total_acc_value += torch.sum(torch.round(value) == winner).item()

                # Update model parameters.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Print training information.
            print(
                f"Epoch: {epoch:03} | "
                f"Loss of model: {total_loss:.3f} | "
                f"Accuracy of policy: {total_acc_policy / data_num * 100:.3f}% | "
                f"Accuracy of value: {total_acc_value / data_num * 100:.3f}%"
            )

        # Set model to evaluation mode.
        self.model.eval()


class RL_Trainer:
    def __init__(self) -> None:
        ...