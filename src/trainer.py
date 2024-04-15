import torch

import numpy as np

from network import H_GO
from torch import nn, optim
from dataReader import DataReader


class SL_Trainer:
    def __init__(self, epochs: int, model: H_GO, device: torch.device, optimizer: optim.RAdam, loss_fu_policy: nn.CrossEntropyLoss, loss_fu_value: nn.BCELoss) -> None:
        """SL_Trainer class

        The supervised learning trainer class for H-Go.

        Args:
            epochs (int): The number of epochs.
            model (H_GO): The AI model.
            device (torch.device): The device of the model.
            optimizer (optim.RAdam): The optimizer of the model.
            loss_fu_policy (nn.CrossEntropyLoss): The loss function for policy.
            loss_fu_value (nn.BCELoss): The loss function for value.

        """

        self.epochs         = epochs
        self.model          = model
        self.device         = device
        self.optimizer      = optimizer
        self.loss_fu_policy = loss_fu_policy
        self.loss_fu_value  = loss_fu_value

    def train(self, data_reader: DataReader, batch_size: int, test_every_epoch: int=-1) -> None:
        """train public method
        
        Trainer for H-Go model.

        Args:
            data_reader (DataReader): The data reader.
            batch_size (int): The batch size.
            test_every_epoch (int, optional): Test model every n epochs. (default: -1)

        """

        # Set model to training mode.
        self.model.train()

        for epoch in range(self.epochs):
            # Set default value.
            total_loss          = 0
            total_acc_policy    = 0
            total_acc_value     = 0

            for _ in range(data_reader.train_data_num // batch_size):
                # Set default datatypes.
                policy  : torch.Tensor
                value   : torch.Tensor
                loss    : torch.Tensor

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
                f"Accuracy of policy: {total_acc_policy / data_reader.train_data_num * 100:.3f}% | "
                f"Accuracy of value: {total_acc_value / data_reader.train_data_num * 100:.3f}%"
            )

            # Test model every n epoch.
            if test_every_epoch == -1:
                continue
            elif test_every_epoch % epoch == 0:
                self.test(data_reader, batch_size)

        # Set model to evaluation mode.
        self.model.eval()

    @torch.no_grad()
    def test(self, data_reader: DataReader, batch_size: int) -> None:
        """test public method
        
        Tester for H-Go model.

        Args:
            data_reader (DataReader): The data reader.
            batch_size (int): The batch size.

        """

        # Set default value.
        total_acc_policy    = 0
        total_acc_value     = 0

        for _ in range(data_reader.test_data_num // batch_size):
            # Set default datatypes.
            policy: torch.Tensor
            value:  torch.Tensor

            # Get training data batch.
            testing_batch = data_reader.get_testing_batch(batch_size=batch_size, shuffle=True)

            # Convert data to numpy.array.
            game_data   = np.array([x[0] for x in testing_batch])
            step        = np.array([x[1] for x in testing_batch])
            winner      = np.array([x[2] for x in testing_batch])

            # Convert data to tensor.
            game_data   = torch.tensor(game_data, dtype=torch.float).to(device=self.device)
            step        = torch.tensor(step, dtype=torch.long).to(device=self.device)
            winner      = torch.tensor(winner, dtype=torch.float).to(device=self.device)

            # Get model output.
            policy, value = self.model(game_data)

            # Calculate accuracy.
            total_acc_policy += torch.sum(policy.argmax(1) == step).item()
            total_acc_value += torch.sum(torch.round(value) == winner).item()

        # Print testing information.
        print(
            f"Accuracy of policy: {total_acc_policy / data_reader.test_data_num * 100:.3f}% | "
            f"Accuracy of value: {total_acc_value / data_reader.test_data_num * 100:.3f}%"
        )


class RL_Trainer:
    def __init__(self) -> None:
        ...