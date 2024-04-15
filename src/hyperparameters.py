# Set hyperparameters.

# DataReader.
dataset_path    = "../dataset/pro"  # The directory of the dataset.
data_num        = 10                # The number of games to load.
train_ratio     = 0.8               # The ratio of training and testing data.
batch_size      = 10                # Batch size.
data_augment    = False             # Whether to augment data. (this will augment data for 8 times)

# Model.
input_size      = 8                 # Input size of model.
output_size     = 361               # Output size of model (the positions of 19x19 game board).
hidden_dim      = 64                # Hidden dimension of model.

# Trainer.
epochs          = 5                 # The number of epochs.
learning_rate   = 0.001             # Learning rate.

# Others.
game_board_size = 19                # The size of the game board.