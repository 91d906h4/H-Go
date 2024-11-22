import torch
import random

import numpy as np

from hts import HTS
from gui import GUI
from network import H_GO
from collections import deque


class Game:
    def __init__(self, size: int, model: H_GO, device: torch.device) -> None:
        """Game class

        A class for game Go.

        Args:
            size (int): The size of the board.
            model (H_GO): The AI model.
            device (torch.device): The device of the model.

        """

        self.size       = size
        self.model      = model
        self.device     = device

        # Set default values.
        self.steps      = set() # The set of visited steps.
        self.gui        = GUI(size=size, label_mode="char")
        self.game_queue = deque(maxlen=7)
        self.game_board = np.zeros((self.size, self.size))

        # Initialize game queue.
        for _ in range(7):
            self.game_queue.append(self.game_board.copy())

    def play(self) -> None:
        """play public method

        The main game loop.
        The expected input format is "row,col", or input "quit" to quit the game.
        By default, the game will be played with the AI as the black player and the human as the white player.

        """

        while True:
            # Initialize HTS.
            hts = HTS(state=self.game_queue, depth=4, breadth=4, temperature=0.5, player=-1, model=self.model, device=self.device)

            # Get bets moves from HTS.
            best_moves = hts.get_best_move()

            # Set defualt AI's move to random
            # to avoid all the moves in best_moves
            # are invalid (visited).
            while True:
                if move := random.randint(0, 360) not in self.steps:
                    ai_move = move
                    break

            # Select the best move from outputs of HTS.
            # If the best move is in the visited steps,
            # then select another move.
            for move in best_moves:
                if move not in self.steps:
                    ai_move = move
                    break

            # Update the set of visited steps.
            self.steps.add(ai_move)

            # Calculate the row and col.
            row, col = ai_move // 19, ai_move % 19

            # Update game board.
            self.game_board[row][col] = -1

            # Update game queue.
            self.game_queue.append(self.game_board.copy())

            # Print game board.
            print(f"AI's move: ({row + 1}, {col + 1})")
            self.gui.display(self.game_board, ai_move)
            print()

            # Get player's move.
            player_move = input()

            # Quit if player inputs "quit".
            if player_move == "quit": break

            # Get row and col.
            row, col = player_move.split(",")
            row, col = int(row), int(col)
            row, col = row - 1, col - 1

            # Update the set of visited steps.
            self.steps.add(player_move)

            # Update game board.
            self.game_board[row][col] = 1
            self.game_queue.append(self.game_board.copy())

            # Print game board.
            print(f"Player's move: ({row}, {col})")
            self.gui.display(self.game_board, row * 19 + col)
            print()

        # Print quit message.
        print("User quit the game.")