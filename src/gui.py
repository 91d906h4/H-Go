import numpy as np


class GUI:
    def __init__(self, size: int=19) -> None:
        """GUI class

        A class for displaying the game board.

        Args:
            size (int): The size of the board.

        """

        self.size = size

    def display(self, state: list, move: int) -> None:
        """display public method

        Display the game board.

        Args:
            state (list): The state of the game.
            move (int): The last move.

        """

        # Set the labels.
        labels = "0123456789ABCDEFGHI"
        stone = {0: "·", 1: "O", -1: "X"}

        # Calculate the last move position.
        last_move_row = move // self.size
        last_move_col = move % self.size

        # Print col numbers.
        print("  " + " ".join(labels[:self.size]))

        for row in range(self.size):
            line = labels[row]

            if row == last_move_row and last_move_col == 0:
                line += "("
            else:
                line += " "

            for col in range(self.size):
                line += stone[state[row][col]]

                if row == last_move_row:
                    if col == last_move_col - 1:
                        line += "("
                    elif col == last_move_col:
                        line += ")"
                    else:
                        line += " "
                else:
                    line += " "

            print(line)