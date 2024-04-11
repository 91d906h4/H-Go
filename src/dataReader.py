import os
import time
import pathlib

import numpy as np

from collections import deque


class DataReader:
    def __init__(self, dir_path: str, load_num: int=-1) -> None:
        """DataReader class

        Reading the SGF files and converting them to training data.

        Args:
            dir_path (int): The path of the folder containing the SGF files.
            load_num (str): How many SGF files to load. (default = -1)

        """

        self.dir_path = dir_path
        self.load_num = load_num

        self.raw_data = list()
        self.clear_data = list()
        self.converted_data = list()
        self.train_data = list()

        # Call private method.
        self._read_raw_data()
        self._clear_data()
        self._convert_data()
        self._make_training_data()

    def _read_raw_data(self) -> None:
        """_read_raw_data private method

        Read the raw data from the SGF files.

        """

        # Return if the raw data is already loaded.
        if self.raw_data: return

        # Set timestamp.
        start_time = time.time()

        # Get all filepaths.
        files = pathlib.Path(self.dir_path).rglob("*.sgf")

        # Set counter.
        total = len(files) if self.load_num == -1 else self.load_num
        counter = 0

        for file in files:
            # Try to read files in UTF-8 format.
            try:
                # Read content from file.
                content = open(file, mode="r", encoding="utf-8").read()

                self.raw_data.append(content)

            # Continue if an error occurs.
            except: continue

            # Print progress message.
            counter += 1
            print(f"Progess: {counter}/{total} ({counter / total * 100:.2f}%) ", end="\r")

            # Break if the number of files is reached.
            if counter == total: break

        # Print complete message.
        print(f"Read raw data completed. ({time.time() - start_time:.2f} s)")

    def _clear_data(self) -> None:
        """_clear_data private method

        Remove unnecessary information and clear the read data.
        Only the game information and the winner will be kept.

        """

        # Return if the clear data is already loaded.
        if self.clear_data: return

        # Set clock.
        start_time = time.time()

        # Set counter.
        total = len(self.raw_data)
        counter = 0

        for content in self.raw_data:
            content: str

            # Remove unnecessary information.
            content = content.split(";", 2)
            game_info = content[1]
            content = content[2]

            # Remove the last right parenthesis.
            content = content[:-1]

            # Remove player information.
            # Since the black player is always the first player,
            # so we can remove all the player information.
            content = content.replace("B", "")
            content = content.replace("W", "")
            content = content.replace("[", "")
            content = content.replace("]", "")

            # Convert content into list.
            content = content.split(";")

            # Get winner.
            if game_info[game_info.find("RE[") + 3] == "W":
                winner = 1
            else:
                winner = -1

            self.clear_data.append((content, winner))

            # Print progress message.
            counter += 1
            print(f"Progess: {counter}/{total} ({counter / total * 100:.2f}%) ", end="\r")

        # Print complete message.
        print(f"Clear data completed. ({time.time() - start_time:.2f} s)")

    def _convert_data(self) -> None:
        """_convert_data private method

        Convert the step data into row and column.

        """

        # Reutnrn if the converted data is already loaded.
        if self.converted_data: return

        # Set clock.
        start_time = time.time()

        # Set counter.
        total = len(self.clear_data)
        counter = 0

        # Create position table.
        position_table = {
            'a': 0,  'b': 1,  'c': 2,  'd': 3,  'e': 4,
            'f': 5,  'g': 6,  'h': 7,  'i': 8,  'j': 9,
            'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14,
            'p': 15, 'q': 16, 'r': 17, 's': 18,
        }

        for game, winner in self.clear_data:
            positions = list()

            for step in game:
                # Break if the step string is not 2 characters long,
                # the game step behind will be dropped.
                if len(step) != 2: break

                # Get row and column.
                row, col = list(step)

                # Convert x and y to numbers.
                row = position_table[row]
                col = position_table[col]

                positions.append((row, col))

            self.converted_data.append((positions, winner))

            # Print progress message.
            counter += 1
            print(f"Progess: {counter}/{total} ({counter / total * 100:.2f}%) ", end="\r")

        # Print complete message.
        print(f"Convert data completed. ({time.time() - start_time:.2f} s)")

    def _make_training_data(self) -> None:
        """_make_training_data private method

        Create 2D game board data from position data and create the 8 layers previous moves data.

        """

        # Return if the train data is already loaded.
        if self.train_data: return

        # Set clock.
        start_time = time.time()

        # Set counter.
        total = len(self.raw_data)
        counter = 0

        for game, winner in self.converted_data:
            # Create game board.
            game_board = np.zeros((19, 19))

            # Create game queue.
            # The input of H-Go is 7 previous moves and 1 "whos_turn layer".
            game_queue = deque(maxlen=7)
            for _ in range(7): game_queue.append(game_board.copy())

            # Set whos_turn.
            # The black player is always the first player, so we set it to -1.
            whos_turn = -1

            for step in game:
                # Make whos_turn layer.
                game_data = list(np.full((1, 19, 19), fill_value=whos_turn)) + list(game_queue)

                # Append data.
                self.train_data.append((game_data, step, whos_turn, winner))

                # Update game board.
                row = step[0]
                col = step[1]
                game_board[row][col] = whos_turn

                # Update game queue.
                game_queue.append(game_board.copy())

                # Update whos_turn.
                whos_turn *= -1

            # Print progress message.
            counter += 1
            print(f"Progess: {counter}/{total} ({counter / total * 100:.2f}%) ", end="\r")

        # Print complete message.
        print(f"Make train data completed. ({time.time() - start_time:.2f} s)")

    def get_train_data(self) -> list:
        """get_train_data public method

        Get the training data.

        Returns:
            list: The training data containing (game_data, step, whos_turn, winner).

        Note:
            game_data: A list of 7 previous moves and the whos_turn layer is -1 or 1.
            step: The expected position of next move.
            whos_turn: The move should be made by black (-1) or white (1).
            winner: The final winner of the game.

        """

        return self.train_data