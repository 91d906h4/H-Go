import torch
import random
import numpy as np

from network import H_GO
from collections import deque


class Node:
    def __init__(self, state: deque, step: int, value: float) -> None:
        """Node class

        Recording the state, step, and value of each move.

        Args:
            state (deque): The state of the game.
            step (int): The step of the move.
            value (float): The value of the move generated by the value network.

        """

        self.state      = state
        self.step       = step
        self.value      = value

        self.children   = list()


class HTS:
    def __init__(self, state: deque, depth: int, breadth: int, temperature: float, player: int, model: H_GO, device: torch.device) -> None:
        """HTS class

        A variant of MCTS (Monte Carlo Tree Search) to search for the best move.

        Args:
            state (deque): The state of the game.
            depth (int): The depth of the search tree.
            breadth (int): The breadth of the search tree.
            temperature (float): The randomness of the search for the step.
            player (int): The owner of this tree.
            model (H_GO): The AI model.
            device (torch.device): The device of the model.

        """

        self.state          = state.copy()
        self.depth          = depth
        self.breadth        = breadth
        self.temperature    = temperature
        self.player         = player
        self.model          = model
        self.device         = device

        self.root = Node(state=self.state, step=0, value=0)

    def _select(self, state: deque, player: int, randomize: bool) -> list:
        """select publi method

        Select the most posible moves.

        Args:
            state (deque): The state of the game.
            player (int): The current player.
            randomize (bool): Whether to randomize the selection.

        Retunrs:
            children (list): The possible children (moves) of input state.

        """

        # Set defualt value.
        children = list()

        # Get policy from model.
        input = list(np.full((1, 19, 19), fill_value=player)) + list(state)
        input = torch.tensor([input], dtype=torch.float).to(self.device)
        policy, _ = self.model(input)

        # Select the most probable moves.
        _, [policy] = torch.topk(policy, self.breadth)

        if randomize:
            # Remove the least probable moves.
            length = int(len(policy) * (1 - self.temperature))

            # Add random moves.
            policy = list(policy[:length]) + [random.randint(1, 361) for _ in range(self.breadth - length)]

        for step in policy:
            temp_state = state.copy()
            last_state = temp_state[-1]

            # Make the move.
            new_state = self._make_move(state=last_state, move=step, player=player)

            # Update state queue.
            temp_state.append(new_state)

            # Calculate value.
            input = list(np.full((1, 19, 19), fill_value=player*-1)) + list(state)
            input = torch.tensor([input], dtype=torch.float).to(self.device)
            _, value = self.model(input)

            # Add to children.
            children.append(Node(state=temp_state, step=step, value=value))

        return children

    def _expand(self, state: deque, depth: int, node: Node, player: int, randomize: bool) -> None:
        """expand publi method

        Expand the state with posible moves.

        Args:
            state (deque): The state of the game.
            depth (int): The current depth of this layer.
            node (Node): The current node.
            player (int): The current player.
            randomize (bool): Whether to randomize the selection.

        """

        if depth == 0: return
        else: depth -= 1

        if len(node.children) == 0:
            node.children = self._select(state=state, player=player, randomize=randomize)

        player *= -1

        for child in node.children:
            self._expand(state=state, depth=depth, node=child, player=player, randomize=False)

    def _search(self) -> list:
        """search publi method

        Search for the best move.

        Returns:
            result (list): The total value of each move.

        """

        result = list()

        for child in self.root.children:
            result.append(float(self._backpropagate(node=child)))

        return result

    def _backpropagate(self, node: Node) -> int:
        # Return if the node is leaf.
        if len(node.children) == 0:
            return node.value

        # Set defualt value.
        value = 0

        # Calculate the value of the node by traversing all the children.
        for child in node.children:
            value += self._backpropagate(child)

        return value

    def _make_move(self, state: list, move: int, player: int) -> list:
        """make_move private method

        Make the move on the state.

        Args:
            state (list): The state of the game.
            move (int): The move to make.
            player (int): The current player.

        Returns:
            state (list): The state after the move.

        """

        # Get row and col.
        row = move // 19
        col = move % 19

        # Make the move.
        state[row][col] = player

        return state

    def get_best_move(self) -> int:
        """get_best_move public method

        Get the best move from the root node.

        Returns:
            best_move (int): The best move.

        """

        # Expand the root node.
        self._expand(state=self.state, depth=self.depth, node=self.root, player=self.player, randomize=True)

        # Search for the best move.
        result = self._search()

        # Get moves from the root node.
        moves = [child.step for child in self.root.children]

        # Check this move is for which player.
        # If it's for black, then we need to select the move with
        # the smallest value, otherwise we need to select the move
        # with the largest value.
        if self.player == -1:
            reverse = False
        else:
            reverse = True

        # Rank the moves.
        sorted_moves = sorted(zip(moves, result), key=lambda x: x[1], reverse=reverse)
        best_moves = [int(x[0]) for x in sorted_moves]

        return best_moves