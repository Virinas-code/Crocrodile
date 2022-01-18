#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Masters Train.

Training sessions for Crocrodile on masters games.
"""
import chess
import random
import my_engine.nn
import typing

neural_network = my_engine.nn.NeuralNetwork()
MAX_MOVES = 6000
all_good_moves: list = open(
    "training_files/training_from_masters.txt").read().split("\n\n")[:-1]


def generate_bad_move(board: chess.Board) -> list:
    """Generates bad moves for Crocordile Masters Train.

    :param chess.Board board: The board passed.
    :param list good_moves: Good moves in board.
    :return: A list of the bad moves of board.
    :rtype: list
    :raises ValueError: Raised when a invalid board is passed.

    """
    moves: list = list(board.legal_moves)
    for move in moves:
        test_couple: str = f"{board.fen()}\n{move.uci()}"
        if test_couple in all_good_moves:
            moves.remove(move)
    if len(moves) == 0:
        return 0
    bad_move: chess.Move = random.choice(moves)
    return bad_move


def main():
    """Main function
    """
    iters = int(input("Iterations : "))
    mutation_rate = float(input("Mutation rate : "))
    mutation_change = float(input("Mutation change : "))
    config = {"mutation_rate": mutation_rate,
              "mutation_change": mutation_change}
    for iter in range(iters):
        generate_count = 0
        good_moves: list = list()
        GOOD_MOVES_QUANTITY: int = random.randint(
            0, MAX_MOVES / 2) + random.randint(0, MAX_MOVES / 2)
        for i in range(GOOD_MOVES_QUANTITY):
            generate_count += 1
            print(
                f"Generating good moves... ({generate_count}/{GOOD_MOVES_QUANTITY})", end="\r")
            test = random.choice(all_good_moves)
            while test in good_moves:
                test = random.choice(all_good_moves)
            good_moves.append(test)
        print("Generating good moves... Done.")
        BAD_MOVES_QUANTITY: int = MAX_MOVES - GOOD_MOVES_QUANTITY
        bad_moves: list = list()
        viewed_positions: list = list()
        generate_count = 0
        for i in range(BAD_MOVES_QUANTITY):
            generate_count += 1
            print(
                f"Generating bad moves... ({generate_count}/{BAD_MOVES_QUANTITY})", end="\r")
            random_position: str = random.choice(all_good_moves)
            while random_position in viewed_positions:
                random_position: str = random.choice(all_good_moves)
            viewed_positions.append(random_position)
            position: chess.Board = chess.Board(random_position.split("\n")[0])
            bad_move_for_pos: typing.Union[chess.Move,
                                           int] = generate_bad_move(position)
            while bad_move_for_pos == 0:
                random_position: str = random.choice(all_good_moves)
                while random_position in viewed_positions:
                    random_position: str = random.choice(all_good_moves)
                viewed_positions.append(random_position)
                position: chess.Board = chess.Board(
                    random_position.split("\n")[0])
                bad_move_for_pos: typing.Union[chess.Move, int] = generate_bad_move(
                    position)
            bad_moves.append(f"{position.fen()}\n{bad_move_for_pos.uci()}")
        good_moves_string: str = "\n\n".join(good_moves)
        bad_moves_string: str = "\n\n".join(bad_moves)
        """
        good_moves = list()
        good_moves_total = random.randint(1, 6000)
        for i in range(good_moves_total):
            test = random.choice(all_good_moves)
            while test in good_moves:
                test = random.choice(all_good_moves)
            good_moves.append(test)
        bad_moves = str()
        good_moves_string = str()
        for position in good_moves:
            board = chess.Board(position.split("\n")[0])
            move = [chess.Move.from_uci(position.split("\n")[1])]
            bad_moves_for_pos = generate_bad_moves(board, move)
            for bad_move in bad_moves_for_pos:
                bad_moves += f"{board.fen()}\n{bad_move.uci()}\n\n"
            good_moves_string += f"{board.fen()}\n{move[0].uci()}\n\n"
        """
        neural_network.masters_genetic_train(
            good_moves_string, bad_moves_string, config)


if __name__ == '__main__':
    main()
