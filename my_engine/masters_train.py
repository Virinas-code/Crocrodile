#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Masters Train.

Training sessions for Crocrodile on masters games.
"""
import chess
import random
import nn

neural_network = nn.NeuralNetwork()


def generate_bad_moves(board: chess.Board, good_moves: list) -> list:
    """Generates bad moves for Crocordile Masters Train.

    :param chess.Board board: The board passed.
    :param list good_moves: Good moves in board.
    :return: A list of the bad moves of board.
    :rtype: list
    :raises ValueError: Raised when a invalid board is passed.

    """
    moves: list = list(board.legal_moves)
    for move in good_moves:
        moves.remove(move)
    bad_moves: list = random.choices(moves, k=min(5, len(moves)))
    return bad_moves


def main():
    all_good_moves = open("training_files/training_from_masters.txt").read().split("\n\n")
    iters = int(input("Iterations : "))
    mutation_rate = float(input("Mutation rate : "))
    mutation_change = float(input("Mutation change : "))
    config = {"mutation_rate": mutation_rate, "mutation_change": mutation_change}
    for iter in range(iters):
        good_moves = list()
        for i in range(30):
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
        neural_network.masters_genetic_train(good_moves_string[:-2], bad_moves[:-2], config)


if __name__ == '__main__':
    main()
