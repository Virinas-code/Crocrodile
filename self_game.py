#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self Crocrodile game.

Game client - Crocrodile vs Crocrodile.
"""
import time
import chess
import my_engine

crocrodile = my_engine.EngineBase("Crocrodile", "Virinas-code")


def calculate(calc_board, clock_time, increment):
    """Run Crocrodile on board."""
    clock_time = clock_time + increment
    if clock_time > 1200:
        best_move = crocrodile.minimax(calc_board, 4, board.turn)[1]
    elif clock_time < 120:
        best_move = crocrodile.minimax(calc_board, 2, board.turn)[1]
    elif clock_time < 30:
        best_move = crocrodile.minimax(calc_board, 1, board.turn)[1]
    else:
        best_move = crocrodile.minimax(calc_board, 3, board.turn)[1]
    return best_move


ITERS = 0
print("********** Crocrodile self game **********")
while True:
    ITERS += 1
    print("* Game #{0}".format(ITERS))
    time_control = float(input("* Time control : ")) * 60
    time_increment = int(input("* Increment : "))
    board = chess.Board()
    white_clock = time_control
    black_clock = time_control
    while not board.outcome() and white_clock > 0 and black_clock > 0:
        print("* > White : {0}:{1} / Black : {2}:{3}  ".format(int(white_clock // 60),
                                                           int(white_clock % 60),
                                                           int(black_clock // 60),
                                                           int(black_clock % 60)
                                                           ), end="\r")
        start = time.time()
        board.push(calculate(board, white_clock, time_increment))
        end = time.time()
        white_clock -= end - start + time_increment
        print("* White : {0}:{1} / > Black : {2}:{3}  ".format(int(white_clock // 60),
                                                           int(white_clock % 60),
                                                           int(black_clock // 60),
                                                           int(black_clock % 60)
                                                           ), end="\r")
        start = time.time()
        board.push(calculate(board, black_clock, time_increment))
        end = time.time()
        black_clock -= end - start + time_increment
    print()
    print("* End FEN :", board.fen())
    if board.outcome().winner is not None:
        print("* Winner : {0}".format("white" if board.outcome().winner else "black"))
    else:
        print("* Winner : draw")
