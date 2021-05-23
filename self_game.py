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


def calculate(calc_board, depth):
    """Run Crocrodile on board."""
    result = crocrodile.minimax_std(calc_board, depth, board.turn)
    print("* Score : {0}       ".format(result[0]), end="\r")
    time.sleep(2)
    return result[1]


ITERS = 0
print("********** Crocrodile self game **********")
while True:
    ITERS += 1
    print("* Game #{0}".format(ITERS))
    depth = int(input("* Depth : "))
    board = chess.Board()
    while not board.outcome():
        if board.turn == chess.WHITE:
            print("* > White / Black  ", end="\r")
        else:
            print("* White / > Black  ", end="\r")
        board.push(calculate(board, depth))
    print()
    print("* End FEN :", board.fen())
    if board.outcome().winner is not None:
        print("* Winner : {0}".format("white" if board.outcome().winner else "black"))
    else:
        print("* Draw")
