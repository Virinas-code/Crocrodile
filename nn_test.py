#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Test.

Test time for evaluation function.
"""
import time
import chess
import my_engine.nn

nn = my_engine.nn.NeuralNetwork()


BOARD = chess.Board().fen()
MOVE = "e2e4"
start = time.time()
for a in range(1000):
    nn.check_move(BOARD, MOVE)
end = time.time()
print("Time :", end - start)
