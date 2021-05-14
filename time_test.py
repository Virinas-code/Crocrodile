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


start = time.time()
fen = chess.STARTING_FEN
for a in range(1000):
    nn.check_move(fen, "e2e4")
end = time.time()
print("Time :", end - start)
