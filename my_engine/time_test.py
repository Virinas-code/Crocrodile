#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Test.

Test time for evaluation function.
"""
import time
import chess

start = time.time()
test_list = list()
board = chess.Board()
for a in range(10000):
    test_list.append(board.generate_legal_moves())
end = time.time()
print("Time :", end - start)
