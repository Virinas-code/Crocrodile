#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Test.

Test time for evaluation function.
"""
import time
import chess
import my_engine

crocrodile = my_engine.EngineBase("Crocrodile", "Virinas-code")

start = time.time()
test_list = list()
board = chess.Board()
for a in range(10000):
    test_list.append(crocrodile.evaluate(board))
end = time.time()
print("Time :", end - start)
