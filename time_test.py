#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Test.

Test time for evaluation function.
"""
import time
import chess
import my_engine.engine

engine = my_engine.engine.EngineBase("Crocrodile v1 - test edition", "Virinas-code")


start = time.time()
board = chess.Board()
for a in range(160000):
    engine.evaluate(board)
end = time.time()
print("Time :", end - start)
