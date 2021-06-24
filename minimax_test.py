#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Test.

Test time for minimax.
"""
import time
import chess
import my_engine.engine

engine = my_engine.engine.EngineBase("Crocrodile v1 - test edition", "Virinas-code")


start = time.time()
board = chess.Board()
for a in range(1):
    engine.minimax_std(board, 4, True)
    board.push(list(board.legal_moves)[0])
end = time.time()
print("Time :", end - start)
