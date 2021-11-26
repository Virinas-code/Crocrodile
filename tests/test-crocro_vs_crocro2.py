#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Crocro vs Crocro 2.

Match 2 Crocrodile versions from crocrodile and crocrodile2.
"""
import os
import time

import chess

import crocrodile.engine
import crocrodile2.engine


def show_config(mode: str, crocro1: crocrodile.engine.EngineBase,
                crocro2: crocrodile.engine.EngineBase, depth: int=0,
                total_time: int=0, increment: int=0):
    print("**********  Config   **********")
    print(f"White: {crocro1.name}")
    print(f"Black: {crocro2.name}")
    if mode == "1":
        print("Mode      : Time control")
        print(f"Total time: {total_time}")
        print(f"Increment : {increment}")
    elif mode == "2":
        print("Mode : Fixed depth")
        print(f"Depth: {depth}")


def show_result(result: chess.Board):
    print("**********  Result   **********")
    print(f"Result     : {result.outcome().result()}")
    print(f"Winner     : {result.outcome().winner if result.outcome().winner else 'Draw'}")
    terminations = {chess.Termination.CHECKMATE: "Checkmate",
                    chess.Termination.STALEMATE: "Stalemate",
                    chess.Termination.INSUFFICIENT_MATERIAL: "Insufficient material",
                    chess.Termination.THREEFOLD_REPETITION: "Threefold repetition",
                    chess.Termination.FIVEFOLD_REPETITION: "Fivefold repetition"}
    print(f"Termination: {terminations[result.outcome().termination]}")
    print(f"Final FEN  : {result.fen()}")



def depth_match(search_depth: int, crocro1: crocrodile.engine.EngineBase,
                crocro2: crocrodile.engine.EngineBase) -> chess.Board:
    """Match Crocrodile and Crocrodile 2 at fixed depth.

    :param int search_depth: Search depth.
    :param crocrodile.engine.EngineBase crocro1: Crocrodile 1
    :param crocrodile.engine.EngineBase crocro2: Crocrodile 2
    :return: Final board.
    :rtype: chess.Board
    """
    print("**********   Match   **********")
    board = chess.Board()
    continued = [chess.Termination.SEVENTYFIVE_MOVES, chess.Termination.FIFTY_MOVES]
    evaluation = 0
    while True:
        if board.turn == chess.WHITE:
            print(f"> White /   Black | {evaluation}", end="\r")
            evaluation, best_move = crocro1.minimax_nn(board, search_depth, True, float('inf'))
            board.push(best_move)
        else:
            print(f"  White / > Black | {evaluation}", end="\r")
            evaluation, best_move = crocro2.minimax_nn(board, search_depth, False, float('inf'))
            board.push(best_move)
        if board.outcome() and board.outcome().termination not in continued:
            break
    return board


def time_match(total_time: int, increment: int, crocro1: crocrodile.engine.EngineBase,
                crocro2: crocrodile.engine.EngineBase) -> chess.Board:
    """Match Crocrodile and Crocrodile 2 with time control.

    :param int search_depth: Search depth.
    :param crocrodile.engine.EngineBase crocro1: Crocrodile 1
    :param crocrodile.engine.EngineBase crocro2: Crocrodile 2
    :return: Final board.
    :rtype: chess.Board
    """
    print("**********   Match   **********")
    board = chess.Board()
    continued = [chess.Termination.SEVENTYFIVE_MOVES, chess.Termination.FIFTY_MOVES]
    evaluation = 0
    white_time = total_time
    black_time = 0
    while True:
        if board.turn == chess.WHITE:
            start = time.time()
            print(f"> White {white_time} /   Black {black_time} | {evaluation}", end="\r")
            if white_time < 30:
                search_depth = 2
            elif white_time < 2*60:
                search_depth = 3
            else:
                search_depth = 4
            evaluation, best_move = crocro1.minimax_nn(board, search_depth, True, float('inf'))
            board.push(best_move)
            end = time.time()
            white_time = int(white_time - (end - start) + increment)
        else:
            if black_time == 0:
                black_time = total_time
            start = time.time()
            print(f"  White {white_time} / > Black {black_time} | {evaluation}", end="\r")
            if black_time < 30:
                search_depth = 2
            elif black_time < 2*60:
                search_depth = 3
            else:
                search_depth = 4
            evaluation, best_move = crocro2.minimax_nn(board, search_depth, False, float('inf'))
            board.push(best_move)
            end = time.time()
            black_time = int(black_time - (end - start) + increment)
        if board.outcome() and board.outcome().termination not in continued:
            break
    return board


def main():
    print("########## Configure ##########")
    print("**********    Mode   **********")
    print("1. Time control")
    print("2. Fixed depth")
    mode = input("Mode: \t")
    if mode == "1":
        total_time = int(input("Total time: \t"))
        increment = int(input("Increment: \t"))
        print("##########  Match 1  ##########")
        crocro1 = crocrodile.engine.EngineBase("Crocrodile 1", "")
        crocro2 = crocrodile2.engine.EngineBase("Crocrodile 2", "")
        show_config(mode, crocro1, crocro2, total_time=total_time, increment=increment)
        result = time_match(total_time, increment, crocro1, crocro2)
        show_result(result)
        print("##########  Match 2  ##########")
        crocro1 = crocrodile.engine.EngineBase("Crocrodile 1", "")
        crocro2 = crocrodile2.engine.EngineBase("Crocrodile 2", "")
        show_config(mode, crocro2, crocro1, total_time=total_time, increment=increment)
        result = time_match(total_time, increment, crocro2, crocro1)
        show_result(result)
    elif mode == "2":
        search_depth = int(input("Depth: \t"))
        print("##########  Match 1  ##########")
        crocro1 = crocrodile.engine.EngineBase("Crocrodile 1", "")
        crocro2 = crocrodile2.engine.EngineBase("Crocrodile 2", "")
        show_config(mode, crocro1, crocro2, depth=search_depth)
        result = depth_match(search_depth, crocro1, crocro2)
        show_result(result)
        print("##########  Match 2  ##########")
        crocro1 = crocrodile.engine.EngineBase("Crocrodile 1", "")
        crocro2 = crocrodile2.engine.EngineBase("Crocrodile 2", "")
        show_config(mode, crocro2, crocro1, depth=search_depth)
        result = depth_match(search_depth, crocro2, crocro1)
        show_result(result)


if __name__ == '__main__':
    main()
