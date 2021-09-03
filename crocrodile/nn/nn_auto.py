#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NN Auto.

Simple tool to create Crocrodile NN Training files.
"""
import chess  # python-chess

if __name__ == '__main__':
    print("Enter nothing to quit.")
    FILE_GOOD = input("Good moves file : ")
    FILE_BAD = input("Bad moves file : ")
    INNER = "continue"
    while INNER != "":
        INNER = input("FEN: ")
        if INNER:
            board = chess.Board(INNER)
            print(board)
            print("Legal moves :", ", ".join(move.uci() for move in board.legal_moves))
            print("Good moves :")
            good_moves = list()
            GOOD_MOVE_INPUT = "continue"
            while GOOD_MOVE_INPUT != "":
                GOOD_MOVE_INPUT = input()
                if GOOD_MOVE_INPUT:
                    if chess.Move.from_uci(GOOD_MOVE_INPUT) in board.legal_moves:
                        if GOOD_MOVE_INPUT in good_moves:
                            print("This move is in the good moves.")
                        else:
                            good_moves.append(GOOD_MOVE_INPUT)
                    else:
                        print("This is not a legal move. It will be ignored.")
            moves = list()
            for move in board.legal_moves:
                moves.append(move)
            for move in good_moves:
                uci_move = chess.Move.from_uci(move)
                moves.remove(uci_move)
            print("Bad moves :", ", ".join(move.uci() for move in moves))
            print("Generated file extension :")
            print("========== GOOD MOVES ==========")
            GOOD_STRING = str()
            for move in good_moves:
                GOOD_STRING += board.fen() + "\n"
                GOOD_STRING += move + "\n\n"
            print(GOOD_STRING)
            print("========== BAD MOVES ==========")
            STRING = str()
            for move in moves:
                STRING += board.fen() + "\n"
                STRING += move.uci() + "\n\n"
            print(STRING)
            print("====================")
            confirm = input("Autowrite in file ? [Y/n] ")
            if confirm.lower() == "n" or confirm.lower() == "no":
                pass
            else:
                read = open(FILE_BAD)
                read_content = read.read()
                file = open(FILE_BAD, 'a')
                if read_content[-2:] == "\n\n":
                    pass
                elif read_content[-1] == "\n":
                    file.write("\n")
                elif read_content[-1] != "\n":
                    file.write("\n\n")
                file.write(STRING[:-2])
                file.close()
                read.close()
                read = open(FILE_GOOD)
                read_content = read.read()
                file = open(FILE_GOOD, 'a')
                if read_content[-2:] == "\n\n":
                    pass
                elif read_content[-1] == "\n":
                    file.write("\n")
                elif read_content[-1] != "\n":
                    file.write("\n\n")
                file.write(GOOD_STRING[:-2])
                file.close()
                read.close()
                print("Done.")
