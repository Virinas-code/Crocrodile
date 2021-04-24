#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NN Auto.

Simple tool to create Crocrodile NN Training files.
"""
import chess  # python-chess

if __name__ == '__main__':
    print("Enter nothing to quit.")
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
                good_moves.append(GOOD_MOVE_INPUT)
            good_moves.pop()
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
                read = open("my_engine/training_mauvaiscoups_ouverture_blancs.txt")
                read_content = read.read()
                file = open("my_engine/training_mauvaiscoups_ouverture_blancs.txt", 'a')
                if read_content[-2:] == "\n\n":
                    pass
                elif read_content[-1] == "\n":
                    file.write("\n")
                elif read_content[-1] != "\n":
                    file.write("\n\n")
                file.write(STRING[:-2])
                file.close()
                read.close()
                read = open("my_engine/training_boncoups_ouverture_blancs.txt")
                read_content = read.read()
                file = open("my_engine/training_boncoups_ouverture_blancs.txt", 'a')
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
