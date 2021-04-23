#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NN Auto.

Simple tool to create Crocrodile NN Training files.
"""
import chess  # python-chess

if __name__ == '__main__':
    print("Enter nothing to quit.")
    inner = "continue"
    while inner != "":
        inner = input("FEN: ")
        if inner:
            board = chess.Board(inner)
            print("Legal moves :", ", ".join(move.uci() for move in board.legal_moves))
            print("Good moves :")
            good_moves = list()
            good_move_input = "continue"
            while good_move_input != "":
                good_move_input = input()
                good_moves.append(good_move_input)
            good_moves.pop()
            moves = list()
            for move in board.legal_moves:
                moves.append(move)
            for move in good_moves:
                uci_move = chess.Move.from_uci(move)
                moves.remove(uci_move)
            print("Bad moves :", ", ".join(move.uci() for move in moves))
            print("Generated file extension :")
            print("====================")
            string = str()
            for move in moves:
                string += board.fen() + "\n"
                string += move.uci() + "\n\n"
            print(string)
            print("====================")
            confirm = input("Autowrite in file ? [Y/n] ")
            if confirm.lower() == "n" or confirm.lower() == "no":
                pass
            else:
                read = open("my_engine/training_boncoups_ouverture_blancs.txt")
                read = read.read()
                file = open("my_engine/training_mauvaiscoups_ouverture_blancs.txt", 'a')
                if read[-2:] == "\n\n":
                    pass
                elif string[-1] == "\n":
                    file.write("\n")
                elif string[-1] != "\n":
                    file.write("\n\n")
                file.write(string[:-2])
                file.close()
                print("Done.")
