#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Training Files.

PGN Auto : convert a PGN to a list of FEN and good moves (UCI Format)
"""
import chess  # Python Chess library

print("PGN Auto - Convert a PGN to a list of FEN and good moves (UCI Format)")
print("Enter nothing to quit")

pgn_file = None

while pgn_file != "":
    pgn_file = input("PGN file location : ")
    try:
        pgn_file = open(pgn_file)
    except FileNotFoundError:
        print(f"Error during reading file : File not found : {pgn_file}")
    except (UnicodeError, UnicodeDecodeError):
        print(f"Error during reading file : Invalid characters in {pgn_file}")
