#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Training Files.

PGN Auto : convert a PGN to a list of FEN and good moves (UCI Format)
"""
import chess.pgn  # Python Chess library for PGN reading

print("PGN Auto - Convert a PGN to a list of FEN and good moves (UCI Format)")

PGN_FILE = None

DESTINATION_FILE = input("Destination file : ")
print("Checking if file exists...", end=" ")
try:
    open(DESTINATION_FILE)
    print("Yes.")
except FileNotFoundError:
    print("No.")
    print("Creating file...", end=" ")
    open(DESTINATION_FILE, "w")
    print("Done.")
print("Detecting \\n...", end=" ")
with open(DESTINATION_FILE) as READ:
    READ_CONTENT = READ.read()
    if READ_CONTENT and len(READ_CONTENT) < 2 and READ_CONTENT[-1] == "\n":
        print("Done.")
        print("Fixing problems...", end=" ")
        FILE = open(READ.name, 'w')
        FILE.write("")
        FILE.close()
        print("Done.")
    elif len(READ_CONTENT) > 2 and READ_CONTENT[-1] == "\n":
        if READ_CONTENT[-2] == "\n":
            print("Done.")
        else:
            print("Done.")
            print("Fixing problems...", end=" ")
            FILE = open(READ, 'a')
            FILE.write("\n")
            FILE.close()
            print("Done.")
    else:
        print("Done.")

while PGN_FILE != "":
    print("Enter nothing to quit")
    # % Input PGN file
    PGN_FILE = input("PGN file location : ")
    try:
        PGN_FILE = open(PGN_FILE)
    except FileNotFoundError:
        if PGN_FILE:
            print(f"Error during reading file : File not found : {PGN_FILE}")
        else:
            print("Exiting")
        continue
    except (UnicodeError, UnicodeDecodeError):
        print(f"Error during reading file : Invalid characters in {PGN_FILE}")
        continue
    # % Parse PGN file
    print("Parsing PGN...", end=" ")
    game = chess.pgn.read_game(PGN_FILE)
    print("Done.")
    head = game.headers
    head_game = f"{head['White']} vs. {head['Black']}"
    head_event = f"{head['Event']} ({head['Site']})"
    print(f"Game : {head_game} during {head_event}")
    print(f"Termination : {head['Termination']} - {head['Result']}")
    board = game.board()
    output = str()
    for move in game.mainline_moves():
        output += board.fen() + "\n"
        output += move.uci() + "\n"
        output += "\n"
        board.push(move)
    print("Writing to file...", end=" ")
    with open(DESTINATION_FILE, 'a') as FILE:
        FILE.write(output)
    print("Done.")
