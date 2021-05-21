#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile UCI.

Created by @Virinas-code.
"""
import my_engine
import chess
# ====== IDLE ======
import os
os.chdir("../")
# ==== END IDLE ====


class UCI:
    """Base class for Crocrodile UCI."""

    def __init__(self):
        """Initialize UCI."""
        self.name = "Crocrodile"
        self.author = "Created by Virinas-code / Co-developed by ZeBox / "
        self.author += "Tested by PerleShetland"
        self.options = {"Hash": "77", "NeuralNetwork": "false"}  # default true when NN complete
        self.debug_mode = False
        self.board = chess.Board()
        print(self.name, self.author.lower())

    def run(self):
        """Run UCI input."""
        inner = ""
        while inner != "quit":
            inner = input()
            self.uci_parse(inner)

    def uci_parse(self, string):
        """Parse UCI command."""
        args = [value for value in string.split(" ") if value != ""]
        if string != "":
            command = args[0]
        else:
            command = ""
        if command == "uci":
            self.uci()
        elif command == "isready":
            print("readyok")
        elif command == "debug" and len(args) > 1:
            self.debug(args[1])
        elif command == "quit":
            pass
        elif command == "register":
            print("No support for register")
        elif command == "ucinewgame":
            self.new_game()
        elif command == "go" and len(args) > 1:
            self.go(args[1:])
        elif command == "setoption" and len(args) > 1:
            self.set_option(args[1:])
        elif command == "crocrodile.display" and self.debug_mode == True:
            print(self.board)
        elif command == "crocrodile.bruh" and self.debug_mode == True:
            print("Yes BRUH.")
            print("https://lichess.org/83hsKBy2/black#2")
        elif len(args) == 0:
            pass
        elif len(args) == 1:
            print("Unknown command: {0} with no arguments".format(string))
        else:
            print("Unknown command: {0}".format(string))

    def uci(self):
        """Uci UCI command."""
        print("id name {0}".format(self.name))
        print("id author {0}".format(self.author))
        print()
        print("option name Hash type spin default 77")
        print("option name NeuralNetwork type check default false")  # default true when NN complete
        print("uciok")

    def debug(self, boolean):
        """Enable or disable debug mode."""
        if boolean == "on":
            self.debug_mode = True
        elif boolean == "off":
            self.debug_mode = False
        else:
            print("Unknown debug mode: {0}".format(boolean))

    def set_option(self, args) -> None:
        """
        Setoption UCI command.

        Configure Crocrodile.
        """
        if len(args) > 3 and args[0] == "name" and args[2] == "value":
            if args[1] in self.options:
                self.options[args[1]] = args[3:]
            else:
                print("Unknow option:", args[1])
        else:
            print("Unknow syntax: setoption", " ".join(args))

    def go(self, args: list) -> None:
        """
        Go UCI command.

        Start calculating.
        """
        print("In developpement.")
    
    def new_game(self):
        self.board = chess.Board()


if __name__ == '__main__':
    uci = UCI()
    uci.run()
