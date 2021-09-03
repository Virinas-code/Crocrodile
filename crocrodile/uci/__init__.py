#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile UCI.

Created by @Virinas-code.
"""
import chess
import my_engine
# ====== IDLE ======
# import os
# os.chdir("../")
# ==== END IDLE ====

NoneType = type(None)


class UCI:
    """Base class for Crocrodile UCI."""

    def __init__(self):
        """Initialize UCI."""
        self.name = "Crocrodile"
        self.author = "Created by Virinas-code / Co-developed by ZeBox / "
        self.author += "Tested by PerleShetland"
        # default true when NN complete
        self.options = {"Hash": "77", "NeuralNetwork": "false"}
        self.debug_mode = False
        self.board = chess.Board()
        self.positionned = False
        print(self.name, self.author.lower())

    def run(self):
        """Run UCI input."""
        inner = ""
        while inner != "quit":
            inner = input()
            self.uci_parse(inner)

    def uci_parse(self, string: str) -> NoneType:
        """
        Parse UCI command.

        :param string: UCI command
        :type string: str
        :return: Nothing
        :rtype: NoneType
        """
        self.info("Received " + string)
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
        elif command == "position" and len(args) > 1:
            self.position(args[1:])
        elif command == "setoption" and len(args) > 1:
            self.set_option(args[1:])
        elif command == "crocrodile.display" and self.debug_mode is True:
            print(self.board)
        elif command == "crocrodile.bruh" and self.debug_mode is True:
            print("Yes BRUH.")
            print("https://lichess.org/83hsKBy2/black#2")
        elif len(args) == 0:
            pass
        elif len(args) == 1:
            print("Unknown command: {0} with no arguments".format(string))
        else:
            print("Unknown command: {0}".format(string))

    def info(self, msg: str) -> NoneType:
        """
        Print debug information.

        :param msg: Message to display.
        :type msg: str
        :return: Nothing
        :rtype: NoneType
        """
        if self.debug_mode:
            print("info string", msg)

    def uci(self) -> NoneType:
        """
        Uci UCI command.

        :return: Nothing
        :rtype: NoneType
        """
        print("id name {0}".format(self.name))
        print("id author {0}".format(self.author))
        print()
        print("option name Hash type spin default 77")
        # default true when NN complete
        print("option name NeuralNetwork type check default false")
        print("uciok")

    def debug(self, boolean: str) -> NoneType:
        """
        Enable or disable debug mode.

        :param boolean: 'on' or 'off'
        :type boolean: str
        :return: Nothing
        :rtype: NoneType
        """
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

    def position(self, args: list) -> None:
        """
        Position UCI command.

        Change current position.
        """
        next_arg = 0
        if args[0] == "startpos":
            self.board = chess.Board()
            self.positionned = True
            next_arg = 1
        elif args[0] == "fen" and len(args) > 6:
            self.board = chess.Board(" ".join(args[1:7]))
            self.positionned = True
            next_arg = 7
        else:
            print("Unknow syntax: position", " ".join(args))
        if next_arg and len(args) > next_arg + 1:
            self.info(args[next_arg])
            self.info(args[next_arg + 1:])
            for uci_move in args[next_arg + 1:]:
                try:
                    self.board.push(chess.Move.from_uci(uci_move))
                except ValueError:
                    print("Unknow UCI move:", uci_move)

    def new_game(self):
        self.board = chess.Board()
        self.positionned = False


if __name__ == '__main__':
    uci = UCI()
    uci.run()
