# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
MyEngine.

UCI System
"""
import time
import sys
import chess


if not (sys.version_info > (3, 0)):
    # Python 2 code in this block
    FileExistsError = OSError


class UCI:
    """MyEngine UCI System."""

    def __init__(self, engine):
        """UCI Main."""
        self.engine = engine
        if self.engine.name == "MyEngine":
            print("UCI Connected")
        else:
            print("MyEngine UCI Control for " + repr(engine))
            self.inner = [" "]
            self.debug_state = False
            self.debug_file = sys.stderr
            self.board = chess.Board(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/"
                                         + "RNBQKBNR w KQkq - 0 1")

    def run(self):
        """Run loop."""
        print(dir(chess.Piece(chess.PAWN, chess.BLACK)))
        while self.inner != "quit":
            try:
                self.inner = input()
            except EOFError:
                self.inner = "null"
                pass
            self.evaluate_uci(self.inner)
        self.debug_file.close()

    def evaluate_uci(self, inner):
        """Evaluate UCI command."""
        inner = inner.split(" ")
        # print("info string", inner)
        if inner[0] == "uci":
            self.uci()
        elif inner[0] == "debug" and len(inner) > 1:
            self.debug(inner[1])
        elif inner[0] == "quit":
            if self.debug_state:
                self.debug_log("quit")
        elif inner[0] == "isready":
            print("readyok")
        elif inner[0] == "ucinewgame":
            self.board = chess.Board(fen="rnbqkbnr/pppppppp/8/8/8/8/P"
                                         + "PPPPPPP/RNBQKBNR w KQkq - 0 1")
        elif inner[0] == "display":
            print(self.board)
        elif inner[0] == "display-engine":
            print(self.engine.board)
        elif inner[0] == "position":
            self.position(inner[1:])
        elif inner[0] == "go":
            self.go(inner[1:])
        elif inner[0] == "setoption":
            self.setoption(inner[1:])
        elif inner[0] == "null":
            pass
        else:
            print("Unknow command: " + " ".join(inner))
            if self.debug_state:
                self.debug_log("Unknow command: "
                               + ' '.join(inner))

    def uci(self):
        """UCI uci command."""
        if self.debug_state:
            self.debug_log("uci")
        print("id name " + self.engine.name)
        print("id author " + self.engine.author)
        print()
        print("option name Hash type Spin default 1")
        print("option name Clear Hash type button")
        print("option name Ponder type check default false")
        print("option name NalimovPath type string")
        print("option name NalimovCache type spin default 4")
        print("option name OwnBook type check default true")
        print("option name MultiPV type spin default 1")
        print("uciok")

    def debug(self, mode):
        """UCI debug command."""
        if mode == "on":
            self.debug_state = True
            try:
                self.debug_file = open("uci.log", 'w')
            except FileExistsError:
                self.debug_file = open("uci.log", 'w')
            self.debug_log("debug on")
        else:
            if self.debug_state:
                self.debug_log("debug off")
            self.debug_state = False
            self.debug_file = sys.stderr

    def debug_log(self, msg):
        """Tool for logs."""
        self.debug_file.write(time.strftime("%d/%m/%Y %H:%M:%S")
                              + " : " + msg + "\n")

    def position(self, args):
        """UCI position command."""
        if len(args) > 6 and args[0] == "fen":
            self.board.set_fen(" ".join(args[1:7]))
            self.engine.board.set_fen(" ".join(args[1:7]))
        elif len(args) > 0 and args[0] == "startpos":
            self.board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB"
                               + "QKBNR w KQkq - 0 1")
            self.engine.board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB"
                                      + "QKBNR w KQkq - 0 1")
        else:
            print("Bad command: position " + ' '.join(args))

        if len(args) > 2 and args[1] == "moves":
            print("moves")
            for move in args[2:]:
                try:
                    self.board.push(chess.Move.from_uci(move))
                    self.engine.board.push(chess.Move.from_uci(move))
                except ValueError:
                    print("Bad move: " + move + " for position command")
        elif len(args) > 8 and args[7] == "moves":
            print("moves")
            for move in args[8:]:
                try:
                    self.engine.board.push(chess.Move.from_uci(move))
                except ValueError:
                    print("Bad move: " + move + " for position command")

    def go(self, args):
        self.engine.minimax(self.board, 3, self.board.turn, False)

    def setoption(self, args):
        """UCI setoption command."""
        if len(args) > 1:
            if args[0] == "name":
                if args[1:] not in ["Hash", "Clear Hash", "Ponder", "NalimovPath", "NalimovCache", "OwnBook", "MultiPV"]:
                    print("No such option: {0}".format(' '.join(args[1:])))
