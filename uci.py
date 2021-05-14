#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile UCI.

Created by @Virinas-code.
"""
import my_engine


class UCI:
    """Base class for Crocrodile UCI."""

    def __init__(self):
        """Initialize UCI."""
        self.name = "Crocrodile"
        self.author = "Created by Virinas-code / Co-developed by ZeBox / "
        self.author += "Tested by PerleShetland"
        self.debug_mode = False

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
        else:
            print("Unknown command: {0}".format(string))

    def uci(self):
        """Uci UCI command."""
        print("id name {0}".format(self.name))
        print("id author {0}".format(self.author))
        print()
        print("option name Log File type string default debug.log")
        print("uciok")

    def debug(self, boolean):
        """Enable or disable debug mode."""
        if boolean == "on":
            self.debug_mode = True
        elif boolean == "off":
            self.debug_mode = False
        else:
            print("Unknown debug mode: {0}".format(boolean))


if __name__ == '__main__':
    uci = UCI()
    uci.run()
