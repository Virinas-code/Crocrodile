#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Setup.

Setup Crocrodile chess engine.
"""
import sys
import os
import colorama
import tempfile

file = tempfile.mkstemp(prefix="setup-")[1]
colorama.init()


def error(msg):
    print(colorama.Style.BRIGHT + colorama.Fore.RED + "ERROR:", colorama.Style.RESET_ALL + msg, "You can find complete logs at", file)
    sys.exit(-1)


def install():
    print("Installing dependencies... (checking for python3)", end="\r", flush=True)
    test = os.system(f"python3 --version >> {file}")
    if test != 0:
        error("Python 3 not found. Please verify that Python 3 is installed on 'python3' command.")
    print("Installing dependencies... (checking for pip3)", end="\r", flush=True)
    test = os.system(f"python3 -m pip --version >> {file}")
    if test != 0:
        print("Installing dependencies... (installing pip3)", end="\r", flush=True)
        test = os.system(f"curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py >> {file}")
        if test != 0:
            error("Failed to download pip3.")
        test = os.system(f"python3 get-pip.py >> {file}")
        if test != 0:
            error("Failed to install pip3.")
    print("Installing dependencies... (installing requirements)", end="\r", flush=True)
    test = os.system(f"python3 -m pip install -r requirements.txt >> {file}")
    if test != 0:
        error("Failed to install dependencies.")


def client(sub_args):
    os.system("python3 client.py " + sub_args)


def help():
    print("Crocrodile setup program")
    print("========================")
    print()
    print("Usage: setup.py COMMAND [SUBCOMMAND] [OPTIONS]")
    print()
    print("Commands:")
    print("    install   : Install Crocrodile")
    print("    launch    : Launch Crocrodile")
    print()
    print("Launch :")
    print("    client    : Launch Lichess client")
    print()
    print("Client :")
    print("    -h, --help           : Show this message and exit")
    print("    -v, --verbose        : Show debug logs")
    print("    -q, --quiet          : Don't show any logs")
    print("    -c, --challenge \"user time increment color\" : Challenge user in time+increment, BOT is playing with color ('white' or 'black')")
    print("    -a, --auto           : Auto challenge BOTs")
    print("    -n, --neural-network : Enable Neural Network")
    print("    -u, --upgrade        : Upgrade to bot account")
    print("    -d, --dev            : Dev account")
    print()
    print("Options :")
    print("    -h, --help: Show this message and exit")


def parse_args():
    args = sys.argv
    if len(args) > 1:
        command = args[1]
        if command == "install":
            install()
        elif command == "launch":
            if len(args) > 2:
                command = args[2]
                if command == "client":
                    if len(args) > 3:
                        client(" ".join(args[3:]))
                    else:
                        client()
                else:
                    help()
                    print()
                    error(f"Unknow target: {command}.")
            else:
                help()
                print()
                error("No subcommand for launch.")
        elif command in ("-h", "--help"):
            help()
        else:
            help()
            print()
            error(f"Unknown command: {command}.")
    else:
        help()


if __name__ == '__main__':
    parse_args()
