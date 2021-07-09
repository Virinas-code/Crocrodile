#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Setup.

Setup Crocrodile chess engine.
"""
import click
import os
import colorama
import tempfile

file = tempfile.mkstemp(prefix="setup-")[1]
colorama.init()


def error(msg):
    print(colorama.Style.BRIGHT + colorama.Fore.RED + "ERROR:", colorama.Style.RESET_ALL + msg, "You can find complete logs at", file)


@click.command()
def install():
    print("Installing dependencies... (checking for python3)", end="\r", flush=True)
    test = os.system(f"python3 --version > {file}")
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




if __name__ == '__main__':
    install()
