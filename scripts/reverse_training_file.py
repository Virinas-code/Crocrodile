#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reverse_training_file.

Reverse positions and good moves in file.
"""


def reverse_training_file(path, dest):
    """
    Reverse training file at path.

    :param str path: Path to training file.
    :param str dest: Destination file.
    :return: None
    """
    print(f"Reading {path}  ...", end=" ")
    file = open(path).read()
    print(f"Done.\nParsing {path}  ...", end=" ")
    content = file.split("\n\n")
    print(f"Done.\nReversing {path}...", end=" ")
    content.reverse()
    print(f"Done.\nJoining {path}  ...", end=" ")
    content = "\n\n".join(content)
    print(f"Done.\nWriting {dest}  ...", end=" ")
    dest = open(dest, "w")
    dest.write(content)
    print("Done.")


if __name__ == "__main__":
    input_path = "training_files/" + input("1. Input file  : ")  # Path to training file
    output_path = "training_files/" + input("2. Output file : ")  # Destination file
    reverse_training_file(input_path, output_path)
