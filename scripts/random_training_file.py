#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_training_file.

Random positions and good moves in file.
"""
import random


def random_training_file(path, dest):
    """
    Random training file at path.

    :param str path: Path to training file.
    :param str dest: Destination file.
    :return: None
    """
    print(f"Reading {path}  ...", end=" ")
    file = open(path).read()
    print(f"Done.\nParsing {path}  ...", end=" ")
    content = file.split("\n\n")
    print(f"Done.\nRandomizing {path}...", end=" ")
    random.shuffle(content)
    print(f"Done.\nJoining {dest}  ...", end=" ")
    content = "\n\n".join(content)
    print(f"Done.\nWriting {dest}  ...", end=" ")
    dest = open(dest, "w")
    dest.write(content)
    print("Done.")


if __name__ == "__main__":
    input_path = "training_files/" + input("1. Input file  : ")  # Path to training file
    output_path = "training_files/" + input("2. Output file : ")  # Destination file
    random_training_file(input_path, output_path)
