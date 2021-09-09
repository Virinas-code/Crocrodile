#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Training.
Program for starting training algorithm.

:author: Virinas-code
"""
import sys
import os

print(os.getcwd())


def input_algorithm() -> int:
    """
    Input training algorithm.

    :return: Training algorithm indice.
    :rtype: int
    """
    print("Select an algorithm :")
    print("\t1. Masters training")
    print("\t2. Basics training")
    while True:
        algorithm = input("\tAlgorithm :")
        try:
            algorithm = int(algorithm)
        except ValueError:
            continue
        if algorithm not in list(range(1, 3)):
            continue
        else:
            return algorithm


def main(argv: list) -> int:
    """
    Main function.

    :param argv: sys.argv
    :type argv: list
    :return: Exit code.
    :rtype: int
    """
    algorithm = input_algorithm()
    if algorithm == 1:
        os.chdir("my_engine/")
        print(os.getcwd())
        exec(compile(open("masters_train.py").read(), "masters_train.py", "exec"))
    elif algorithm == 2:
        import my_engine.basics_train
        my_engine.basics_train.main()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
