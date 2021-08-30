#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Training.

Back to basics.

:author: Virinas-code and ZeBox
"""
import json

config = json.loads(open("basics_train.json").read())
NoneType = type(None)


def ask() -> str:
    """
    Ask for inputs.

    :return: Good moves file.
    :rtype: str
    """
    good_moves_file = input("Good moves file (Enter for default) : ")
    if good_moves_file == "":
        good_moves_file = config["good_moves"]
    else:
        config["good_moves"] = good_moves_file
        open("basics_train.json", 'w').write(json.dumps(config))
    return good_moves_file


def parse_good_moves(good_moves_file: str) -> list:
    good_moves_content = open(good_moves_file).read().split("\n\n")
    # Remove \n at the end
    good_moves_content[-1] = good_moves_content[-1][:-1]
    good_moves_list = list()
    for move in good_moves_content:
        if move in good_moves_list:
            continue
        good_moves_list.append(move)
    print(good_moves_list)
    return good_moves_list


if __name__ == '__main__':
    GOOD_MOVES_FILE = ask()
    GOOD_MOVES_LIST = parse_good_moves(GOOD_MOVES_FILE)
    GOOD_MOVES_TRAIN = list()
    BAD_MOVES_LIST = list()
    for good_move in GOOD_MOVES_LIST:
        GOOD_MOVES_TRAIN.append(good_move)
        BAD_MOVES_LIST.append(generate_bad_moves(good_move))
