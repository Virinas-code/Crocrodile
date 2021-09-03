#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Training.

Back to basics.

:author: Virinas-code and ZeBox
"""
import json
import chess
import my_engine.nn

neural_network = my_engine.nn.NeuralNetwork()
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
    """
    Parse good moves in good_moves_file. good_moves_file is only a file path.

    :param good_moves_file: Path to the good moves file.
    :type good_moves_file: str
    :return: The list of FENs + good move.
    :rtype: list
    """
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


def generate_bad_moves(good_move_pos: str, good_moves_list, bad_moves_list):
    """
    Generate bad moves for position.

    :param good_move_pos: Godd move in position (FEN + good move)
    :type good_move_pos: str
    """
    result = list()
    position = chess.Board(good_move_pos.split("\n")[0])
    for move in position.legal_moves:
        generated_position = position.fen() + "\n" + move.uci()
        if generated_position not in good_moves_list and generated_position not in bad_moves_list:
            result.append(generated_position)
    return result


def main():
    """Main function."""
    GOOD_MOVES_FILE = ask()
    GOOD_MOVES_LIST = parse_good_moves(GOOD_MOVES_FILE)
    GOOD_MOVES_TRAIN = list()
    BAD_MOVES_LIST = list()
    for good_move in GOOD_MOVES_LIST:
        GOOD_MOVES_TRAIN.append(good_move)
        print(f"########## Training #{len(GOOD_MOVES_TRAIN)} ##########")
        BAD_MOVES_LIST.extend(generate_bad_moves(
            good_move, GOOD_MOVES_LIST, BAD_MOVES_LIST))
        print(
            f"Bad moves: {len(BAD_MOVES_LIST)} / Good moves: {len(GOOD_MOVES_TRAIN)}")
        print("Training...", end="\r")


if __name__ == '__main__':
    main()
