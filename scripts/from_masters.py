#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""From masters.

Generate training files from Lichess masters' database.
"""
import time

import berserk
import chess

client = berserk.Client()


def from_masters(fen: str, depth: int, min_rate: float) -> list:
    """Get training file from Lichess masters' database.

    :param str fen: Starting FEN.
    :param int depth: Search depth.
    :param float min_rate: Minimum move rating.
    :return: Good moves ["<FEN>\n<Good move>", "<FEN>\n<Good move>"]
    :rtype: list
    """
    result: list = list()
    while True:
        try:
            response: dict = client.opening_explorer.masters(fen)
            break
        except (berserk.exceptions.ResponseError, berserk.exceptions.ApiError):
            print("⚠️", end=" ", flush=True)
            time.sleep(15)
    for move in response['moves']:
        total_moves = response['white'] + response['draws'] + response['black']
        if move['white'] + move['draws'] + move['black'] > min_rate * total_moves:
            result.append(f"{fen}\n{move['uci']}")
            if depth > 1:
                board = chess.Board(fen)
                board.push(chess.Move.from_uci(move['uci']))
                result.extend(from_masters(board.fen(), depth - 1, min_rate))
    return result


if __name__ == "__main__":
    search_depth = int(input("Step 1: Search depth\t"))
    search_min_rate = int(input("Step 2: Minimum rating\t")) / 100
    print("Step 3: Generating...", end=" ", flush=True)
    data = from_masters("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        search_depth, search_min_rate)
    print("Done.")
    open("training_files/" + input("Step 4: Output file\t"), 'w').write("\n\n".join(data))
