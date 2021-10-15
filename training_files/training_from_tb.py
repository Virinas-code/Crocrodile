#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates training files for training.
Uses random FEN generation.
"""
import random
import sys
import chess
import requests


def main(positions: int = 500000, dtm: int = 1, piece: str = "Q", destination: str = "training_files/tb.txt") -> None:
    returned: list = []
    for iters in range(positions):
        print(f"Generating position #{iters + 1}...", end="\r")
        board = chess.Board(None)
        while not board.is_valid():
            board = chess.Board(None)
            squares = list(range(64))
            random_white_king = random.choice(squares)
            del squares[random_white_king]
            random_black_king = random.choice(squares)
            squares.remove(random_black_king)
            random_piece = random.choice(squares)
            board.set_piece_at(random_white_king, chess.Piece.from_symbol("K"))
            board.set_piece_at(random_black_king, chess.Piece.from_symbol("k"))
            board.set_piece_at(random_piece, chess.Piece.from_symbol(piece))

        request = requests.get("https://tablebase.lichess.ovh/standard",
                               params={"fen": board.fen()})
        try:
            result = request.json()
            if result["dtm"] <= dtm:
                full = board.fen() + "\n" + result["moves"][0]["uci"] + "\n\n"
                if full not in returned:
                    returned.append(full)
        except:
            pass
    open(destination, "w").write("".join(returned))



if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])
