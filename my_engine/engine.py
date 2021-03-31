# -*- coding: utf-8 -*-
"""
MyEngine Engine base.

Base engine
"""
from __future__ import print_function
import chess
import chess.polyglot
import requests
import copy

PAWN_VALUE = 100
KNIGHT_VALUE = 290
BISHOP_VALUE = 310
ROOK_VALUE = 500
QUEEN_VALUE = 900
KING_VALUE = 0  # Infinity is too complex
PIECES_VALUES = {"p": PAWN_VALUE, "n": KNIGHT_VALUE, "b": BISHOP_VALUE,
                 "r": ROOK_VALUE, "q": QUEEN_VALUE, "k": KING_VALUE}
CENTRAL_SQUARES = [36, 35, 28, 27]
ELARGED_SQUARES = [45, 44, 43, 42, 37, 34, 29, 26, 21, 20, 19, 18]
SEVENTH_ROW = [55, 54, 53, 52, 51, 50, 49, 48]
SECOND_ROW = [15, 14, 13, 12, 11, 10, 9, 8]


def printi(*args):
    """Debug mode printer."""
    print("info string", args)


class EngineBase:
    """Engine base."""

    def __init__(self, name, author, board=chess.Board()):
        """Initialise engine."""
        self.name = name
        self.author = author
        self.board = board

    def evaluate(self, board):
        """Evaluate position."""
        white_score = 0
        black_score = 0
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -10000
            else:
                return 10000
        piece_map = board.piece_map()
        """if len(piece_map) <= 7:
            return float("inf")"""
        for piece in piece_map:
            if piece_map[piece].symbol().isupper():
                white_score += PIECES_VALUES[piece_map[piece].symbol().lower()]
                if piece in CENTRAL_SQUARES:
                    white_score += 10
                if piece in ELARGED_SQUARES:
                    white_score += 5
                if piece_map[piece].symbol().lower() == 'p' and piece in SEVENTH_ROW:
                    white_score += 20
            else:
                black_score += PIECES_VALUES[piece_map[piece].symbol()]
                if piece in CENTRAL_SQUARES:
                    black_score += 10
                if piece in ELARGED_SQUARES:
                    black_score += 5
                if piece_map[piece].symbol().lower() == 'p' and piece in SECOND_ROW:
                    black_score += 20
        if board.has_castling_rights(chess.WHITE):
            white_score += 30
        if board.has_castling_rights(chess.BLACK):
            black_score += 30
        return white_score-black_score

    def search(self, depth, board):
        """Search best move (Minimax from wikipedia)."""

    def minimax(self, board, depth, maximimize_white, subcall):
        """Minimax algorithm from Wikipedia."""
        if depth == 0 or board.is_game_over():
            evaluation = self.evaluate(board)
            """if evaluation == float("inf"):
                r = requests.get("http://tablebase.lichess.ovh/standard?fen={0}".format(board.fen().replace(" ", "_"))).json()
                dtm = r["moves"][0]["dtm"]
                if board.turn == chess.WHITE:
                    if dtm:
                        if dtm > 0:
                            mate_in = -10000 + dtm
                        else:
                            mate_in = 10000 - dtm
                    else:
                        mate_in = 0
                else:
                    if dtm:
                        if dtm > 0:
                            mate_in = 10000 - dtm
                        else:
                            mate_in = -10000 + dtm
                    else:
                        mate_in = 0
                return mate_in, r["moves"][0]["uci"]
                # color will be mated in..."""
            attackers = board.attackers(board.turn, board.peek().to_square)
            if len(attackers) > 0:
                # print("retake ! board : \n", board, "\n last move :", board.peek())
                if board.turn == chess.WHITE:
                    evaluation += PIECES_VALUES[board.piece_map()[board.peek().to_square].symbol().lower()]
                else:
                    evaluation -= PIECES_VALUES[board.piece_map()[board.peek().to_square].symbol().lower()]
            return evaluation, chess.Move.from_uci("0000")
        if maximimize_white:
            value = -float('inf')
            for move in board.legal_moves:
                best_move = move
                break
            for move in board.legal_moves:
                e = chess.Board(fen=board.fen())
                e.push(move)
                evaluation = self.minimax(e, depth-1, False, True)[0]
                if value < evaluation:
                    value = evaluation
                    best_move = move
            return value, best_move
        else:
            # minimizing white
            value = float('inf')
            for move in board.legal_moves:
                best_move = move
                break
            for move in board.legal_moves:
                e = chess.Board(fen=board.fen())
                e.push(move)
                evaluation = self.minimax(e, depth-1, True, True)[0]
                if value > evaluation:
                    value = evaluation
                    best_move = move
            return value, best_move
        """depth = int(depth)
        if depth == 1:
            return self.evaluate(board)
        else:
            moves_to_get = list()
            eval = -float("inf") if board.turn == chess.BLACK else float('inf')
            for move in board.legal_moves:
                printi("move", move)
                test = chess.Board(fen=board.fen())
                moves_to_get.append(move)
                test.push(move)
                print(test)
                score = self.evaluate(test)
                if board.turn == chess.WHITE:
                    eval = max(score, eval)
                else:
                    eval = min(score, eval)
            return eval"""
