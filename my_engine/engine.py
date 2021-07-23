# -*- coding: utf-8 -*-
"""
MyEngine Engine base.

Base engine
"""
from __future__ import print_function
import math
import random
# import requests
# import copy
import csv
import time
import chess
import chess.polyglot
import my_engine.nn as nn

PAWN_VALUE = 130
KNIGHT_VALUE = 290
BISHOP_VALUE = 310
ROOK_VALUE = 500
QUEEN_VALUE = 901
KING_VALUE = 0  # Infinity is too complex
BISHOPS_PAIR = 50
PROTECTED_KING = 5
PIECES_VALUES = {"p": PAWN_VALUE, "n": KNIGHT_VALUE, "b": BISHOP_VALUE,
                 "r": ROOK_VALUE, "q": QUEEN_VALUE, "k": KING_VALUE}
CENTRAL_SQUARES = [36, 35, 28, 27]
ELARGED_SQUARES = [45, 44, 43, 42, 37, 34, 29, 26, 21, 20, 19, 18]
SEVENTH_ROW = [55, 54, 53, 52, 51, 50, 49, 48]
EIGHT_ROW = [56, 57, 58, 59, 60, 61, 62, 63]
SECOND_ROW = [15, 14, 13, 12, 11, 10, 9, 8]
FIRST_ROW = [0, 1, 2, 3, 4, 5, 6, 7]
VARIANTS = ['standard', 'chess960']
neural_network = nn.NeuralNetwork()


def csv_to_array(csv_path):
    """CSV file top Python array."""
    results = []
    with open(csv_path) as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            results.append(row)
    return results




def normalisation(val):
    """Sigmoïde modifiée."""
    return (1 / (1 + math.exp(-val))) * 2 - 1


def nn_opening_white_check_move(fen, move): # Move is UCI str
    """Neural network check if move is interesting for board."""
    board = chess.Board(fen=fen)
    pieces = board.piece_map()
    inputs_values = {'': 0, 'P': 0.1, 'N': 0.2, 'B': 0.3, 'R': 0.5, 'Q': 0.6, 'K': 0.7, 'p': -0.1,
                     'n': -0.2, 'b': -0.3, 'r': -0.5, 'q': -0.6, 'k': -0.7}
    inputs = []
    for count in range(64):
        if pieces.get(count, None):
            inputs.append(inputs_values.get(pieces[count].symbol(), 0))
        else:
            inputs.append(0)
    if board.has_kingside_castling_rights(chess.WHITE):
        inputs.append(1)
    else:
        inputs.append(0)
    if board.has_queenside_castling_rights(chess.WHITE):
        inputs.append(1)
    else:
        inputs.append(0)
    if board.has_kingside_castling_rights(chess.BLACK):
        inputs.append(1)
    else:
        inputs.append(0)
    if board.has_queenside_castling_rights(chess.BLACK):
        inputs.append(1)
    else:
        inputs.append(0)
    if board.has_legal_en_passant():
        inputs.append(chess.square_file(board.ep_square) / 10)
    else:
        inputs.append(-1)
    move = chess.Move.from_uci(move)
    from_square = move.from_square
    inputs.append(chess.square_file(from_square) / 10)
    inputs.append(chess.square_rank(from_square) / 10)
    to_square = move.to_square
    inputs.append(chess.square_file(to_square) / 10)
    inputs.append(chess.square_rank(to_square) / 10)
    inputs.append(1)
    # print("Inputs :", inputs)
    cache1 = list()
    cache2 = list()
    for count in range(38):
        cache1.append(0)
        cache2.append(0)
    output = float()
    for j in range(38):
        current = 0
        for i in range(74):
            current += inputs[i] * wa[i][j]
        cache1[j] = normalisation(current)
    # print(cache1)
    # print("Moyenne :", sum(cache1) / len(cache1))
    for j in range(38):
        current = 0
        for i in range(38):
            current += cache1[i] * wb[i][j]
        cache2[j] = normalisation(current)
    # print(cache2)
    # print("Moyenne :", sum(cache2) / len(cache2))
    for j in range(1):
        current = 0
        for i in range(38):
            current += cache2[i] * wc[i][j]
        output = current
    # print("Brut :", output)
    if output >= 0:
        output = 1
    else:
        output = -1
    # print("Output :", output)
    return output


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
        self.tb = dict()
        self.tb_limit = 10000000
        self.nn_tb = dict()

    @staticmethod
    def evaluate(board):
        """Evaluate position."""
        white_score = 0
        black_score = 0
        if board.is_stalemate():
            return 0
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -10000
            return 10000
        piece_map = board.piece_map()
        white_bishops = 0
        black_bishops = 0
        for piece in piece_map:
            if piece_map[piece].symbol().isupper():
                white_score += PIECES_VALUES[piece_map[piece].symbol().lower()]
                if piece in CENTRAL_SQUARES and not piece_map[piece].symbol() == "Q":
                    white_score += 10
                if piece in ELARGED_SQUARES and not piece_map[piece].symbol() == "Q":
                    white_score += 5
                if piece_map[piece].symbol() == 'P' and piece in SEVENTH_ROW:
                    white_score += 20
                if piece_map[piece].symbol() == 'P' and piece in EIGHT_ROW:
                    white_score += QUEEN_VALUE
                if piece_map[piece].symbol() == 'B':
                    white_bishops += 1
                if piece_map[piece].symbol() == 'Q' and len(piece_map) > 20:
                    white_score -= 30
                if piece_map[piece].symbol() == 'K':
                    if piece + 7 in piece_map and piece_map[piece + 7].symbol() == 'P' and len(piece_map) > 16:
                        white_score += PROTECTED_KING
                    if piece + 8 in piece_map and piece_map[piece + 8].symbol() == 'P' and len(piece_map) > 16:
                        white_score += PROTECTED_KING
                    if piece + 9 in piece_map and piece_map[piece + 9].symbol() == 'P' and len(piece_map) > 16:
                        white_score += PROTECTED_KING
            else:
                black_score += PIECES_VALUES[piece_map[piece].symbol()]
                if piece in CENTRAL_SQUARES and not piece_map[piece].symbol() == "q":
                    black_score += 10
                if piece in ELARGED_SQUARES and not piece_map[piece].symbol() == "q":
                    black_score += 5
                if piece_map[piece].symbol() == 'p' and piece in SECOND_ROW:
                    black_score += 20
                if piece_map[piece].symbol() == 'p' and piece in FIRST_ROW:
                    black_score += QUEEN_VALUE
                if piece_map[piece].symbol() == 'b':
                    black_bishops += 1
                if piece_map[piece].symbol() == 'k':
                    if piece - 7 in piece_map and piece_map[piece - 7].symbol() == 'p' and len(piece_map) > 16:
                        black_score += PROTECTED_KING
                    if piece - 8 in piece_map and piece_map[piece - 8].symbol() == 'p' and len(piece_map) > 16:
                        black_score += PROTECTED_KING
                    if piece - 9 in piece_map and piece_map[piece - 9].symbol() == 'p' and len(piece_map) > 16:
                        black_score += PROTECTED_KING
                if piece_map[piece].symbol() == 'q' and len(piece_map) > 28:
                    black_score -= 30
        if white_bishops >= 2:
            white_score += BISHOPS_PAIR
        if black_bishops >= 2:
            black_score += BISHOPS_PAIR
        if board.has_kingside_castling_rights(chess.WHITE):
            white_score += 7
        if board.has_kingside_castling_rights(chess.BLACK):
            black_score += 7
        if board.has_queenside_castling_rights(chess.WHITE):
            white_score += 7
        if board.has_queenside_castling_rights(chess.BLACK):
            black_score += 7
        # if board.peek().uci() in ['e1g1', 'e1c1']:
            # white_score += 101
            # print("white castle !")
        # if board.peek().uci() in ['e8g8', 'e8c8']:
            # black_score += 101
            # print("black castle !")
        if board.turn == chess.WHITE:
            white_score += len(list(board.legal_moves))
            board.push(chess.Move.from_uci("0000"))
            black_score += len(list(board.legal_moves))
            board.pop()
        else:
            black_score += len(list(board.legal_moves))
            board.push(chess.Move.from_uci("0000"))
            white_score += len(list(board.legal_moves))
            board.pop()
        return white_score-black_score

    def search(self, depth, board):
        """Search best move (Minimax from wikipedia)."""

    def minimax_std(self, board, depth, maximimize_white):
        """Minimax algorithm from Wikipedia without NN."""
        if depth == 0 or board.is_game_over():
            zobrist_hash = chess.polyglot.zobrist_hash(board)
            if zobrist_hash not in self.tb:
                self.tb[zobrist_hash] = self.evaluate(board)
                if len(self.tb) > self.tb_limit:
                    del self.tb[list(self.tb.keys())[0]]
            evaluation = self.tb[zobrist_hash]
            # evaluation = self.evaluate(board)
            attackers = board.attackers(board.turn, board.peek().to_square)
            if len(attackers) > 0:
                # Quiescent
                if board.turn == chess.WHITE:
                    evaluation += PIECES_VALUES[board.piece_map()\
                                                [board.peek().to_square].\
                                                symbol().lower()]
                else:
                    evaluation -= PIECES_VALUES[board.piece_map()\
                                                [board.peek().to_square].\
                                                symbol().lower()]
            return evaluation, chess.Move.from_uci("0000")
        if maximimize_white:
            value = -float('inf')
            legal_moves = list(board.legal_moves)
            list_best_moves = [legal_moves[0]]
            for move in legal_moves:
                test_board = chess.Board(fen=board.fen())
                test_board.push(move)
                evaluation = self.minimax_std(test_board, depth-1, False)[0]
                if move.uci() in ['e1g1', 'e1c1']:
                    evaluation += 11
                    # print('castle')
                if value == evaluation:
                    list_best_moves.append(move)
                elif value < evaluation:
                    value = evaluation
                    list_best_moves = [move]
            return value, random.choice(list_best_moves)
        else:
            # minimizing white
            value = float('inf')
            legal_moves = list(board.legal_moves)
            list_best_moves = [legal_moves[0]]
            for move in legal_moves:
                test_board = chess.Board(fen=board.fen())
                test_board.push(move)
                evaluation = self.minimax_std(test_board, depth-1, True)[0]
                if move.uci() in ['e8g8', 'e8c8']:
                    evaluation -= 11
                    # print('castle')
                if value == evaluation:
                    list_best_moves.append(move)
                elif value > evaluation:
                    value = evaluation
                    list_best_moves = [move]
            return value, random.choice(list_best_moves)

    def nn_select_best_moves(self, board):
        """Select best moves in board."""
        hash = chess.polyglot.zobrist_hash(board)
        if hash not in self.nn_tb:
            good_moves = list()
            for move in board.legal_moves:
                if neural_network.check_move(board.fen(), move.uci()):
                    good_moves.append(move)
            self.nn_tb[hash] = good_moves
        good_moves = self.nn_tb[hash]
        if not good_moves:
            good_moves = list(board.legal_moves)
        return good_moves

    def minimax_nn(self, board, depth, maximimize_white, limit_time):  # + param time + param best move depth-1 + param evaluation
        """Minimax algorithm from Wikipedia with NN."""
        if depth == 0 or board.is_game_over():
            zobrist_hash = chess.polyglot.zobrist_hash(board)
            if zobrist_hash not in self.tb:
                # if j'ai du temps
                self.tb[zobrist_hash] = self.evaluate(board)
                if len(self.tb) > self.tb_limit:
                    del self.tb[list(self.tb.keys())[0]]
                # else
                #
            evaluation = self.tb[zobrist_hash]
            # evaluation = self.evaluate(board)
            attackers = board.attackers(board.turn, board.peek().to_square)
            if len(attackers) > 0:
                # Quiescent
                if board.turn == chess.WHITE:
                    evaluation += PIECES_VALUES[board.piece_map()\
                                                [board.peek().to_square].\
                                                symbol().lower()]
                else:
                    evaluation -= PIECES_VALUES[board.piece_map()\
                                                [board.peek().to_square].\
                                                symbol().lower()]
            return evaluation, chess.Move.from_uci("0000")
        if maximimize_white:
            value = -float('inf')
            legal_moves = list(board.legal_moves)
            list_best_moves = [legal_moves[0]]
            for move in self.nn_select_best_moves(board):
                if time.time() > limit_time:
                    return float('inf'), chess.Move.from_uci("0000")
                test_board = chess.Board(fen=board.fen())
                test_board.push(move)
                evaluation = self.minimax_nn(test_board, depth-1, False, limit_time)[0]
                if move.uci() in ['e1g1', 'e1c1']:
                    evaluation += 11
                    # print('castle')
                if value == evaluation:
                    list_best_moves.append(move)
                elif value < evaluation:
                    value = evaluation
                    list_best_moves = [move]
            return value, random.choice(list_best_moves)
        else:
            # minimizing white
            value = float('inf')
            legal_moves = list(board.legal_moves)
            list_best_moves = [legal_moves[0]]
            for move in self.nn_select_best_moves(board):
                if time.time() > limit_time:
                    return float('inf'), chess.Move.from_uci("0000")
                test_board = chess.Board(fen=board.fen())
                test_board.push(move)
                evaluation = self.minimax_nn(test_board, depth-1, True, limit_time)[0]
                if move.uci() in ['e8g8', 'e8c8']:
                    evaluation -= 11
                    # print('castle')
                if value == evaluation:
                    list_best_moves.append(move)
                elif value > evaluation:
                    value = evaluation
                    list_best_moves = [move]
            return value, random.choice(list_best_moves)
