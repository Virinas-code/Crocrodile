#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Evaluation.

Evaluation: Evaluate position.
"""
import chess

BISHOPS_PAIR = 50
PROTECTED_KING = 70  # TODO: Review
CENTER_BONUS = 20
PAWN_SEVENTH_ROW = 50
CENTRAL_SQUARES = [36, 35, 28, 27]
ELARGED_SQUARES = [45, 44, 43, 42, 37, 34, 29, 26, 21, 20, 19, 18]
SEVENTH_ROW = [55, 54, 53, 52, 51, 50, 49, 48]
EIGHT_ROW = [56, 57, 58, 59, 60, 61, 62, 63]
SECOND_ROW = [15, 14, 13, 12, 11, 10, 9, 8]
FIRST_ROW = [0, 1, 2, 3, 4, 5, 6, 7]
THIRD_ROW = [16, 17, 18, 19, 20, 21, 22, 23]
COLUMN_A = [0, 8, 16, 24, 32, 40, 48, 56]
COLUMN_B = [1, 9, 17, 25, 33, 41, 49, 57]
COLUMN_C = [2, 10, 18, 26, 34, 42, 50, 58]
COLUMN_D = [3, 11, 19, 27, 35, 43, 51, 59]
COLUMN_E = [4, 12, 20, 28, 36, 44, 52, 60]
COLUMN_F = [5, 13, 21, 29, 37, 45, 53, 61]
COLUMN_G = [6, 14, 22, 30, 38, 46, 54, 62]
COLUMN_H = [7, 15, 23, 31, 39, 47, 55, 63]
COLUMNS = [COLUMN_A, COLUMN_B, COLUMN_C, COLUMN_D,
           COLUMN_E, COLUMN_F, COLUMN_G, COLUMN_H]
DOUBLED_PAWNS = 15
TRIPLED_PAWNS = 35
QUADRUPLED_PAWNS = 60
ISOLATED_PAWN = 17
PASSED_PAWN = 22


def pawn_on_column(column, pawn, piece_map):
    pawns_count = 0
    for square in column:
        if piece_map.get(square, None) == pawn:
            pawns_count += 1
    return pawns_count


def check_passed_pawns(board: chess.Board, color: bool) -> int:
    """Evaluation: Check paseed pawns.

    :param chess.Board board: Board.
    :param bool color: Color to add bonus (True is white).
    :return: Bonus points.
    :rtype: int

    """
    result = 0
    piece_map = board.piece_map()
    pawn = ("P" if color else "p")

    def pawn_on_column_after_rank(column, rank, pawn, piece_map):
        pawns_count = 0
        for square in column:
            if square > rank * 7 + 1:
                if square in piece_map:
                    if piece_map[square].symbol() == pawn:
                        pawns_count += 1
        return pawns_count
    for square in piece_map:
        if piece_map[square].symbol() == pawn:
            # Get rank and column of square
            rank = chess.square_rank(square)
            column = chess.square_file(square)
            if pawn_on_column_after_rank(COLUMNS[column], rank, pawn.swapcase(), piece_map) == 0 \
                    and pawn_on_column_after_rank(COLUMNS[min(7, column+1)], rank, pawn.swapcase(), piece_map) == 0 \
                    and pawn_on_column_after_rank(COLUMNS[max(0, column-1)], rank, pawn.swapcase(), piece_map) == 0:
                result += PASSED_PAWN
    return result


def old_evaluate(board: chess.Board):
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
    # Counters for bishop pair
    white_bishops = 0
    black_bishops = 0
    for piece in piece_map:
        if piece_map[piece].symbol().isupper():
            white_score += PIECES_VALUES[piece_map[piece].symbol().lower()]
            if piece in CENTRAL_SQUARES:
                white_score += CENTER_BONUS
            if piece in ELARGED_SQUARES:
                white_score += int(CENTER_BONUS / 2)
            if piece_map[piece].symbol() == 'P' and piece in SEVENTH_ROW:
                white_score += PAWN_SEVENTH_ROW
            if piece_map[piece].symbol() == 'P' and piece in EIGHT_ROW:
                white_score += QUEEN_VALUE
            if piece_map[piece].symbol() == 'B':
                white_bishops += 1
            """if piece_map[piece].symbol() == 'Q' and piece in CENTRAL_SQUARES and len(piece_map) > 20:
                white_score -= 30"""
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
    # Check isolated pawns
    for index, column in enumerate(COLUMNS):
        column_pawns = pawn_on_column(column, "P", piece_map)
        if column_pawns == 2:
            white_score -= DOUBLED_PAWNS
        if column_pawns == 3:
            white_score -= TRIPLED_PAWNS
        if column_pawns >= 4:
            white_score -= QUADRUPLED_PAWNS
        if column_pawns > 0:
            if column == COLUMN_A and pawn_on_column(COLUMN_B, "P", piece_map) == 0:
                white_score -= ISOLATED_PAWN
            elif column == COLUMN_H and pawn_on_column(COLUMN_G, "P", piece_map) == 0:
                white_score -= ISOLATED_PAWN
            else:
                if pawn_on_column(COLUMNS[index-1], "P", piece_map) == 0 and pawn_on_column(COLUMNS[index+1], "P", piece_map) == 0:
                    white_score -= ISOLATED_PAWN
    white_score += check_passed_pawns(board, True)
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


def evaluate2(board: chess.Board) -> int:
    """Evalutaion function number 2.

    :param chess.Board board: Board to evaluate.
    :return: Evaluation in centipawns.
    :rtype: int
    """
    piece_map = board.piece_map()
    score = 0
    # Constants

    # Material advantage
    for piece in piece_map.values():
        if piece.color == chess.WHITE:
            score += pieces_values[piece.piece_type]
    # King security
    king_position = list(piece_map.keys())[list(piece_map.values()).index(chess.Piece(chess.KING,
                                                                                      chess.WHITE))]
    king_squares = [
        king_position - 1,
        king_position + 1,
        king_position - 7,
        king_position - 8,
        king_position - 9,
        king_position + 7,
        king_position + 8,
        king_position + 9,
    ]
    king_protection = - (2 * king_protected)
    for square in king_squares:
        if square in piece_map and piece_map[square].piece_type in bad_edges_pieces \
                and piece_map[square].color == chess.WHITE:
            king_protection += king_protected
    # Pieces developement and activity
    for square in knight_base:
        if square in piece_map and piece_map[square].piece_type == chess.KNIGHT \
                and piece_map[square].color == chess.WHITE:
            score -= knight_at_home
    for square in bishop_base:
        if square in piece_map and piece_map[square].piece_type == chess.BISHOP \
                and piece_map[square].color == chess.WHITE:
            score -= bishop_at_home
    for square in bad_edges:
        if square in piece_map and piece_map[square].piece_type in bad_edges_pieces \
                and piece_map[square].color == chess.WHITE:
            score -= bad_edges_malus
    # Return
    return score


PAWN_VALUE = 100
KNIGHT_VALUE = 290
BISHOP_VALUE = 310
ROOK_VALUE = 500
QUEEN_VALUE = 900
PIECES_VALUES = {chess.PAWN: PAWN_VALUE, chess.KNIGHT: KNIGHT_VALUE,
                 chess.BISHOP: BISHOP_VALUE, chess.ROOK: ROOK_VALUE,
                 chess.QUEEN: QUEEN_VALUE, chess.KING: 0}
KING_PROTECTED = 5
KNIGHT_BASE = [1, 6]
KNIGHT_AT_HOME = 15
BISHOP_BASE = [2, 5]
BISHOP_AT_HOME = 10
BAD_EDGES = [8, 16, 24, 32, 40, 15, 23, 31, 39, 47]
BAD_EDGES_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
BAD_EDGES_MALUS = 20
CENTER_BONUS: int = 30


class Evaluator():
    """Base class for evaluation (v2)."""

    def evaluate(self, position: chess.Board, move: chess.Move = chess.Move.null()) -> int:
        """Evaluate position and return white's advantage in centipans.

        :param chess.Board position: Position to evaluate.
        :param chess.Move move: Move who will be played.
        :return: White's advantage as centipawns.
        :rtype: int
        """
        score = 0
        if position.is_checkmate():
            if position.turn == chess.WHITE:
                return -10000
            return 10000
        white_score = self.evaluate_position(position)
        white_score += self.evaluate_move(position, move)
        black_score = self.evaluate_position(position.mirror())
        score = white_score - black_score
        return score

    def evaluate_position(self, position: chess.Board) -> int:
        """Wrapper for evaluating position.

        :param chess.Board position: Position to evaluate.
        :return: White's advantage as centipawns.
        :rtype: int
        """
        score = 0
        score += self.eval_material(position)
        score += self.eval_center(position)
        score += self.eval_king_protection(position)
        score += self.eval_developpement(position)
        return score

    @staticmethod
    def eval_material(position: chess.Board) -> int:
        """Evaluate material advantage of White in the given position.

        :param chess.Board position: Position to evaluate.
        :return: White's material advantage as centipawns.
        :rtype: int
        """
        score = 0
        for piece in position.piece_map().values():
            if piece.color == chess.WHITE:
                score += PIECES_VALUES[piece.piece_type]
        return score

    @staticmethod
    def eval_king_protection(position: chess.Board) -> int:
        """Evaluate king protection score of White in the given position.

        :param chess.Board position: Position to evaluate.
        :return: White's material advantage as centipawns.
        :rtype: int
        """
        piece_map = position.piece_map()
        king_position = list(piece_map.keys())[list(piece_map.values()).index(
            chess.Piece(chess.KING, chess.WHITE))]
        king_squares = [
            king_position - 1,
            king_position + 1,
            king_position - 7,
            king_position - 8,
            king_position - 9,
            king_position + 7,
            king_position + 8,
            king_position + 9,
        ]
        score = - (2 * KING_PROTECTED)
        for square in king_squares:
            if square in piece_map and piece_map[square].color == chess.WHITE:
                score += KING_PROTECTED
        return score

    @staticmethod
    def eval_developpement(position: chess.Board):
        """Evaluate pieces developpement and activity in the given position.

        :param chess.Board position: Position to evaluate.
        :return: White's material advantage as centipawns.
        :rtype: int
        """
        piece_map = position.piece_map()
        score = 0
        for square in KNIGHT_BASE:
            if square in piece_map and piece_map[square].piece_type == chess.KNIGHT \
                    and piece_map[square].color == chess.WHITE:
                score -= KNIGHT_AT_HOME
        for square in BISHOP_BASE:
            if square in piece_map and piece_map[square].piece_type == chess.BISHOP \
                    and piece_map[square].color == chess.WHITE:
                score -= BISHOP_AT_HOME
        for square in BAD_EDGES:
            if square in piece_map and piece_map[square].piece_type in BAD_EDGES_PIECES \
                    and piece_map[square].color == chess.WHITE:
                score -= BAD_EDGES_MALUS
        return score

    def evaluate_move(self, position: chess.Board, move: chess.Move) -> int:
        """Wrapper for evaluating move.

        :param chess.Board position: Position to evaluate.
        :param chess.Move move: Move to evaluate.
        :return: White's advantage as centipawns.
        :rtype: int
        """
        score = 0
        score += self.eval_castling(position, move)
        return score

    @staticmethod
    def eval_castling(position: chess.Board, move: chess.Move) -> int:
        """Evaluate castling in the given position.

        :param chess.Board position: Position to evaluate.
        :param chess.Move move: Move to evaluate.
        :return: White's castling advantage as centipawns.
        :rtype: int
        """
        score = 0
        if move.from_square in position.piece_map() \
                and position.piece_map()[move.from_square].piece_type == chess.KING \
                and move.to_square in (chess.G1, chess.C1):
            score += 70
        return score

    @staticmethod
    def eval_center(position: chess.Board) -> int:
        """
        Evaluate center occupation.

        :param chess.Board position: Position to evaluate.
        :return: White's center advantage as centipawns.
        :rtype: int
        """
        score: int = 0
        for square, piece in position.piece_map().items():
            if piece.symbol().isupper():
                if square in CENTRAL_SQUARES:
                    score += CENTER_BONUS
                elif square in ELARGED_SQUARES:
                    score += int(CENTER_BONUS / 2)
        return score
