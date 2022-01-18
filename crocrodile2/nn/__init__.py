#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Neural Network.

Base class for Crocrodile NN.
"""
from __future__ import annotations
import csv
import heapq
import json
import multiprocessing
import os  # Path problems
import random
import sys
from typing import Optional, Tuple

import chess
import chess.polyglot
import numpy

# ====== IDLE ======
# import os
# os.chdir("../")
# ==== END IDLE ====

NoneType = type(None)
LAYERS = 5


class NeuralNetwork:
    """Base class for NN."""

    def __init__(self):
        """Initialize NN."""
        self.weight1 = self.csv_to_array("nns/default/w1.csv")
        self.weight2 = self.csv_to_array("nns/default/w2.csv")
        self.weight3 = self.csv_to_array("nns/default/w3.csv")
        self.weight4 = self.csv_to_array("nns/default/w4.csv")
        self.weight5 = self.csv_to_array("nns/default/w5.csv")
        self.b1 = self.csv_to_array("nns/default/b1.csv")
        self.b2 = self.csv_to_array("nns/default/b2.csv")
        self.b3 = self.csv_to_array("nns/default/b3.csv")
        self.b4 = self.csv_to_array("nns/default/b4.csv")
        self.b5 = self.csv_to_array("nns/default/b5.csv")
        self.cweight1 = self.csv_to_array("nns/default/cw1.csv")
        self.cweight2 = self.csv_to_array("nns/default/cw2.csv")
        self.cweight3 = self.csv_to_array("nns/default/cw3.csv")
        self.cweight4 = self.csv_to_array("nns/default/cw4.csv")
        self.cweight5 = self.csv_to_array("nns/default/cw5.csv")
        self.cb1 = self.csv_to_array("nns/default/cb1.csv")
        self.cb2 = self.csv_to_array("nns/default/cb2.csv")
        self.cb3 = self.csv_to_array("nns/default/cb3.csv")
        self.cb4 = self.csv_to_array("nns/default/cb4.csv")
        self.cb5 = self.csv_to_array("nns/default/cb5.csv")
        self.pre_input_layer = numpy.zeros(768)
        self.input_layer = numpy.zeros(64)
        self.hidden_layer_1 = numpy.zeros(64)
        self.hidden_layer_2 = numpy.zeros(64)
        self.hidden_layer_3 = numpy.zeros(64)
        self.hidden_layer_4 = numpy.zeros(1)
        self.output_layer = numpy.zeros(1)
        self.genetic_train_settings = json.load(open("nns/settings.json"))
        self.train_good = (
            open(self.genetic_train_settings["train_good"]).read().split(
                "\n\n")
        )
        self.train_bad = (
            open(self.genetic_train_settings["train_bad"]).read().split("\n\n")
        )
        self.test_good = (
            open(self.genetic_train_settings["test_good"]).read().split("\n\n")
        )
        self.test_bad = (
            open(self.genetic_train_settings["test_bad"]).read().split("\n\n")
        )
        self.result = None  # Basics training
        self.perfs: tuple[int, int] = (0, 0)
        self.old_good_moves_result: Optional[int] = None
        self.old_bad_moves_result: Optional[int] = None
        self.indice: int = -1
        self.layers: list[numpy.ndarray] = list()  # Multiple layers
        self.bias: list[numpy.ndarray] = list()  # Multiple layers
        self.last_layer: Optional[numpy.ndarray] = None
        self.last_bias: Optional[numpy.ndarray] = None
        # Multiprocesses global results
        self.multiprocesses_result: list[int] = [0, 0]

        # Pawns and pieces
        self.w_pawns: list[numpy.ndarray] = list()
        self.w_pieces: list[numpy.ndarray] = list()
        self.b_pawns: list[numpy.ndarray] = list()
        self.b_pieces: list[numpy.ndarray] = list()
        self.w_last: numpy.ndarray = list()
        self.b_last: numpy.ndarray = list()
        self.transpositions: dict[str, dict[str, bool]] = {}

    def load_networks(self) -> None:
        """
        Load networks from folder nns/

        :return: None
        :rtype: None
        """
        print("Loading networks... (counting networks)", end="\r", flush=True)
        self.tests_weight1 = list()
        self.tests_weight2 = list()
        self.tests_weight3 = list()
        self.tests_weight4 = list()
        self.tests_weight5 = list()
        self.tests_bias1 = list()
        self.tests_bias2 = list()
        self.tests_bias3 = list()
        self.tests_bias4 = list()
        self.tests_bias5 = list()
        population = self.genetic_train_settings["population"]
        for loop in range(population):
            print(
                f"Loading networks... ({loop}/{population})       ",
                end="\r",
                flush=True,
            )
            self.tests_weight1.append(self.csv_to_array(f"nns/{loop}-w1.csv"))
            self.tests_weight2.append(self.csv_to_array(f"nns/{loop}-w2.csv"))
            self.tests_weight3.append(self.csv_to_array(f"nns/{loop}-w3.csv"))
            self.tests_weight4.append(self.csv_to_array(f"nns/{loop}-w4.csv"))
            self.tests_weight5.append(self.csv_to_array(f"nns/{loop}-w5.csv"))
            self.tests_bias1.append(self.csv_to_array(f"nns/{loop}-b1.csv"))
            self.tests_bias2.append(self.csv_to_array(f"nns/{loop}-b2.csv"))
            self.tests_bias3.append(self.csv_to_array(f"nns/{loop}-b3.csv"))
            self.tests_bias4.append(self.csv_to_array(f"nns/{loop}-b4.csv"))
            self.tests_bias5.append(self.csv_to_array(f"nns/{loop}-b5.csv"))
        print("Loading networks... Done.               ")

    def output(self):
        """Return NN output."""
        try:
            if self.output_layer < 0:
                return True
            return False
        except ValueError:
            if self.output_layer[0] < 0:
                return True
            return False

    def generate_inputs(self, board: str, move: str) -> bool:
        """
        Generate inputs for move move in board.

        :param str board: FEN of the board.
        :param str move: UCI notation of the move to check.
        :return: False if this is a check, True else.
        :rtype: bool
        """
        def board_to_matrix(board: chess.Board) -> numpy.ndarray:
            """Convert chess board to matrix."""
            # Base values
            inputs_values: dict[str, float] = {
                "": 0.0,
                "P": 0.1,
                "N": 0.2,
                "B": 0.3,
                "R": 0.5,
                "Q": 0.9,
                "K": 1,
                "p": -0.1,
                "n": -0.2,
                "b": -0.3,
                "r": -0.5,
                "q": -0.9,
                "k": -1,
            }

            # Castling rights
            castling_bonus: dict[str, float] = {
                "k": 0.05,
                "q": 0.10,
            }
            if board.has_kingside_castling_rights(chess.WHITE):
                inputs_values["K"] += castling_bonus["k"]
            if board.has_queenside_castling_rights(chess.WHITE):
                inputs_values["K"] += castling_bonus["q"]
            if board.has_kingside_castling_rights(chess.BLACK):
                inputs_values["k"] -= castling_bonus["k"]
            if board.has_queenside_castling_rights(chess.BLACK):
                inputs_values["k"] -= castling_bonus["q"]

            # Generate board
            result: numpy.ndarray = numpy.zeros((8, 8))
            piece_map = board.piece_map()
            for rank in range(8):
                for file in range(8):
                    square: int = rank * 8 + file
                    piece: chess.Piece | bool = piece_map.get(square, False)
                    if piece:
                        result[rank][file] = inputs_values[piece.symbol()]
                    else:
                        result[rank][file] = inputs_values[""]

            # En passant
            if board.has_legal_en_passant():
                ep_square: chess.Square = board.ep_square
                result[chess.square_rank(
                    ep_square)][chess.square_file(ep_square)] = 0.01

            # Return
            return result

        # Mirror board
        def flip_rank(rank: str) -> str:
            ranks = list("12345678")
            reversed_ranks = list(reversed(ranks))
            return reversed_ranks[ranks.index(rank)]

        def flip_move_vertical(uci: str) -> str:
            """
            Flip move vertically.

            :param str uci: UCI representation of move
            :return: UCI representation of flipped move
            :rtype: str
            """
            from_square = list(uci[0:2])
            to_square = list(uci[2:4])
            if len(uci) == 5:  # Promotion
                promotion = uci[4]
            else:
                promotion = ""
            from_square[1] = flip_rank(from_square[1])
            to_square[1] = flip_rank(to_square[1])
            return "".join(from_square) + "".join(to_square) + promotion

        board: chess.Board = chess.Board(board)
        if board.turn == chess.BLACK:
            board: chess.Board = board.mirror()
            move = flip_move_vertical(move)

        pieces_values = {
            "k": 1000,
            "q": 9,
            "r": 5,
            "b": 3,
            "n": 3,
            "p": 1,
        }
        piece_map = board.piece_map()
        try:
            piece_from = piece_map[chess.Move.from_uci(
                move).from_square].symbol().lower()
            try:
                piece_to = piece_map[chess.Move.from_uci(
                    move).to_square].symbol().lower()
            except KeyError:
                pass
            else:
                if pieces_values[piece_to] > pieces_values[piece_from]:
                    return False
        except KeyError:
            pass
        # First board
        board1: numpy.ndarray = board_to_matrix(board)

        # Second board
        board.push(chess.Move.from_uci(move))
        if board.is_check():  # Check is always a good move
            return False
        board2: numpy.ndarray = board_to_matrix(board)

        # Result
        zeros = numpy.zeros((8, 8))
        final = numpy.block([[board1, zeros], [zeros, board2]])
        self.input_layer = final
        return True

        # Old code
        #
        # pieces: dict[int, chess.Piece] = board.piece_map()
        # inputs: list[list[float]] = []
        # inputs_values: dict[str, float] = {
        #     "": 0.0,
        #     "P": 0.1,
        #     "N": 0.2,
        #     "B": 0.3,
        #     "R": 0.5,
        #     "Q": 0.9,
        #     "K": 1,
        #     "p": -0.1,
        #     "n": -0.2,
        #     "b": -0.3,
        #     "r": -0.5,
        #     "q": -0.9,
        #     "k": -1,
        # }
        # for rank in chess.RANK_NAMES:
        #     inputs.append([])
        #     for file in chess.FILE_NAMES:
        #         square: int = chess.parse_square(file + rank)
        #         if pieces.get(square, False):
        #             inputs[chess.RANK_NAMES.index(rank)].append(
        #                 inputs_values[pieces[square].symbol()]
        #             )
        #         else:
        #             inputs[chess.RANK_NAMES.index(rank)].append(
        #                 inputs_values[""])
        # if board.has_legal_en_passant():
        #     inputs[chess.square_rank(board.ep_square)][
        #         chess.square_file(board.ep_square)
        #     ] = -0.05
        # move: chess.Move = chess.Move.from_uci(move)
        # from_square: chess.Square = move.from_square
        # to_square: chess.Square = move.to_square
        # move_inputs: list[float] = [float()] * 4
        # move_inputs[0]: float = chess.square_file(from_square) / 10
        # move_inputs[1]: float = chess.square_rank(from_square) / 10
        # move_inputs[2]: float = chess.square_file(to_square) / 10
        # move_inputs[3]: float = chess.square_rank(to_square) / 10
        # castling_inputs: list[float] = []
        # if board.has_kingside_castling_rights(chess.WHITE):
        #     castling_inputs.append(1)
        # else:
        #     castling_inputs.append(0)
        # if board.has_queenside_castling_rights(chess.WHITE):
        #     castling_inputs.append(1)
        # else:
        #     castling_inputs.append(0)
        # if board.has_kingside_castling_rights(chess.BLACK):
        #     castling_inputs.append(1)
        # else:
        #     castling_inputs.append(0)
        # if board.has_queenside_castling_rights(chess.BLACK):
        #     castling_inputs.append(1)
        # else:
        #     castling_inputs.append(0)
        # en_passant_input: float = 0.0
        # if board.has_legal_en_passant():
        #     en_passant_input: float = (
        #         chess.square_file(board.ep_square) + 1) / 10
        # inputs.append(move_inputs + castling_inputs + [en_passant_input])
        # for indice in range(4):
        #     inputs[indice].append(move_inputs[indice])
        #     inputs[indice + 4].append(castling_inputs[indice])
        # self.input_layer: numpy.array = numpy.array(inputs)

    @staticmethod
    def csv_to_array(csv_path):
        """Read CSV file and return array."""
        result = list()
        with open(csv_path) as file:
            reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                result.append(row)
        return numpy.array(result)

    def full_calculate(self):
        """Calculate NN result with all hidden layers."""
        normalizer = self.normalisation
        self.hidden_layer_1 = self.weight1 @ self.input_layer + self.b1
        self.hidden_layer_1 = normalizer(self.hidden_layer_1)
        self.hidden_layer_2 = self.hidden_layer_1 @ self.weight2 + self.b2
        self.hidden_layer_2 = normalizer(self.hidden_layer_2)
        self.hidden_layer_3 = self.hidden_layer_2 @ self.weight3 + self.b3
        self.hidden_layer_3 = normalizer(self.hidden_layer_3)
        self.hidden_layer_4 = self.weight4 @ self.hidden_layer_3 + self.b4
        self.hidden_layer_4 = normalizer(self.hidden_layer_4)
        self.output_layer = self.hidden_layer_4 @ self.weight5 + self.b5
        self.output_layer = normalizer(self.output_layer)
        print(
            f"hl1 : {self.hidden_layer_1.shape} / hl2 : {self.hidden_layer_2.shape} / hl3 : {self.hidden_layer_3.shape} / hl4 : {self.hidden_layer_4.shape} / output : {self.output_layer.shape}"
        )

    def old_calculate(self):
        """Calculate NN result."""
        normalizer = self.normalisation
        self.output_layer = normalizer(
            normalizer(
                self.weight4
                @ normalizer(
                    normalizer(
                        normalizer(self.weight1 @ self.input_layer + self.b1)
                        @ self.weight2
                        + self.b2
                    )
                    @ self.weight3
                    + self.b3
                )
                + self.b4
            )
            @ self.weight5
            + self.b5
        )
        # self.output_layer = ((self.weight4 @ normalizer(((self.weight1 @ self.input_layer + self.b1) @ self.weight2 + self.b2) @ self.weight3 + self.b3) + self.b4) @ self.weight5 + self.b5)

    def calculate(self) -> numpy.ndarray:
        """
        Calculate NN result by using :attr:NeuralNetwork.layers with multiple layers.

        :return: Output layer.
        :rtype: numpy.ndarray
        """
        hidden_layer: numpy.ndarray = self.input_layer
        mask_false = 16 * [16 * [False]]
        for layer_index in range(LAYERS):
            mask_pawns = numpy.ma.masked_outside(
                abs(hidden_layer), 0.08, 0.1699).mask
            mask_pieces = numpy.ma.masked_outside(
                abs(hidden_layer), 0.17, 1.16).mask
            hidden_layer = (numpy.ma.array(hidden_layer, mask=mask_pawns) @ self.w_pawns[layer_index]
                            + numpy.ma.array(self.b_pawns[layer_index], mask=mask_pawns)) + \
                           (numpy.ma.array(hidden_layer, mask=mask_pieces) @ self.w_pieces[layer_index]
                            + numpy.ma.array(self.b_pieces[layer_index], mask=mask_pieces))
            hidden_layer.mask = mask_false
        column_mask = 16 * [[False]]
        line_mask = [16 * [False]]
        hidden_layer = (hidden_layer @ numpy.ma.array(self.w_pawns[-1], mask=column_mask) + self.b_pawns[-1]) + \
                       (hidden_layer @
                        numpy.ma.array(self.w_pieces[-1], mask=column_mask) + self.b_pieces[-1])
        self.output_layer = numpy.ma.array(
            self.w_last, mask=line_mask) @ hidden_layer + self.b_last
        return self.output_layer

    def check_move_with_transpos(self, board: str, move: str) -> bool:
        """Check a move with transposition.

        Generate inputs, calculate and return output.

        :param board: FEN of the board to check.
        :type board: str
        :param move: Move who will be played.
        :type move: str
        :return: Wether if move is interesting or not.
        :rtype: bool
        """
        zobrist: str = chess.polyglot.zobrist_hash(chess.Board(board))
        transpos_result: bool | None = self.transpositions.get(
            zobrist, None).get(move, None)
        if transpos_result is None:
            if self.generate_inputs(board, move):
                self.calculate()
                output = self.output()
                try:
                    self.transpositions[zobrist][move] = output
                except KeyError:
                    self.transpositions[zobrist] = {move: output}
                return output
            else:
                try:
                    self.transpositions[zobrist][move] = True
                except KeyError:
                    self.transpositions[zobrist] = {move: True}
                return True
        else:
            # if len(transpos_result)
            return transpos_result

    def check_move(self, board, move):
        """Generate inputs, calculate and return output."""
        if self.generate_inputs(board, move):
            self.calculate()
            return self.output()
        else:
            return True

    def check_test(self):
        """Check NN on test dataset."""
        file_goodmoves = self.test_good
        file_badmoves = self.test_bad
        length = len(file_goodmoves) + len(file_badmoves)
        errs = 0
        good = 0
        for inputs in file_goodmoves:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            if self.check_move(pos, mve):
                good += 1
            else:
                errs += 1
        for inputs in file_badmoves:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            if not self.check_move(pos, mve):
                good += 1
            else:
                errs += 1
        return good / length * 100

    def check_train(self):
        """Check NN on train dataset."""
        file_goodmoves = self.train_good
        file_badmoves = self.train_bad
        errs = 0
        good = 0
        correct_on_good_moves = 0
        correct_on_bad_moves = 0
        for inputs in file_goodmoves:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            if self.check_move(pos, mve):
                good += 1
                correct_on_good_moves += 1
            else:
                errs += 1
        for inputs in file_badmoves:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            if not self.check_move(pos, mve):
                good += 1
                correct_on_bad_moves += 1
            else:
                errs += 1
        return (
            correct_on_good_moves,
            correct_on_bad_moves,
            len(file_goodmoves),
            len(file_badmoves),
        )

    def test_full_multiprocesses(self, list_good_moves: list,
                                 list_bad_moves: list) -> Tuple[int, int]:
        """
        Test neural network on full files with multi-processing.
        Lists list_good_moves and list_bad_moves can be obtained from training
        files with open("<path>").read().split("\\n\\n").
        Standard format is ["<FEN>\\n<move>", "<FEN>\\n<move>"]

        :param list_good_moves: List of good moves at standard format
        :type list_good_moves: list
        :param list_bad_moves: List of bad moves at standard format
        :type list_bad_moves: list
        :return: Number of correct answers on good moves, on bad moves
        :rtype: Tuple[int, int]
        """
        ## print(f"DEBUG: {list_good_moves=} {list_bad_moves=}")
        # Make 2 sub-lists
        list_good_moves1 = list_good_moves[:len(list_good_moves)//2]
        list_good_moves2 = list_good_moves[len(list_good_moves)//2:]
        list_bad_moves1 = list_bad_moves[:len(list_bad_moves)//2]
        list_bad_moves2 = list_bad_moves[len(list_bad_moves)//2:]
        ## print(f"DEBUG: {list_good_moves1=} {list_good_moves2=}")
        ## print(f"DEBUG: {list_bad_moves1=} {list_bad_moves2=}")
        # Create shared variables
        good_moves_data = multiprocessing.Value('i', 0)
        bad_moves_data = multiprocessing.Value('i', 0)
        # Create processes
        process1 = multiprocessing.Process(target=self.test_full, args=(list_good_moves1,
                                                                        list_bad_moves1, True,
                                                                        good_moves_data,
                                                                        bad_moves_data))
        process2 = multiprocessing.Process(target=self.test_full, args=(list_good_moves2,
                                                                        list_bad_moves2, True,
                                                                        good_moves_data,
                                                                        bad_moves_data))
        # Start them
        self.multiprocesses_result: list[int] = [0, 0]
        process1.start()
        process2.start()
        # Get results
        process1.join()
        process2.join()
        good_moves_result = good_moves_data.value
        bad_moves_result = bad_moves_data.value
        # Old results
        self.old_good_moves_result = good_moves_result
        self.old_bad_moves_result = bad_moves_result
        # Return
        return good_moves_result, bad_moves_result

    def test_full(self, list_good_moves: list, list_bad_moves: list, sub=False,
                  good_moves_data=0, bad_moves_data=0) -> Tuple[int, int]:
        """
        Test neural network on full files.
        Lists list_good_moves and list_bad_moves can be obtained from training
        files with open("<path>").read().split("\\n\\n").
        Standard format is ["<FEN>\\n<move>", "<FEN>\\n<move>"]

        :param list_good_moves: List of good moves at standard format
        :type list_good_moves: list
        :param list_bad_moves: List of bad moves at standard format
        :type list_bad_moves: list
        :return: Number of correct answers on good moves, on bad moves
        :rtype: Tuple[int, int]
        """
        if len(list_good_moves) > 2 and not sub:
            return self.test_full_multiprocesses(list_good_moves, list_bad_moves)
        good_moves_result = 0
        bad_moves_result = 0
        for position_and_move in list_good_moves:
            position = position_and_move.split("\n")[0]
            move = position_and_move.split("\n")[1]
            if self.check_move(position, move):
                good_moves_result += 1
        for position_and_move in list_bad_moves:
            position = position_and_move.split("\n")[0]
            move = position_and_move.split("\n")[1]
            if not self.check_move(position, move):
                bad_moves_result += 1
        ## print(f"DEBUG: crocrodile.nn.NeuralNetwork.test_full: {good_moves_result=} {bad_moves_result=}")
        try:
            good_moves_data.value = good_moves_data.value + good_moves_result
            bad_moves_data.value = bad_moves_data.value + bad_moves_result
        except:
            pass
        self.old_good_moves_result = good_moves_result
        self.old_bad_moves_result = bad_moves_result
        ## print(f"DEBUG: crocrodile.nn.NeuralNetwork.test_full: {self.multiprocesses_result=}")
        return good_moves_result, bad_moves_result

    def test_new(self, new_good_move: str, new_bad_moves: list) -> Tuple[int, int]:
        """
        Test neural network on new moves new_good_move and new_bad_moves.
        Standard format is ["<FEN>\\n<move>", "<FEN>\\n<move>"]
        Standard format for new_good_move is "<FEN>\\n<move>"

        :param list_good_moves: The new good move to test at standard format
        :type list_good_moves: list
        :param list_bad_moves: The new bad moves to test at standard format
        :type list_bad_moves: list
        :return: Total number of correct answers on good moves, on bad moves
        :rtype: Tuple[int, int]
        """
        good_moves_result = int(self.old_good_moves_result)
        bad_moves_result = int(self.old_bad_moves_result)
        position = new_good_move.split("\n")[0]
        move = new_good_move.split("\n")[1]
        if self.check_move(position, move):
            good_moves_result += 1
        for position_and_move in new_bad_moves:
            position = position_and_move.split("\n")[0]
            move = position_and_move.split("\n")[1]
            if not self.check_move(position, move):
                bad_moves_result += 1
        self.old_good_moves_result = good_moves_result
        self.old_bad_moves_result = bad_moves_result
        return good_moves_result, bad_moves_result

    def test(self, list_good_moves: list, list_bad_moves: list) -> tuple[int]:
        """
        Test neural network.
        Used by basics training.

        :param list_good_moves: List of good moves at format ["<FEN>\\n<Good move>", "<FEN>\\n<Good move>"]
        :type list_good_moves: list
        :param list_bad_moves: List of bad moves at format ["<FEN>\\n<Bad move>", "<FEN>\\n<Bad move>"]
        :type list_bad_moves: list
        :return: Tupple (Number of correct answers on good moves, Number of correct answers on bad moves)
        :rtype: tuple[int]
        """
        good_moves_result = 0
        bad_moves_result = 0
        for position_and_move in list_good_moves:
            position = position_and_move.split("\n")[0]
            move = position_and_move.split("\n")[1]
            if self.check_move(position, move):
                good_moves_result += 1
        for position_and_move in list_bad_moves:
            position = position_and_move.split("\n")[0]
            move = position_and_move.split("\n")[1]
            if self.check_move(position, move):
                bad_moves_result += 1
        return good_moves_result, bad_moves_result

    def train(self):
        """Train Neural Network."""
        self.change_files()
        max_iters = int(input("Maximum iterations : "))
        iters = 0
        success_objective = float(input("Success objective (in percents) : "))
        max_diff = float(
            input("Maximal difference between training and test" +
                  " success rates : ")
        )
        balance = float(
            input(
                "Balance between good moves and bad moves (>1 to enhance good moves success rate) : "
            )
        )
        on_good_moves, on_bad_moves, good_moves, bad_moves = self.check_train()
        old_on_good_moves, old_on_bad_moves, old_good_moves, old_bad_moves = (
            on_good_moves,
            on_bad_moves,
            good_moves,
            bad_moves,
        )
        success = (
            (balance * on_good_moves + on_bad_moves)
            / (balance * good_moves + bad_moves)
            * 100
        )
        diff = success - self.check_test()
        precedent_difference = abs(
            ((on_good_moves / good_moves) - (on_bad_moves / bad_moves)) * 100
        )
        mutation_rate = float(input("Mutation rate (in percents) : "))
        mutation_change = float(input("Mutation change : "))
        inverse_rate = 100 / mutation_rate
        print(
            f"Success : {success} / Diff : {diff} / Precedent difference : {precedent_difference}"
        )
        normalizer = numpy.vectorize(self.normalisation)
        while iters < max_iters and success_objective > success and diff < max_diff:
            iters += 1
            print("Training #" + str(iters))
            random_matrix1 = (
                numpy.random.rand(64, 64) *
                (2 * mutation_change) - mutation_change
            )
            random_matrix4 = (
                numpy.random.rand(1, 64) *
                (2 * mutation_change) - mutation_change
            )
            random_matrix5 = (
                numpy.random.rand(64, 1) *
                (2 * mutation_change) - mutation_change
            )
            random_matrixb5 = (
                numpy.random.rand(1, 1) *
                (2 * mutation_change) - mutation_change
            )
            rand1 = numpy.random.rand(
                64, 64) * inverse_rate + (1 - inverse_rate)
            rand2 = numpy.random.rand(
                64, 64) * inverse_rate + (1 - inverse_rate)
            rand3 = numpy.random.rand(
                64, 64) * inverse_rate + (1 - inverse_rate)
            rand4 = numpy.random.rand(
                1, 64) * inverse_rate + (1 - inverse_rate)
            rand5 = numpy.random.rand(
                64, 1) * inverse_rate + (1 - inverse_rate)
            randb1 = numpy.random.rand(
                64, 64) * inverse_rate + (1 - inverse_rate)
            randb2 = numpy.random.rand(
                64, 64) * inverse_rate + (1 - inverse_rate)
            randb3 = numpy.random.rand(
                64, 64) * inverse_rate + (1 - inverse_rate)
            randb4 = numpy.random.rand(
                1, 64) * inverse_rate + (1 - inverse_rate)
            randb5 = numpy.random.rand(
                1, 1) * inverse_rate + (1 - inverse_rate)
            new_weight1 = numpy.heaviside(rand1, 0) * self.cweight1
            new_weight2 = numpy.heaviside(rand2, 0) * self.cweight2
            new_weight3 = numpy.heaviside(rand3, 0) * self.cweight3
            new_weight4 = numpy.heaviside(rand4, 0) * self.cweight4
            new_weight5 = numpy.heaviside(rand5, 0) * self.cweight5
            new_b1 = numpy.heaviside(randb1, 0) * self.cb1
            new_b2 = numpy.heaviside(randb2, 0) * self.cb2
            new_b3 = numpy.heaviside(randb3, 0) * self.cb3
            new_b4 = numpy.heaviside(randb4, 0) * self.cb4
            new_b5 = numpy.heaviside(randb5, 0) * self.cb5
            self.weight1 = self.weight1 + random_matrix1 * new_weight1
            self.weight2 = self.weight2 + random_matrix1 * new_weight2
            self.weight3 = self.weight3 + random_matrix1 * new_weight3
            self.weight4 = self.weight4 + random_matrix4 * new_weight4
            self.weight5 = self.weight5 + random_matrix5 * new_weight5
            self.b1 = self.b1 + random_matrix1 * new_b1
            self.b2 = self.b2 + random_matrix1 * new_b2
            self.b3 = self.b3 + random_matrix1 * new_b3
            self.b4 = self.b4 + random_matrix4 * new_b4
            self.b5 = self.b5 + random_matrixb5 * new_b5
            on_good_moves, on_bad_moves, good_moves, bad_moves = self.check_train()
            next_success = (
                (balance * on_good_moves + on_bad_moves)
                / (balance * good_moves + bad_moves)
                * 100
            )
            print(
                "Test success rate :",
                next_success,
                "(on good moves :",
                (on_good_moves / good_moves) * 100,
                "% / on bad moves :",
                (on_bad_moves / bad_moves) * 100,
                "% )",
            )
            # difference = abs(((on_good_moves / good_moves) - (on_bad_moves / bad_moves)) * 100)
            # - 0.5 * (difference - precedent_difference): # or difference > precedent_difference:
            if next_success < success:
                print("Reseting")
                self.weight1 = self.weight1 - random_matrix1 * new_weight1
                self.weight2 = self.weight2 - random_matrix1 * new_weight2
                self.weight3 = self.weight3 - random_matrix1 * new_weight3
                self.weight4 = self.weight4 - random_matrix4 * new_weight4
                self.weight5 = self.weight5 - random_matrix5 * new_weight5
                self.b1 = self.b1 - random_matrix1 * new_b1
                self.b2 = self.b2 - random_matrix1 * new_b2
                self.b3 = self.b3 - random_matrix1 * new_b3
                self.b4 = self.b4 - random_matrix4 * new_b4
                self.b5 = self.b5 - random_matrixb5 * new_b5
                # Nouvelle matrice consolidation = Ancienne - 0.05 * heaviside(matrice aléatorie, 0) * ancienne
                self.cweight1 = (
                    self.cweight1 - 0.05 *
                    numpy.heaviside(rand1, 0) * self.cweight1
                )
                self.cweight2 = (
                    self.cweight2 - 0.05 *
                    numpy.heaviside(rand2, 0) * self.cweight2
                )
                self.cweight3 = (
                    self.cweight3 - 0.05 *
                    numpy.heaviside(rand3, 0) * self.cweight3
                )
                self.cweight4 = (
                    self.cweight4 - 0.05 *
                    numpy.heaviside(rand4, 0) * self.cweight4
                )
                self.cweight5 = (
                    self.cweight5 - 0.05 *
                    numpy.heaviside(rand5, 0) * self.cweight5
                )
                self.cb1 = self.cb1 - 0.05 * \
                    numpy.heaviside(randb1, 0) * self.cb1
                self.cb2 = self.cb2 - 0.05 * \
                    numpy.heaviside(randb2, 0) * self.cb2
                self.cb3 = self.cb3 - 0.05 * \
                    numpy.heaviside(randb3, 0) * self.cb3
                self.cb4 = self.cb4 - 0.05 * \
                    numpy.heaviside(randb4, 0) * self.cb4
                self.cb5 = self.cb5 - 0.05 * \
                    numpy.heaviside(randb5, 0) * self.cb5
                on_good_moves, on_bad_moves, good_moves, bad_moves = (
                    old_on_good_moves,
                    old_on_bad_moves,
                    old_good_moves,
                    old_bad_moves,
                )
            elif next_success == success:
                print("Equal")
            else:
                # Check test take some time, but it's essential not to overtrain
                diff = success - self.check_test()
                # Nouvelle matrice consolidation = Normalisation(Ancienne + 0.05 * heaviside(matrice aléatorie, 0) * ancienne)
                self.cweight1 = normalizer(
                    self.cweight1 + 0.05 *
                    numpy.heaviside(rand1, 0) * self.cweight1
                )
                self.cweight2 = normalizer(
                    self.cweight2 + 0.05 *
                    numpy.heaviside(rand2, 0) * self.cweight2
                )
                self.cweight3 = normalizer(
                    self.cweight3 + 0.05 *
                    numpy.heaviside(rand3, 0) * self.cweight3
                )
                self.cweight4 = normalizer(
                    self.cweight4 + 0.05 *
                    numpy.heaviside(rand4, 0) * self.cweight4
                )
                self.cweight5 = normalizer(
                    self.cweight5 + 0.05 *
                    numpy.heaviside(rand5, 0) * self.cweight5
                )
                self.cb1 = normalizer(
                    self.cb1 + 0.05 * numpy.heaviside(randb1, 0) * self.cb1
                )
                self.cb2 = normalizer(
                    self.cb2 + 0.05 * numpy.heaviside(randb2, 0) * self.cb2
                )
                self.cb3 = normalizer(
                    self.cb3 + 0.05 * numpy.heaviside(randb3, 0) * self.cb3
                )
                self.cb4 = normalizer(
                    self.cb4 + 0.05 * numpy.heaviside(randb4, 0) * self.cb4
                )
                self.cb5 = normalizer(
                    self.cb5 + 0.05 * numpy.heaviside(randb5, 0) * self.cb5
                )
            old_on_good_moves, old_on_bad_moves, old_good_moves, old_bad_moves = (
                on_good_moves,
                on_bad_moves,
                good_moves,
                bad_moves,
            )
            success = (
                (balance * on_good_moves + on_bad_moves)
                / (balance * good_moves + bad_moves)
                * 100
            )
            precedent_difference = abs(
                ((on_good_moves / good_moves) - (on_bad_moves / bad_moves)) * 100
            )
            print(
                "New success rate :",
                success,
                "(on good moves :",
                (on_good_moves / good_moves) * 100,
                "% / on bad moves :",
                (on_bad_moves / bad_moves) * 100,
                "% )",
            )
            if iters % 100 == 0:
                self.save()
                print("Saved")
        self.save()
        print("Saved.")

    def genetic_train(self):
        """
        Genetic training algorithm.

        New training algorithm using a real genetic algorithm.
        """
        self.load_networks()

        def sprint(value):
            centered = value.center(18)
            print("********** {0} **********".format(centered))

        tests_weight1 = list()
        tests_weight2 = list()
        tests_weight3 = list()
        tests_weight4 = list()
        tests_weight5 = list()
        tests_bias1 = list()
        tests_bias2 = list()
        tests_bias3 = list()
        tests_bias4 = list()
        tests_bias5 = list()
        self.genetic_configure()
        max_iters = self.genetic_train_settings["max_iters"]
        max_success = self.genetic_train_settings["max_success"]
        balance = self.genetic_train_settings["balance"]
        inverse_rate = 100 / self.genetic_train_settings["mutation_rate"]
        mutation_change = self.genetic_train_settings["mutation_change"]
        sprint("Initialize")
        iters = 0
        print("Calculating first success...", end=" ", flush=True)
        on_good_moves, on_bad_moves, good_moves, bad_moves = self.check_train()
        success = (
            (balance * on_good_moves + on_bad_moves)
            / (balance * good_moves + bad_moves)
            * 100
        )
        print("Done.")
        print("Loading networks... (counting networks)", end="\r", flush=True)
        population = self.genetic_train_settings["population"]
        for loop in range(population):
            print(
                f"Loading networks... ({loop}/{population})       ",
                end="\r",
                flush=True,
            )
            tests_weight1.append(self.csv_to_array(f"nns/{loop}-w1.csv"))
            tests_weight2.append(self.csv_to_array(f"nns/{loop}-w2.csv"))
            tests_weight3.append(self.csv_to_array(f"nns/{loop}-w3.csv"))
            tests_weight4.append(self.csv_to_array(f"nns/{loop}-w4.csv"))
            tests_weight5.append(self.csv_to_array(f"nns/{loop}-w5.csv"))
            tests_bias1.append(self.csv_to_array(f"nns/{loop}-b1.csv"))
            tests_bias2.append(self.csv_to_array(f"nns/{loop}-b2.csv"))
            tests_bias3.append(self.csv_to_array(f"nns/{loop}-b3.csv"))
            tests_bias4.append(self.csv_to_array(f"nns/{loop}-b4.csv"))
            tests_bias5.append(self.csv_to_array(f"nns/{loop}-b5.csv"))
        print("Loading networks... Done.               ")
        print("Loading tests results...", end=" ", flush=True)
        tests_results = self.csv_to_array("nns/results.csv")
        tests_results = list(tests_results)
        for indice, element in enumerate(tests_results):
            tests_results[indice] = element[0]
        print("Done.")
        # https://stackoverflow.com/questions/16225677/get-the-second-largest-number-in-a-list-in-linear-time
        while iters < max_iters and success < max_success:
            iters += 1
            sprint("Training #{0}".format(iters))
            print("Selecting best networks...", end=" ", flush=True)
            maxis_brut = heapq.nlargest(32, tests_results)
            maxis = list()
            for element in maxis_brut:
                maxis.append(element)
            maxis_indices = []
            for element in range(32):
                maxis_indices.append(tests_results.index(maxis[element]))
            minis_brut = heapq.nsmallest(32, tests_results)
            minis = list()
            for element in minis_brut:
                if type(element) == list:
                    minis.append(element[0])
                else:
                    minis.append(element)
            minis_indices = sorted(
                range(len(tests_results)), key=lambda sub: tests_results[sub]
            )[:32]
            print("Done.")
            liste = []
            for count in range(8):
                liste.append(
                    "#" + str(minis_indices[count]) +
                    " (" + str(minis[count]) + ")"
                )
            print(f"Worst networks : {', '.join(liste)}")
            liste = []
            for count in range(8):
                liste.append(
                    "#" + str(maxis_indices[count]) +
                    " (" + str(maxis[count]) + ")"
                )
            print(f"Best networks : {', '.join(liste)}")
            for network_indice in range(32):
                print(
                    f"Coupling network #{network_indice + 1}... (selecting second network)",
                    end="\r",
                    flush=True,
                )
                cont = True
                while cont:
                    cont = False
                    rand = random.randint(0, population - 1)
                    if rand in minis_indices or rand in maxis_indices:
                        cont = True
                second_network = rand
                print(
                    f"Coupling network #{network_indice + 1}... (generating coupling matrixes)",
                    end="\r",
                    flush=True,
                )
                choose_w1 = numpy.zeros((64, 64))
                choose_w2 = numpy.zeros((64, 64))
                choose_w3 = numpy.zeros((64, 64))
                choose_matrixes1 = [choose_w1, choose_w2, choose_w3]
                for choose_matrix in choose_matrixes1:
                    # True : column else line
                    direction = bool(random.getrandbits(1))
                    choose = bool(random.getrandbits(1))
                    for line in range(len(choose_matrix)):
                        for column in range(len(choose_matrix[0])):
                            if random.random() < 0.001:
                                choose = not choose
                            fill = int(choose)
                            if direction:
                                choose_matrix[line][column] = fill
                            else:
                                choose_matrix[column][line] = fill
                choose = bool(random.getrandbits(1))
                choose_w5 = numpy.zeros((64, 1))
                for line in range(64):
                    if random.random() < 0.0625:
                        choose = not choose
                    choose_w5[line][0] = int(choose)
                choose = bool(random.getrandbits(1))
                choose_w4 = numpy.zeros((1, 64))
                for column in range(64):
                    if random.random() < 0.0625:
                        choose = not choose
                    choose_w4[0][column] = int(choose)
                choose_b1 = choose_w1
                choose_b2 = choose_w2
                choose_b3 = choose_w3
                choose_b4 = numpy.zeros((1, 64))
                choose = bool(random.getrandbits(1))
                for column in range(64):
                    if random.random() < 0.0625:
                        choose = not choose
                    choose_b4[0][column] = int(choose)
                choose_b5 = numpy.array([[int(bool(random.getrandbits(1)))]])
                print(
                    f"Coupling network #{network_indice + 1}... (coupling)                    ",
                    end="\r",
                    flush=True,
                )
                tests_weight1[minis_indices[network_indice]] = tests_weight1[
                    maxis_indices[network_indice]
                ] * choose_w1 + tests_weight1[second_network] * (1 - choose_w1)
                tests_weight2[minis_indices[network_indice]] = tests_weight2[
                    maxis_indices[network_indice]
                ] * choose_w2 + tests_weight2[second_network] * (1 - choose_w2)
                tests_weight3[minis_indices[network_indice]] = tests_weight3[
                    maxis_indices[network_indice]
                ] * choose_w3 + tests_weight3[second_network] * (1 - choose_w3)
                tests_weight4[minis_indices[network_indice]] = tests_weight4[
                    maxis_indices[network_indice]
                ] * choose_w4 + tests_weight4[second_network] * (1 - choose_w4)
                tests_weight5[minis_indices[network_indice]] = tests_weight5[
                    maxis_indices[network_indice]
                ] * choose_w5 + tests_weight5[second_network] * (1 - choose_w5)
                tests_bias1[minis_indices[network_indice]] = tests_bias1[
                    maxis_indices[network_indice]
                ] * choose_b1 + tests_bias1[second_network] * (1 - choose_b1)
                tests_bias2[minis_indices[network_indice]] = tests_bias2[
                    maxis_indices[network_indice]
                ] * choose_b2 + tests_bias2[second_network] * (1 - choose_b2)
                tests_bias3[minis_indices[network_indice]] = tests_bias3[
                    maxis_indices[network_indice]
                ] * choose_b3 + tests_bias3[second_network] * (1 - choose_b3)
                tests_bias4[minis_indices[network_indice]] = tests_bias4[
                    maxis_indices[network_indice]
                ] * choose_b4 + tests_bias4[second_network] * (1 - choose_b4)
                tests_bias5[minis_indices[network_indice]] = tests_bias5[
                    maxis_indices[network_indice]
                ] * choose_b5 + tests_bias5[second_network] * (1 - choose_b5)
                tests_weight1[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_weight2[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_weight3[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_weight4[minis_indices[network_indice]] += (
                    (numpy.random.rand(1, 64) *
                     (2 * mutation_change) - mutation_change)
                ) * numpy.heaviside(
                    numpy.random.rand(1, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_weight5[minis_indices[network_indice]] += (
                    (numpy.random.rand(64, 1) *
                     (2 * mutation_change) - mutation_change)
                ) * numpy.heaviside(
                    numpy.random.rand(64, 1) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias1[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias2[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias3[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias4[minis_indices[network_indice]] += (
                    (numpy.random.rand(1, 64) *
                     (2 * mutation_change) - mutation_change)
                ) * numpy.heaviside(
                    numpy.random.rand(1, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias5[minis_indices[network_indice]] += (
                    (numpy.random.rand(1, 1) *
                     (2 * mutation_change) - mutation_change)
                ) * numpy.heaviside(
                    numpy.random.rand(1, 1) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                print(
                    f"Coupling network #{network_indice + 1}... (testing)                    ",
                    end="\r",
                    flush=True,
                )
                self.weight1 = tests_weight1[minis_indices[network_indice]]
                self.weight2 = tests_weight2[minis_indices[network_indice]]
                self.weight3 = tests_weight3[minis_indices[network_indice]]
                self.weight4 = tests_weight4[minis_indices[network_indice]]
                self.weight5 = tests_weight5[minis_indices[network_indice]]
                self.b1 = tests_bias1[minis_indices[network_indice]]
                self.b2 = tests_bias2[minis_indices[network_indice]]
                self.b3 = tests_bias3[minis_indices[network_indice]]
                self.b4 = tests_bias4[minis_indices[network_indice]]
                self.b5 = tests_bias5[minis_indices[network_indice]]
                on_good_moves, on_bad_moves, good_moves, bad_moves = self.check_train()
                difference = abs(
                    ((on_good_moves / good_moves) -
                     (on_bad_moves / bad_moves)) * 100
                )
                success = (
                    (balance * on_good_moves + on_bad_moves)
                    / (balance * good_moves + bad_moves)
                    * 100
                )
                tests_results[minis_indices[network_indice]
                              ] = success - difference
                print(
                    f"Coupling network #{network_indice + 1}... Done.   ",
                    end="\r",
                    flush=True,
                )
                """
                random_matrix1 = numpy.random.rand(
                    64, 64) * (2 * mutation_change) - mutation_change
                rand1 = numpy.random.rand(
                    64, 64) * inverse_rate + (1 - inverse_rate)
                new_weight1 = numpy.heaviside(rand1, 0) * self.cweight1
                self.weight1 = self.weight1 + random_matrix1 * new_weight1
                """
            print("Coupling networks... Done.                           ")
            print(
                f"Mean performance : {(sum(tests_results) / len(tests_results))}")
            if iters % 3 == 0 and iters:
                for loop in range(population):
                    print(
                        f"Saving networks... ({loop}/{population})",
                        end="\r",
                        flush=True,
                    )
                    self.array_to_csv(
                        tests_weight1[loop], f"nns/{loop}-w1.csv")
                    self.array_to_csv(
                        tests_weight2[loop], f"nns/{loop}-w2.csv")
                    self.array_to_csv(
                        tests_weight3[loop], f"nns/{loop}-w3.csv")
                    self.array_to_csv(
                        tests_weight4[loop], f"nns/{loop}-w4.csv")
                    self.array_to_csv(
                        tests_weight5[loop], f"nns/{loop}-w5.csv")
                    self.array_to_csv(tests_bias1[loop], f"nns/{loop}-b1.csv")
                    self.array_to_csv(tests_bias2[loop], f"nns/{loop}-b2.csv")
                    self.array_to_csv(tests_bias3[loop], f"nns/{loop}-b3.csv")
                    self.array_to_csv(tests_bias4[loop], f"nns/{loop}-b4.csv")
                    self.array_to_csv(tests_bias5[loop], f"nns/{loop}-b5.csv")
                print("Saving networks... Done.          ")
                print("Saving tests result...", end=" ", flush=True)
                saved_results = list()
                for element in tests_results:
                    saved_results.append([float(element)])
                self.array_to_csv(saved_results, "nns/results.csv")
                print("Done.")

    def genetic_random(self):
        """Random NNs for genetic algorithm."""
        tests_weight1 = list()
        tests_weight2 = list()
        tests_weight3 = list()
        tests_weight4 = list()
        tests_weight5 = list()
        tests_bias1 = list()
        tests_bias2 = list()
        tests_bias3 = list()
        tests_bias4 = list()
        tests_bias5 = list()
        number = int(input("Population : "))
        for loop in range(number):
            print(
                f"Generating random networks... ({loop}/{number})", end="\r", flush=True
            )
            tests_weight1.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_weight2.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_weight3.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_weight4.append(numpy.random.rand(1, 64) * 2 - 1)
            tests_weight5.append(numpy.random.rand(64, 1) * 2 - 1)
            tests_bias1.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_bias2.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_bias3.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_bias4.append(numpy.random.rand(1, 64) * 2 - 1)
            tests_bias5.append(numpy.random.rand(1, 1) * 2 - 1)
        print("Generating random networks... Done.          ")
        for loop in range(number):
            print(
                f"Saving random networks... ({loop}/{number})", end="\r", flush=True)
            self.array_to_csv(tests_weight1[loop], f"nns/{loop}-w1.csv")
            self.array_to_csv(tests_weight2[loop], f"nns/{loop}-w2.csv")
            self.array_to_csv(tests_weight3[loop], f"nns/{loop}-w3.csv")
            self.array_to_csv(tests_weight4[loop], f"nns/{loop}-w4.csv")
            self.array_to_csv(tests_weight5[loop], f"nns/{loop}-w5.csv")
            self.array_to_csv(tests_bias1[loop], f"nns/{loop}-b1.csv")
            self.array_to_csv(tests_bias2[loop], f"nns/{loop}-b2.csv")
            self.array_to_csv(tests_bias3[loop], f"nns/{loop}-b3.csv")
            self.array_to_csv(tests_bias4[loop], f"nns/{loop}-b4.csv")
            self.array_to_csv(tests_bias5[loop], f"nns/{loop}-b5.csv")
        print("Saving random networks... Done.          ")
        print("Configure testing...")
        self.change_files()
        balance = float(
            input(
                "Balance between good moves and bad moves (>1 to enhance good moves success rate) : "
            )
        )
        print("Done.")
        tests_results = list()
        for loop in range(number):
            print(
                f"Testing networks... ({loop}/{number})", end="\r", flush=True)
            self.weight1 = tests_weight1[loop]
            self.weight2 = tests_weight2[loop]
            self.weight3 = tests_weight3[loop]
            self.weight4 = tests_weight4[loop]
            self.weight5 = tests_weight5[loop]
            self.b1 = tests_bias1[loop]
            self.b2 = tests_bias2[loop]
            self.b3 = tests_bias3[loop]
            self.b4 = tests_bias4[loop]
            self.b5 = tests_bias5[loop]
            on_good_moves, on_bad_moves, good_moves, bad_moves = self.check_train()
            difference = abs(
                ((on_good_moves / good_moves) - (on_bad_moves / bad_moves)) * 100
            )
            success = (
                (balance * on_good_moves + on_bad_moves)
                / (balance * good_moves + bad_moves)
                * 100
            )
            tests_results.append([success - difference])
        print("Testing networks... Done.          ")
        print("Saving tests result...", end=" ", flush=True)
        self.array_to_csv(tests_results, "nns/results.csv")
        print("Done.")
        print("Reseting neural network...", end=" ", flush=True)
        self.weight1 = self.csv_to_array("w1.csv")
        self.weight2 = self.csv_to_array("w2.csv")
        self.weight3 = self.csv_to_array("w3.csv")
        self.weight4 = self.csv_to_array("w4.csv")
        self.weight5 = self.csv_to_array("w5.csv")
        self.b1 = self.csv_to_array("b1.csv")
        self.b2 = self.csv_to_array("b2.csv")
        self.b3 = self.csv_to_array("b3.csv")
        self.b4 = self.csv_to_array("b4.csv")
        self.b5 = self.csv_to_array("b5.csv")
        print("Done.")
        print("Configuring network...", end=" ", flush=True)
        self.genetic_train_settings["population"] = number
        self.genetic_save(confirmation=False)
        print("Done.")

    def genetic_configure(self):
        """Configure genetic training algorithm."""

        def sprint(value):
            centered = value.center(18)
            print("********** {0} **********".format(centered))

        sprint("Configuration")
        confirm = input("Do you want to configure training ? [y/N] ")
        if confirm.lower() in ("y", "yes"):
            sprint("Select files")
            self.change_files()
            sprint("Configure training")
            self.genetic_train_settings["max_iters"] = int(
                input("Maximum iterations : ")
            )
            self.genetic_train_settings["max_success"] = float(
                input("Maximum success rate : ")
            )
            self.genetic_train_settings["balance"] = float(
                input(
                    "Balance between good moves and bad moves (>1 to enhance good moves success rate) : "
                )
            )
            self.genetic_train_settings["mutation_rate"] = float(
                input("Mutation rate (in percents) : ")
            )
            self.genetic_train_settings["mutation_change"] = float(
                input("Mutation change : ")
            )
            self.genetic_save()

    def genetic_save(self, confirmation=True):
        """Save genetic algorithm configuration."""
        if confirmation:
            confirm = input("Save configuration ? [y/N] ")
            if confirm.lower() in ("y", "yes"):
                with open("nns/settings.json", "w") as file:
                    file.write(json.dumps(self.genetic_train_settings))
        else:
            with open("nns/settings.json", "w") as file:
                file.write(json.dumps(self.genetic_train_settings))

    def check_always_same(self):
        """Check success rating on good moves and on bad moves."""
        with open(self.train_good) as file:
            file_goodmoves = file.read()
            file.close()
        with open(self.train_bad) as file:
            file_badmoves = file.read()
            file.close()
        file_goodmoves = file_goodmoves.split("\n\n")
        file_badmoves = file_badmoves.split("\n\n")
        errs = 0
        good = 0
        good_on_good_moves = 0
        good_on_bad_moves = 0
        for inputs in file_goodmoves:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            if self.check_move(pos, mve):
                good += 1
                good_on_good_moves += 1
            else:
                errs += 1
        for inputs in file_badmoves:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            if not self.check_move(pos, mve):
                good += 1
                good_on_bad_moves += 1
            else:
                errs += 1
        print(
            "Success rate on good moves : {0}%".format(
                good_on_good_moves / len(file_goodmoves) * 100
            )
        )
        print(
            "Success rate on bad moves : {0}%".format(
                good_on_bad_moves / len(file_badmoves) * 100
            )
        )

    def check_difference(self):
        """Check success rate on good moves and on bad moves and return it."""
        file_goodmoves = self.train_good
        file_badmoves = self.train_bad
        file_goodmoves = file_goodmoves.split("\n\n")
        file_badmoves = file_badmoves.split("\n\n")
        errs = 0
        good = 0
        good_on_good_moves = 0
        good_on_bad_moves = 0
        for inputs in file_goodmoves:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            if self.check_move(pos, mve):
                good += 1
                good_on_good_moves += 1
            else:
                errs += 1
        for inputs in file_badmoves:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            if not self.check_move(pos, mve):
                good += 1
                good_on_bad_moves += 1
            else:
                errs += 1
        return abs(
            (good_on_good_moves / len(file_goodmoves) * 100)
            - (good_on_bad_moves / len(file_badmoves) * 100)
        )

    @staticmethod
    def array_to_csv(array, csv_path):
        """Write array in csv_path CSV file."""
        array = list(array)
        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(array)
            file.close()
        return 0

    def save(self):
        """Save Neural Network."""
        self.array_to_csv(self.weight1, "w1.csv")
        self.array_to_csv(self.weight2, "w2.csv")
        self.array_to_csv(self.weight3, "w3.csv")
        self.array_to_csv(self.weight4, "w4.csv")
        self.array_to_csv(self.weight5, "w5.csv")
        self.array_to_csv(self.b1, "b1.csv")
        self.array_to_csv(self.b2, "b2.csv")
        self.array_to_csv(self.b3, "b3.csv")
        self.array_to_csv(self.b4, "b4.csv")
        self.array_to_csv(self.b5, "b5.csv")
        self.array_to_csv(self.cweight1, "cw1.csv")
        self.array_to_csv(self.cweight2, "cw2.csv")
        self.array_to_csv(self.cweight3, "cw3.csv")
        self.array_to_csv(self.cweight4, "cw4.csv")
        self.array_to_csv(self.cweight5, "cw5.csv")
        self.array_to_csv(self.cb1, "cb1.csv")
        self.array_to_csv(self.cb2, "cb2.csv")
        self.array_to_csv(self.cb3, "cb3.csv")
        self.array_to_csv(self.cb4, "cb4.csv")
        self.array_to_csv(self.cb5, "cb5.csv")

    @staticmethod
    def normalisation(array):
        """Normalisation."""
        return numpy.clip(array, 0, 1)

    def change_files(self):
        """Change files locations."""
        self.train_good = input("Good moves training file : ")
        self.train_bad = input("Bad moves training file : ")
        self.test_good = input("Good moves test file : ")
        self.test_bad = input("Bad moves test file : ")
        self.genetic_train_settings["train_good"] = self.train_good
        self.genetic_train_settings["test_good"] = self.test_good
        self.genetic_train_settings["train_bad"] = self.train_bad
        self.genetic_train_settings["test_bad"] = self.test_bad
        self.train_good = open(self.train_good).read().split("\n\n")
        self.train_bad = open(self.train_bad).read().split("\n\n")
        self.test_good = open(self.test_good).read().split("\n\n")
        self.test_bad = open(self.test_bad).read().split("\n\n")

    def masters_check_train(self):
        """Check NN on train dataset."""
        file_goodmoves = self.masters_train_good
        file_badmoves = self.masters_train_bad
        errs = 0
        good = 0
        correct_on_good_moves = 0
        correct_on_bad_moves = 0
        for inputs in file_goodmoves:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            if self.check_move(pos, mve):
                good += 1
                correct_on_good_moves += 1
            else:
                errs += 1
        for inputs in file_badmoves:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            if not self.check_move(pos, mve):
                good += 1
                correct_on_bad_moves += 1
            else:
                errs += 1
        return (
            correct_on_good_moves,
            correct_on_bad_moves,
            len(file_goodmoves),
            len(file_badmoves),
        )

    def masters_genetic_train(self, masters_good_moves, masters_bad_moves, config):
        """
        Genetic training algorithm.

        New training algorithm using a real genetic algorithm.
        """
        self.load_networks()

        def sprint(value):
            centered = value.center(18)
            print("********** {0} **********".format(centered))

        tests_weight1 = self.tests_weight1
        tests_weight2 = self.tests_weight2
        tests_weight3 = self.tests_weight3
        tests_weight4 = self.tests_weight4
        tests_weight5 = self.tests_weight5
        tests_bias1 = self.tests_bias1
        tests_bias2 = self.tests_bias2
        tests_bias3 = self.tests_bias3
        tests_bias4 = self.tests_bias4
        tests_bias5 = self.tests_bias5
        self.masters_train_good = masters_good_moves.split("\n\n")
        self.masters_train_bad = masters_bad_moves.split("\n\n")
        max_iters = 200
        max_success = 100
        inverse_rate = 100 / config["mutation_rate"]
        mutation_change = config["mutation_change"]
        sprint("Initialize")
        iters = 0
        tests_results = list()
        population = self.genetic_train_settings["population"]
        for loop in range(population):
            print(
                f"Testing networks... ({loop}/{population})", end="\r", flush=True)
            self.weight1 = tests_weight1[loop]
            self.weight2 = tests_weight2[loop]
            self.weight3 = tests_weight3[loop]
            self.weight4 = tests_weight4[loop]
            self.weight5 = tests_weight5[loop]
            self.b1 = tests_bias1[loop]
            self.b2 = tests_bias2[loop]
            self.b3 = tests_bias3[loop]
            self.b4 = tests_bias4[loop]
            self.b5 = tests_bias5[loop]
            (
                on_good_moves,
                on_bad_moves,
                good_moves,
                bad_moves,
            ) = self.masters_check_train()
            success = (
                (on_good_moves / good_moves) ** 2 * (on_bad_moves / bad_moves)
            ) * 100
            tests_results.append([success])
        print("Testing networks... Done.          ")
        print("Saving tests result...", end=" ", flush=True)
        self.array_to_csv(tests_results, "nns/results.csv")
        print("Done.")
        print("Reseting neural network...", end=" ", flush=True)
        self.weight1 = self.csv_to_array("w1.csv")
        self.weight2 = self.csv_to_array("w2.csv")
        self.weight3 = self.csv_to_array("w3.csv")
        self.weight4 = self.csv_to_array("w4.csv")
        self.weight5 = self.csv_to_array("w5.csv")
        self.b1 = self.csv_to_array("b1.csv")
        self.b2 = self.csv_to_array("b2.csv")
        self.b3 = self.csv_to_array("b3.csv")
        self.b4 = self.csv_to_array("b4.csv")
        self.b5 = self.csv_to_array("b5.csv")
        print("Done.")
        print("Loading tests results...", end=" ", flush=True)
        tests_results = self.csv_to_array("nns/results.csv")
        tests_results = list(tests_results)
        for indice, element in enumerate(tests_results):
            tests_results[indice] = element[0]
        print("Done.")
        # https://stackoverflow.com/questions/16225677/get-the-second-largest-number-in-a-list-in-linear-time
        while iters < max_iters and success < max_success:
            iters += 1
            sprint("Training #{0}".format(iters))
            print("Selecting best networks...", end=" ", flush=True)
            maxis_brut = sorted(tests_results, reverse=True)[:32]
            maxis = list()
            for element in maxis_brut:
                maxis.append(element)
            maxis_indices = sorted(
                range(len(tests_results)),
                key=lambda sub: tests_results[sub],
                reverse=True,
            )[:4]
            minis_brut = sorted(tests_results)[:4]
            minis = list()
            for element in minis_brut:
                minis.append(element)
            minis_indices = sorted(
                range(len(tests_results)), key=lambda sub: tests_results[sub]
            )[:4]
            print("Done.")
            liste = []
            for count in range(4):
                liste.append(
                    "#" + str(minis_indices[count]) +
                    " (" + str(minis[count]) + ")"
                )
            print(f"Worst networks : {', '.join(liste)}")
            liste = []
            for count in range(4):
                liste.append(
                    "#" + str(maxis_indices[count]) +
                    " (" + str(maxis[count]) + ")"
                )
            print(f"Best networks : {', '.join(liste)}")
            for network_indice in range(4):
                print(
                    f"Coupling network #{network_indice + 1}... (selecting second network)",
                    end="\r",
                    flush=True,
                )
                cont = True
                while cont:
                    cont = False
                    rand = random.randint(0, population - 1)
                    if rand in minis_indices or rand in maxis_indices:
                        cont = True
                second_network = rand
                print(
                    f"Coupling network #{network_indice + 1}... (generating coupling matrixes)",
                    end="\r",
                    flush=True,
                )
                choose_w1 = numpy.zeros((64, 64))
                choose_w2 = numpy.zeros((64, 64))
                choose_w3 = numpy.zeros((64, 64))
                choose_matrixes1 = [choose_w1, choose_w2, choose_w3]
                for choose_matrix in choose_matrixes1:
                    # True : column else line
                    direction = bool(random.getrandbits(1))
                    choose = bool(random.getrandbits(1))
                    for line in range(len(choose_matrix)):
                        for column in range(len(choose_matrix[0])):
                            if random.random() < 0.001:
                                choose = not choose
                            fill = int(choose)
                            if direction:
                                choose_matrix[line][column] = fill
                            else:
                                choose_matrix[column][line] = fill
                choose = bool(random.getrandbits(1))
                choose_w5 = numpy.zeros((64, 1))
                for line in range(64):
                    if random.random() < 0.0625:
                        choose = not choose
                    choose_w5[line][0] = int(choose)
                choose = bool(random.getrandbits(1))
                choose_w4 = numpy.zeros((1, 64))
                for column in range(64):
                    if random.random() < 0.0625:
                        choose = not choose
                    choose_w4[0][column] = int(choose)
                choose_b1 = choose_w1
                choose_b2 = choose_w2
                choose_b3 = choose_w3
                choose_b4 = numpy.zeros((1, 64))
                choose = bool(random.getrandbits(1))
                for column in range(64):
                    if random.random() < 0.0625:
                        choose = not choose
                    choose_b4[0][column] = int(choose)
                choose_b5 = numpy.array([[int(bool(random.getrandbits(1)))]])
                print(
                    f"Coupling network #{network_indice + 1}... (coupling)                    ",
                    end="\r",
                    flush=True,
                )
                tests_weight1[minis_indices[network_indice]] = tests_weight1[
                    maxis_indices[network_indice]
                ] * choose_w1 + tests_weight1[second_network] * (1 - choose_w1)
                tests_weight2[minis_indices[network_indice]] = tests_weight2[
                    maxis_indices[network_indice]
                ] * choose_w2 + tests_weight2[second_network] * (1 - choose_w2)
                tests_weight3[minis_indices[network_indice]] = tests_weight3[
                    maxis_indices[network_indice]
                ] * choose_w3 + tests_weight3[second_network] * (1 - choose_w3)
                tests_weight4[minis_indices[network_indice]] = tests_weight4[
                    maxis_indices[network_indice]
                ] * choose_w4 + tests_weight4[second_network] * (1 - choose_w4)
                tests_weight5[minis_indices[network_indice]] = tests_weight5[
                    maxis_indices[network_indice]
                ] * choose_w5 + tests_weight5[second_network] * (1 - choose_w5)
                tests_bias1[minis_indices[network_indice]] = tests_bias1[
                    maxis_indices[network_indice]
                ] * choose_b1 + tests_bias1[second_network] * (1 - choose_b1)
                tests_bias2[minis_indices[network_indice]] = tests_bias2[
                    maxis_indices[network_indice]
                ] * choose_b2 + tests_bias2[second_network] * (1 - choose_b2)
                tests_bias3[minis_indices[network_indice]] = tests_bias3[
                    maxis_indices[network_indice]
                ] * choose_b3 + tests_bias3[second_network] * (1 - choose_b3)
                tests_bias4[minis_indices[network_indice]] = tests_bias4[
                    maxis_indices[network_indice]
                ] * choose_b4 + tests_bias4[second_network] * (1 - choose_b4)
                tests_bias5[minis_indices[network_indice]] = tests_bias5[
                    maxis_indices[network_indice]
                ] * choose_b5 + tests_bias5[second_network] * (1 - choose_b5)
                tests_weight1[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_weight2[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_weight3[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_weight4[minis_indices[network_indice]] += (
                    (numpy.random.rand(1, 64) *
                     (2 * mutation_change) - mutation_change)
                ) * numpy.heaviside(
                    numpy.random.rand(1, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_weight5[minis_indices[network_indice]] += (
                    (numpy.random.rand(64, 1) *
                     (2 * mutation_change) - mutation_change)
                ) * numpy.heaviside(
                    numpy.random.rand(64, 1) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias1[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias2[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias3[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias4[minis_indices[network_indice]] += (
                    (numpy.random.rand(1, 64) *
                     (2 * mutation_change) - mutation_change)
                ) * numpy.heaviside(
                    numpy.random.rand(1, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias5[minis_indices[network_indice]] += (
                    (numpy.random.rand(1, 1) *
                     (2 * mutation_change) - mutation_change)
                ) * numpy.heaviside(
                    numpy.random.rand(1, 1) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                print(
                    f"Coupling network #{network_indice + 1}... (testing)                    ",
                    end="\r",
                    flush=True,
                )
                self.weight1 = tests_weight1[minis_indices[network_indice]]
                self.weight2 = tests_weight2[minis_indices[network_indice]]
                self.weight3 = tests_weight3[minis_indices[network_indice]]
                self.weight4 = tests_weight4[minis_indices[network_indice]]
                self.weight5 = tests_weight5[minis_indices[network_indice]]
                self.b1 = tests_bias1[minis_indices[network_indice]]
                self.b2 = tests_bias2[minis_indices[network_indice]]
                self.b3 = tests_bias3[minis_indices[network_indice]]
                self.b4 = tests_bias4[minis_indices[network_indice]]
                self.b5 = tests_bias5[minis_indices[network_indice]]
                (
                    on_good_moves,
                    on_bad_moves,
                    good_moves,
                    bad_moves,
                ) = self.masters_check_train()
                success = (
                    (on_good_moves / good_moves) ** 2 *
                    (on_bad_moves / bad_moves)
                ) * 100
                tests_results[minis_indices[network_indice]] = success
                print(
                    f"Coupling network #{network_indice + 1}... Done.   ",
                    end="\r",
                    flush=True,
                )
                """
                random_matrix1 = numpy.random.rand(
                    64, 64) * (2 * mutation_change) - mutation_change
                rand1 = numpy.random.rand(
                    64, 64) * inverse_rate + (1 - inverse_rate)
                new_weight1 = numpy.heaviside(rand1, 0) * self.cweight1
                self.weight1 = self.weight1 + random_matrix1 * new_weight1
                """
            print("Coupling networks... Done.                           ")
            print(
                f"Mean performance : {(sum(tests_results) / len(tests_results))}")
        for loop in range(population):
            print(
                f"Saving networks... ({loop}/{population})", end="\r", flush=True)
            self.array_to_csv(tests_weight1[loop], f"nns/{loop}-w1.csv")
            self.array_to_csv(tests_weight2[loop], f"nns/{loop}-w2.csv")
            self.array_to_csv(tests_weight3[loop], f"nns/{loop}-w3.csv")
            self.array_to_csv(tests_weight4[loop], f"nns/{loop}-w4.csv")
            self.array_to_csv(tests_weight5[loop], f"nns/{loop}-w5.csv")
            self.array_to_csv(tests_bias1[loop], f"nns/{loop}-b1.csv")
            self.array_to_csv(tests_bias2[loop], f"nns/{loop}-b2.csv")
            self.array_to_csv(tests_bias3[loop], f"nns/{loop}-b3.csv")
            self.array_to_csv(tests_bias4[loop], f"nns/{loop}-b4.csv")
            self.array_to_csv(tests_bias5[loop], f"nns/{loop}-b5.csv")
        print("Saving networks... Done.          ")
        print("Saving tests result...", end=" ", flush=True)
        saved_results = list()
        for element in tests_results:
            saved_results.append([float(element)])
        self.array_to_csv(saved_results, "nns/results.csv")
        print("Done.")

    def masters_random(self):
        """Random NNs for genetic masters training algorithm."""
        tests_weight1 = list()
        tests_weight2 = list()
        tests_weight3 = list()
        tests_weight4 = list()
        tests_weight5 = list()
        tests_bias1 = list()
        tests_bias2 = list()
        tests_bias3 = list()
        tests_bias4 = list()
        tests_bias5 = list()
        number = int(input("Population : "))
        for loop in range(number):
            print(
                f"Generating random networks... ({loop}/{number})", end="\r", flush=True
            )
            tests_weight1.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_weight2.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_weight3.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_weight4.append(numpy.random.rand(1, 64) * 2 - 1)
            tests_weight5.append(numpy.random.rand(64, 1) * 2 - 1)
            tests_bias1.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_bias2.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_bias3.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_bias4.append(numpy.random.rand(1, 64) * 2 - 1)
            tests_bias5.append(numpy.random.rand(1, 1) * 2 - 1)
        print("Generating random networks... Done.          ")
        for loop in range(number):
            print(
                f"Saving random networks... ({loop}/{number})", end="\r", flush=True)
            self.array_to_csv(tests_weight1[loop], f"nns/{loop}-w1.csv")
            self.array_to_csv(tests_weight2[loop], f"nns/{loop}-w2.csv")
            self.array_to_csv(tests_weight3[loop], f"nns/{loop}-w3.csv")
            self.array_to_csv(tests_weight4[loop], f"nns/{loop}-w4.csv")
            self.array_to_csv(tests_weight5[loop], f"nns/{loop}-w5.csv")
            self.array_to_csv(tests_bias1[loop], f"nns/{loop}-b1.csv")
            self.array_to_csv(tests_bias2[loop], f"nns/{loop}-b2.csv")
            self.array_to_csv(tests_bias3[loop], f"nns/{loop}-b3.csv")
            self.array_to_csv(tests_bias4[loop], f"nns/{loop}-b4.csv")
            self.array_to_csv(tests_bias5[loop], f"nns/{loop}-b5.csv")
        print("Saving random networks... Done.          ")

    def basics_training(self, good_moves_list: list, bad_moves_list: list) -> NoneType:
        def sprint(value):
            centered = value.center(18)
            print("********** {0} **********".format(centered))

        tests_weight1 = self.tests_weight1
        tests_weight2 = self.tests_weight2
        tests_weight3 = self.tests_weight3
        tests_weight4 = self.tests_weight4
        tests_weight5 = self.tests_weight5
        tests_bias1 = self.tests_bias1
        tests_bias2 = self.tests_bias2
        tests_bias3 = self.tests_bias3
        tests_bias4 = self.tests_bias4
        tests_bias5 = self.tests_bias5
        self.masters_train_good = good_moves_list
        self.masters_train_bad = bad_moves_list
        config = self.genetic_train_settings
        max_iters = 200
        max_success = 100
        inverse_rate = 100 / config["mutation_rate"]
        mutation_change = config["mutation_change"]
        sprint("Initialize")
        iters = 0
        tests_results = list()
        population = self.genetic_train_settings["population"]
        for loop in range(population):
            print(
                f"Testing networks... ({loop}/{population})", end="\r", flush=True)
            self.weight1 = tests_weight1[loop]
            self.weight2 = tests_weight2[loop]
            self.weight3 = tests_weight3[loop]
            self.weight4 = tests_weight4[loop]
            self.weight5 = tests_weight5[loop]
            self.b1 = tests_bias1[loop]
            self.b2 = tests_bias2[loop]
            self.b3 = tests_bias3[loop]
            self.b4 = tests_bias4[loop]
            self.b5 = tests_bias5[loop]
            (
                on_good_moves,
                on_bad_moves,
                good_moves,
                bad_moves,
            ) = self.masters_check_train()
            success = (
                (on_good_moves / good_moves) ** 2 * (on_bad_moves / bad_moves)
            ) * 100
            tests_results.append([success])
        print("Testing networks... Done.          ")
        print("Saving tests result...", end=" ", flush=True)
        self.array_to_csv(tests_results, "nns/results.csv")
        print("Done.")
        print("Reseting neural network...", end=" ", flush=True)
        self.weight1 = self.csv_to_array("w1.csv")
        self.weight2 = self.csv_to_array("w2.csv")
        self.weight3 = self.csv_to_array("w3.csv")
        self.weight4 = self.csv_to_array("w4.csv")
        self.weight5 = self.csv_to_array("w5.csv")
        self.b1 = self.csv_to_array("b1.csv")
        self.b2 = self.csv_to_array("b2.csv")
        self.b3 = self.csv_to_array("b3.csv")
        self.b4 = self.csv_to_array("b4.csv")
        self.b5 = self.csv_to_array("b5.csv")
        print("Done.")
        print("Loading tests results...", end=" ", flush=True)
        tests_results = self.csv_to_array("nns/results.csv")
        tests_results = list(tests_results)
        for indice, element in enumerate(tests_results):
            tests_results[indice] = element[0]
        print("Done.")
        # https://stackoverflow.com/questions/16225677/get-the-second-largest-number-in-a-list-in-linear-time
        while iters < max_iters and success < max_success:
            print("debug")
            print(on_good_moves)
            print(on_bad_moves)
            print("end")
            iters += 1
            sprint("Training #{0}".format(iters))
            print("Selecting best networks...", end=" ", flush=True)
            maxis_brut = sorted(tests_results, reverse=True)[:32]
            maxis = list()
            for element in maxis_brut:
                maxis.append(element)
            maxis_indices = sorted(
                range(len(tests_results)),
                key=lambda sub: tests_results[sub],
                reverse=True,
            )[:4]
            minis_brut = sorted(tests_results)[:4]
            minis = list()
            for element in minis_brut:
                minis.append(element)
            minis_indices = sorted(
                range(len(tests_results)), key=lambda sub: tests_results[sub]
            )[:4]
            print("Done.")
            liste = []
            for count in range(4):
                liste.append(
                    "#" + str(minis_indices[count]) +
                    " (" + str(minis[count]) + ")"
                )
            print(f"Worst networks : {', '.join(liste)}")
            liste = []
            for count in range(4):
                liste.append(
                    "#" + str(maxis_indices[count]) +
                    " (" + str(maxis[count]) + ")"
                )
            print(f"Best networks : {', '.join(liste)}")
            for network_indice in range(4):
                print(
                    f"Coupling network #{network_indice + 1}... (selecting second network)",
                    end="\r",
                    flush=True,
                )
                cont = True
                while cont:
                    cont = False
                    rand = random.randint(0, population - 1)
                    if rand in minis_indices or rand in maxis_indices:
                        cont = True
                second_network = rand
                print(
                    f"Coupling network #{network_indice + 1}... (generating coupling matrixes)",
                    end="\r",
                    flush=True,
                )
                choose_w1 = numpy.zeros((64, 64))
                choose_w2 = numpy.zeros((64, 64))
                choose_w3 = numpy.zeros((64, 64))
                choose_matrixes1 = [choose_w1, choose_w2, choose_w3]
                for choose_matrix in choose_matrixes1:
                    # True : column else line
                    direction = bool(random.getrandbits(1))
                    choose = bool(random.getrandbits(1))
                    for line in range(len(choose_matrix)):
                        for column in range(len(choose_matrix[0])):
                            if random.random() < 0.001:
                                choose = not choose
                            fill = int(choose)
                            if direction:
                                choose_matrix[line][column] = fill
                            else:
                                choose_matrix[column][line] = fill
                choose = bool(random.getrandbits(1))
                choose_w5 = numpy.zeros((64, 1))
                for line in range(64):
                    if random.random() < 0.0625:
                        choose = not choose
                    choose_w5[line][0] = int(choose)
                choose = bool(random.getrandbits(1))
                choose_w4 = numpy.zeros((1, 64))
                for column in range(64):
                    if random.random() < 0.0625:
                        choose = not choose
                    choose_w4[0][column] = int(choose)
                choose_b1 = choose_w1
                choose_b2 = choose_w2
                choose_b3 = choose_w3
                choose_b4 = numpy.zeros((1, 64))
                choose = bool(random.getrandbits(1))
                for column in range(64):
                    if random.random() < 0.0625:
                        choose = not choose
                    choose_b4[0][column] = int(choose)
                choose_b5 = numpy.array([[int(bool(random.getrandbits(1)))]])
                print(
                    f"Coupling network #{network_indice + 1}... (coupling)                    ",
                    end="\r",
                    flush=True,
                )
                tests_weight1[minis_indices[network_indice]] = tests_weight1[
                    maxis_indices[network_indice]
                ] * choose_w1 + tests_weight1[second_network] * (1 - choose_w1)
                tests_weight2[minis_indices[network_indice]] = tests_weight2[
                    maxis_indices[network_indice]
                ] * choose_w2 + tests_weight2[second_network] * (1 - choose_w2)
                tests_weight3[minis_indices[network_indice]] = tests_weight3[
                    maxis_indices[network_indice]
                ] * choose_w3 + tests_weight3[second_network] * (1 - choose_w3)
                tests_weight4[minis_indices[network_indice]] = tests_weight4[
                    maxis_indices[network_indice]
                ] * choose_w4 + tests_weight4[second_network] * (1 - choose_w4)
                tests_weight5[minis_indices[network_indice]] = tests_weight5[
                    maxis_indices[network_indice]
                ] * choose_w5 + tests_weight5[second_network] * (1 - choose_w5)
                tests_bias1[minis_indices[network_indice]] = tests_bias1[
                    maxis_indices[network_indice]
                ] * choose_b1 + tests_bias1[second_network] * (1 - choose_b1)
                tests_bias2[minis_indices[network_indice]] = tests_bias2[
                    maxis_indices[network_indice]
                ] * choose_b2 + tests_bias2[second_network] * (1 - choose_b2)
                tests_bias3[minis_indices[network_indice]] = tests_bias3[
                    maxis_indices[network_indice]
                ] * choose_b3 + tests_bias3[second_network] * (1 - choose_b3)
                tests_bias4[minis_indices[network_indice]] = tests_bias4[
                    maxis_indices[network_indice]
                ] * choose_b4 + tests_bias4[second_network] * (1 - choose_b4)
                tests_bias5[minis_indices[network_indice]] = tests_bias5[
                    maxis_indices[network_indice]
                ] * choose_b5 + tests_bias5[second_network] * (1 - choose_b5)
                tests_weight1[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_weight2[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_weight3[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_weight4[minis_indices[network_indice]] += (
                    (numpy.random.rand(1, 64) *
                     (2 * mutation_change) - mutation_change)
                ) * numpy.heaviside(
                    numpy.random.rand(1, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_weight5[minis_indices[network_indice]] += (
                    (numpy.random.rand(64, 1) *
                     (2 * mutation_change) - mutation_change)
                ) * numpy.heaviside(
                    numpy.random.rand(64, 1) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias1[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias2[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias3[minis_indices[network_indice]] += (
                    (
                        numpy.random.rand(64, 64) * (2 * mutation_change)
                        - mutation_change
                    )
                ) * numpy.heaviside(
                    numpy.random.rand(64, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias4[minis_indices[network_indice]] += (
                    (numpy.random.rand(1, 64) *
                     (2 * mutation_change) - mutation_change)
                ) * numpy.heaviside(
                    numpy.random.rand(1, 64) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                tests_bias5[minis_indices[network_indice]] += (
                    (numpy.random.rand(1, 1) *
                     (2 * mutation_change) - mutation_change)
                ) * numpy.heaviside(
                    numpy.random.rand(1, 1) * inverse_rate +
                    (1 - inverse_rate), 0
                )
                print(
                    f"Coupling network #{network_indice + 1}... (testing)                    ",
                    end="\r",
                    flush=True,
                )
                self.weight1 = tests_weight1[minis_indices[network_indice]]
                self.weight2 = tests_weight2[minis_indices[network_indice]]
                self.weight3 = tests_weight3[minis_indices[network_indice]]
                self.weight4 = tests_weight4[minis_indices[network_indice]]
                self.weight5 = tests_weight5[minis_indices[network_indice]]
                self.b1 = tests_bias1[minis_indices[network_indice]]
                self.b2 = tests_bias2[minis_indices[network_indice]]
                self.b3 = tests_bias3[minis_indices[network_indice]]
                self.b4 = tests_bias4[minis_indices[network_indice]]
                self.b5 = tests_bias5[minis_indices[network_indice]]
                (
                    on_good_moves,
                    on_bad_moves,
                    good_moves,
                    bad_moves,
                ) = self.masters_check_train()
                success = (
                    (on_good_moves / good_moves) ** 2 *
                    (on_bad_moves / bad_moves)
                ) * 100
                tests_results[minis_indices[network_indice]] = success
                print(
                    f"Coupling network #{network_indice + 1}... Done.   ",
                    end="\r",
                    flush=True,
                )
                """
                random_matrix1 = numpy.random.rand(
                    64, 64) * (2 * mutation_change) - mutation_change
                rand1 = numpy.random.rand(
                    64, 64) * inverse_rate + (1 - inverse_rate)
                new_weight1 = numpy.heaviside(rand1, 0) * self.cweight1
                self.weight1 = self.weight1 + random_matrix1 * new_weight1
                """
            print("Coupling networks... Done.                           ")
            print(
                f"Mean performance : {(sum(tests_results) / len(tests_results))}")
        for loop in range(population):
            print(
                f"Saving networks... ({loop}/{population})", end="\r", flush=True)
            self.array_to_csv(tests_weight1[loop], f"nns/{loop}-w1.csv")
            self.array_to_csv(tests_weight2[loop], f"nns/{loop}-w2.csv")
            self.array_to_csv(tests_weight3[loop], f"nns/{loop}-w3.csv")
            self.array_to_csv(tests_weight4[loop], f"nns/{loop}-w4.csv")
            self.array_to_csv(tests_weight5[loop], f"nns/{loop}-w5.csv")
            self.array_to_csv(tests_bias1[loop], f"nns/{loop}-b1.csv")
            self.array_to_csv(tests_bias2[loop], f"nns/{loop}-b2.csv")
            self.array_to_csv(tests_bias3[loop], f"nns/{loop}-b3.csv")
            self.array_to_csv(tests_bias4[loop], f"nns/{loop}-b4.csv")
            self.array_to_csv(tests_bias5[loop], f"nns/{loop}-b5.csv")
        print("Saving networks... Done.          ")
        print("Saving tests result...", end=" ", flush=True)
        saved_results = list()
        for element in tests_results:
            saved_results.append([float(element)])
        self.array_to_csv(saved_results, "nns/results.csv")
        print("Done.")

    def save(self, nn: int) -> None:
        for layer in range(LAYERS):
            numpy.savetxt(f"nns/{nn}-wpawns-{layer}.csv",
                          self.w_pawns[layer], delimiter=",")
            numpy.savetxt(f"nns/{nn}-wpieces-{layer}.csv",
                          self.w_pieces[layer], delimiter=",")
            numpy.savetxt(f"nns/{nn}-bpawns-{layer}.csv",
                          self.b_pawns[layer], delimiter=",")
            numpy.savetxt(f"nns/{nn}-bpieces-{layer}.csv",
                          self.b_pieces[layer], delimiter=",")
        numpy.savetxt(f"nns/{nn}-wpawns-beforelast.csv",
                      self.w_pawns[-1], delimiter=",")
        numpy.savetxt(f"nns/{nn}-wpieces-beforelast.csv",
                      self.w_pieces[-1], delimiter=",")
        numpy.savetxt(f"nns/{nn}-bpawns-beforelast.csv",
                      self.b_pawns[-1], delimiter=",")
        numpy.savetxt(f"nns/{nn}-bpieces-beforelast.csv",
                      self.b_pieces[-1], delimiter=",")
        numpy.savetxt(f"nns/{nn}-w-last.csv", self.w_last, delimiter=",")
        numpy.savetxt(f"nns/{nn}-b-last.csv", self.b_last, delimiter=",")

    def load_layers(self, nn: int) -> None:
        """
        Load layers with a NN ID from nns/.

        :param int nn: NN to load (nns/<nn>-....csv)
        :return: Nothing.
        :rtype: None
        """
        for layer in range(LAYERS):
            self.w_pawns.append(numpy.genfromtxt(
                f"nns/{nn}-wpawns-{layer}.csv", delimiter=","))
            self.b_pawns.append(numpy.genfromtxt(
                f"nns/{nn}-bpawns-{layer}.csv", delimiter=","))
            self.w_pieces.append(numpy.genfromtxt(
                f"nns/{nn}-wpieces-{layer}.csv", delimiter=","))
            self.b_pieces.append(numpy.genfromtxt(
                f"nns/{nn}-bpieces-{layer}.csv", delimiter=","))
        self.w_pawns.append(numpy.genfromtxt(
            f"nns/{nn}-wpawns-beforelast.csv", delimiter=","))
        self.b_pawns.append(numpy.genfromtxt(
            f"nns/{nn}-bpawns-beforelast.csv", delimiter=","))
        self.w_pieces.append(numpy.genfromtxt(
            f"nns/{nn}-wpieces-beforelast.csv", delimiter=","))
        self.b_pieces.append(numpy.genfromtxt(
            f"nns/{nn}-bpieces-beforelast.csv", delimiter=","))
        self.w_pawns[-1] = self.w_pawns[-1].reshape(16, 1)
        self.b_pawns[-1] = self.b_pawns[-1].reshape(16, 1)
        self.w_pieces[-1] = self.w_pieces[-1].reshape(16, 1)
        self.b_pieces[-1] = self.b_pieces[-1].reshape(16, 1)
        self.w_last = numpy.genfromtxt(f"nns/{nn}-w-last.csv", delimiter=",")
        self.b_last = numpy.genfromtxt(f"nns/{nn}-b-last.csv", delimiter=",")
        self.w_last = self.w_last.reshape(1, 16)
        self.b_last = self.b_last.reshape(1, 1)

    def clear(self) -> None:
        """Clear layers."""
        zero: numpy.ndarray = numpy.zeros((1, 1))
        self.w_pawns.clear()
        self.w_pieces.clear()
        self.b_pawns.clear()
        self.b_pieces.clear()
        for layer in range(LAYERS + 1):
            self.w_pawns.append(zero)
            self.w_pieces.append(zero)
            self.b_pawns.append(zero)
            self.b_pieces.append(zero)
        self.w_last = zero
        self.b_last = zero

    def generate(self) -> None:
        """Generate new matrixes."""
        identity: numpy.ndarray = numpy.identity(16)
        zeros: numpy.ndarray = numpy.zeros((16, 16))
        column_ones: numpy.ndarray = numpy.ones((8, 1))
        column_zeros: numpy.ndarray = numpy.zeros((16, 1))
        column_negative_ones: numpy.ndarray = numpy.empty((8, 1))
        column_negative_ones.fill(-1)
        column_matrix: numpy.ndarray = numpy.block(
            [[column_ones], [column_negative_ones]])
        line_ones: numpy.ndarray = numpy.ones((1, 16))
        zero: numpy.ndarray = numpy.zeros((1, 1))
        self.clear()
        for layer in range(LAYERS):
            self.w_pawns[layer] = identity
            self.w_pieces[layer] = identity
            self.b_pawns[layer] = zeros
            self.b_pieces[layer] = zeros
        self.w_pawns[-1] = column_matrix
        self.w_pieces[-1] = column_matrix
        self.b_pawns[-1] = column_zeros
        self.b_pieces[-1] = column_zeros
        self.w_last: numpy.ndarray = line_ones
        self.b_last: numpy.ndarray = zero

    def __str__(self):
        """
        Implements str(self).

        :return: Neural Network #{id}
        :rtype: str
        """
        return f"Neural Network #{self.indice} (performance {self.result} / on good moves {self.perfs[0]} / on bad moves {self.perfs[1]})"

    def __repr__(self):
        """
        Implements repr(self).

        :return: <NeuralNetwork object #{id(self)}>
        :rtype: str
        """
        return f"<NeuralNetwork object #{self.indice} (performance {self.result} / on good moves {self.perfs[0]} / on bad moves {self.perfs[1]})>"


if __name__ == "__main__":
    test = NeuralNetwork()
    test.generate()
    print(test.check_move(chess.STARTING_FEN, "e2e4"))
    print(test.output_layer.data)
