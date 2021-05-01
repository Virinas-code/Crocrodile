#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Neural Network.

Base class for Crocrodile NN.
"""
import csv
import math
import numpy
import chess


class NeuralNetwork:
    """Base class for NN."""

    def __init__(self):
        """Initialize NN."""
        self.weight1 = self.csv_to_array("w1.csv")
        self.weight2 = self.csv_to_array("w2.csv")
        self.weight3 = self.csv_to_array("w3.csv")
        self.weight4 = self.csv_to_array("w4.csv")
        self.input_layer = numpy.zeros(74)
        self.hidden_layer_1 = numpy.zeros(74)
        self.hidden_layer_2 = numpy.zeros(74)
        self.hidden_layer_3 = numpy.zeros(74)
        self.output_layer = numpy.zeros(1)

    def output(self):
        """Return NN output."""
        if self.output_layer[0] > 0:
            return True
        return False

    def generate_inputs(self, board, move):
        """Generate inputs for move move in board."""
        board = chess.Board(board)
        pieces = board.piece_map()
        inputs = []
        inputs_values = {'': 0, 'P': 0.1, 'N': 0.2, 'B': 0.3, 'R': 0.5,
                         'Q': 0.6, 'K': 0.7, 'p': -0.1, 'n': -0.2, 'b': -0.3,
                         'r': -0.5, 'q': -0.6, 'k': -0.7}
        for square in range(64):
            if pieces.get(square, None):
                inputs.append(inputs_values.get(pieces[square].symbol(), 0))
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
        self.input_layer = numpy.array(inputs)
        self.input_layer = self.input_layer

    @staticmethod
    def csv_to_array(csv_path):
        """Read CSV file and return array."""
        result = []
        with open(csv_path) as file:
            reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                result.append(row)
        return numpy.array(result)

    def calculate(self):
        """Calculate NN result."""
        normalizer = numpy.vectorize(self.normalisation)
        self.hidden_layer_1 = self.input_layer @ self.weight1
        self.hidden_layer_1 = normalizer(self.hidden_layer_1)
        self.hidden_layer_2 = self.hidden_layer_1 @ self.weight2
        self.hidden_layer_2 = normalizer(self.hidden_layer_2)
        self.hidden_layer_3 = self.hidden_layer_2 @ self.weight3
        self.hidden_layer_3 = normalizer(self.hidden_layer_3)
        self.output_layer = self.hidden_layer_3 @ self.weight4
        self.output_layer = normalizer(self.output_layer)

    def check_move(self, board, move):
        """Generate inputs, calculate and return output."""
        self.generate_inputs(board, move)
        self.calculate()
        return self.output()

    def check_test(self):
        """Check NN on test dataset."""
        with open("my_engine/test_data_goodmoves.txt") as file:
            file_goodmoves = file.read()
            file.close()
        with open("my_engine/test_data_badmoves.txt") as file:
            file_badmoves = file.read()
            file.close()
        file_goodmoves = file_goodmoves.split("\n\n")
        file_badmoves = file_badmoves.split("\n\n")
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
        with open("my_engine/train_data_goodmoves.txt") as file:
            file_goodmoves = file.read()
            file.close()
        with open("my_engine/train_data_badmoves.txt") as file:
            file_badmoves = file.read()
            file.close()
        file_goodmoves = file_goodmoves.split("\n\n")
        file_badmoves = file_badmoves.split("\n\n")
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

    def train(self):
        """Train Neural Network."""
        max_iters = int(input("Maximum iterations : "))
        iters = 0
        success_objective = float(input("Success objective (in percents) : "))
        max_diff = float(input("Maximal difference between training and test" +
                               " success rates : "))
        success = self.check_train()
        diff = self.check_train() - self.check_test()
        precedent_difference = self.check_difference()
        mutation_rate = float(input("Mutation rate (in percents) : "))
        mutation_change = float(input("Mutation change : "))
        inverse_rate = 100 / mutation_rate
        while iters < max_iters and success_objective > success and diff < max_diff:
            iters += 1
            print("Training #" + str(iters))
            # Code here
            # Ready
            # DECIDE
            random_matrix1 = numpy.random.rand(74, 74) * (2 * mutation_change) - mutation_change
            random_matrix4 = numpy.random.rand(74, 1) * (2 * mutation_change) - mutation_change
            # x2 is zero
            new_weight1 = numpy.heaviside(numpy.random.rand(74, 74) * inverse_rate + (1 - inverse_rate), 0)
            new_weight2 = numpy.heaviside(numpy.random.rand(74, 74) * inverse_rate + (1 - inverse_rate), 0)
            new_weight3 = numpy.heaviside(numpy.random.rand(74, 74) * inverse_rate + (1 - inverse_rate), 0)
            new_weight4 = numpy.heaviside(numpy.random.rand(74, 1) * inverse_rate + (1 - inverse_rate), 0)
            self.weight1 = self.weight1 + random_matrix1 * new_weight1
            self.weight2 = self.weight2 + random_matrix1 * new_weight2
            self.weight3 = self.weight3 + random_matrix1 * new_weight3
            self.weight4 = self.weight4 + random_matrix4 * new_weight4
            next_success = self.check_train()
            print(f"{next_success}, {success}")
            if next_success < success or self.check_difference() > precedent_difference:
                print("Reseting")
                self.weight1 = self.weight1 - random_matrix1 * new_weight1
                self.weight2 = self.weight2 - random_matrix1 * new_weight2
                self.weight3 = self.weight3 - random_matrix1 * new_weight3
                self.weight4 = self.weight4 - random_matrix4 * new_weight4
            precedent_difference = self.check_difference()
            success = self.check_train()
            diff = success - self.check_test()
            print("New success rate :", success)
        self.save()
        print("Saved.")

    def check_always_same(self):
        """Check success rating on good moves and on bad moves."""
        with open("my_engine/train_data_goodmoves.txt") as file:
            file_goodmoves = file.read()
            file.close()
        with open("my_engine/train_data_badmoves.txt") as file:
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
        print("Success rate on good moves : {0}%".format(good_on_good_moves / len(file_goodmoves) * 100))
        print("Success rate on bad moves : {0}%".format(good_on_bad_moves / len(file_badmoves) * 100))

    def check_difference(self):
        """Check success rating on good moves and on bad moves and return it."""
        with open("my_engine/train_data_goodmoves.txt") as file:
            file_goodmoves = file.read()
            file.close()
        with open("my_engine/train_data_badmoves.txt") as file:
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
        return abs((good_on_good_moves / len(file_goodmoves) * 100) - (good_on_bad_moves / len(file_badmoves) * 100))

    @staticmethod
    def array_to_csv(array, csv_path):
        """Write array in csv_path CSV file."""
        array = list(array)
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for row in array:
                writer.writerow(row)
            file.close()
        return 0

    def save(self):
        """Save Neural Network."""
        self.array_to_csv(self.weight1, "w1.csv")
        self.array_to_csv(self.weight2, "w2.csv")
        self.array_to_csv(self.weight3, "w3.csv")
        self.array_to_csv(self.weight4, "w4.csv")

    @staticmethod
    def normalisation(value):
        """Sigmoide modified."""
        return (1 / (1 + math.exp(-value))) * 2 - 1
