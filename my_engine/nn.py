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
        self.weight5 = self.csv_to_array("w5.csv")
        self.weight6 = self.csv_to_array("w6.csv")
        self.weight7 = self.csv_to_array("w7.csv")
        self.weight8 = self.csv_to_array("w8.csv")
        self.weight9 = self.csv_to_array("w9.csv")
        self.input_layer = numpy.zeros(813)
        self.hidden_layer_1 = numpy.zeros(1024)
        self.hidden_layer_2 = numpy.zeros(1024)
        self.hidden_layer_3 = numpy.zeros(1024)
        self.hidden_layer_4 = numpy.zeros(1024)
        self.hidden_layer_5 = numpy.zeros(1024)
        self.hidden_layer_6 = numpy.zeros(1024)
        self.hidden_layer_7 = numpy.zeros(1024)
        self.hidden_layer_8 = numpy.zeros(1024)
        self.output_layer = numpy.zeros(1)
        self.test_good = "my_engine/test_data_goodmoves.txt"
        self.test_bad = "my_engine/test_data_badmoves.txt"
        self.train_good = "my_engine/train_data_goodmoves.txt"
        self.train_bad = "my_engine/train_data_badmoves.txt"

    def output(self):
        """Return NN output."""
        if self.output_layer[0] > 0.5:
            return True
        return False

    def generate_inputs(self, board, move):
        """Generate inputs for move move in board."""
        board = chess.Board(board)
        if board.turn == chess.BLACK:
            board = board.mirror()
        pieces = board.piece_map()
        inputs = []
        inputs_values = {'': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         'P': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         'N': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         'B': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         'R': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         'Q': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         'K': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         'p': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         'n': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         'b': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         'r': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         'q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         'k': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
        for square in range(64):
            if pieces.get(square, None):
                inputs.extend(inputs_values[pieces[square].symbol()])
            else:
                inputs.extend(inputs_values[""])
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
        cols = [0, 0, 0, 0, 0, 0, 0, 0]
        if board.has_legal_en_passant():
            cols[chess.square_file(board.ep_square)] = 1
        inputs.extend(cols)
        move = chess.Move.from_uci(move)
        from_square = move.from_square
        cols = [0, 0, 0, 0, 0, 0, 0, 0]
        cols[chess.square_file(from_square)] = 1
        inputs.extend(cols)
        cols = [0, 0, 0, 0, 0, 0, 0, 0]
        cols[chess.square_rank(from_square)] = 1
        inputs.extend(cols)
        to_square = move.to_square
        cols = [0, 0, 0, 0, 0, 0, 0, 0]
        cols[chess.square_file(to_square)] = 1
        inputs.extend(cols)
        cols = [0, 0, 0, 0, 0, 0, 0, 0]
        cols[chess.square_rank(to_square)] = 1
        inputs.extend(cols)
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
        self.hidden_layer_4 = self.hidden_layer_3 @ self.weight4
        self.hidden_layer_4 = normalizer(self.hidden_layer_4)
        self.hidden_layer_5 = self.hidden_layer_4 @ self.weight5
        self.hidden_layer_5 = normalizer(self.hidden_layer_5)
        self.hidden_layer_6 = self.hidden_layer_5 @ self.weight6
        self.hidden_layer_6 = normalizer(self.hidden_layer_6)
        self.hidden_layer_7 = self.hidden_layer_5 @ self.weight7
        self.hidden_layer_7 = normalizer(self.hidden_layer_7)
        self.hidden_layer_8 = self.hidden_layer_5 @ self.weight8
        self.hidden_layer_8 = normalizer(self.hidden_layer_8)
        self.output_layer = self.hidden_layer_8 @ self.weight9
        self.output_layer = normalizer(self.output_layer)

    def check_move(self, board, move):
        """Generate inputs, calculate and return output."""
        self.generate_inputs(board, move)
        self.calculate()
        return self.output()

    def check_test(self):
        """Check NN on test dataset."""
        with open(self.test_good) as file:
            file_goodmoves = file.read()
            file.close()
        with open(self.test_bad) as file:
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
        return correct_on_good_moves, correct_on_bad_moves, len(file_goodmoves), len(file_badmoves)

    def train(self):
        """Train Neural Network."""
        self.change_files()
        max_iters = int(input("Maximum iterations : "))
        iters = 0
        success_objective = float(input("Success objective (in percents) : "))
        max_diff = float(input("Maximal difference between training and test" +
                               " success rates : "))
        on_good_moves, on_bad_moves, good_moves, bad_moves = self.check_train()
        old_on_good_moves, old_on_bad_moves, old_good_moves, old_bad_moves = on_good_moves, on_bad_moves, good_moves, bad_moves
        success = (on_good_moves + on_bad_moves) / (good_moves + bad_moves) * 100
        diff = success - self.check_test()
        precedent_difference = abs(((on_good_moves / good_moves) - (on_bad_moves / bad_moves)) * 100)
        mutation_rate = float(input("Mutation rate (in percents) : "))
        mutation_change = float(input("Mutation change : "))
        inverse_rate = 100 / mutation_rate
        print(f"Success : {success} / Diff : {diff} / Precedent difference : {precedent_difference}")
        while iters < max_iters and success_objective > success and diff < max_diff:
            iters += 1
            print("Training #" + str(iters))
            # Code here
            # Ready
            # DECIDE
            random_matrix1 = numpy.random.rand(813, 1024) * (2 * mutation_change) - mutation_change
            random_matrix2 = numpy.random.rand(1024, 1024) * (2 * mutation_change) - mutation_change
            random_matrix4 = numpy.random.rand(1024, 1) * (2 * mutation_change) - mutation_change
            # x2 is zero
            new_weight1 = numpy.heaviside(numpy.random.rand(813, 1024) * inverse_rate + (1 - inverse_rate), 0)
            new_weight2 = numpy.heaviside(numpy.random.rand(1024, 1024) * inverse_rate + (1 - inverse_rate), 0)
            new_weight3 = numpy.heaviside(numpy.random.rand(1024, 1024) * inverse_rate + (1 - inverse_rate), 0)
            new_weight4 = numpy.heaviside(numpy.random.rand(1024, 1024) * inverse_rate + (1 - inverse_rate), 0)
            new_weight5 = numpy.heaviside(numpy.random.rand(1024, 1024) * inverse_rate + (1 - inverse_rate), 0)
            new_weight6 = numpy.heaviside(numpy.random.rand(1024, 1024) * inverse_rate + (1 - inverse_rate), 0)
            new_weight7 = numpy.heaviside(numpy.random.rand(1024, 1024) * inverse_rate + (1 - inverse_rate), 0)
            new_weight8 = numpy.heaviside(numpy.random.rand(1024, 1024) * inverse_rate + (1 - inverse_rate), 0)
            new_weight9 = numpy.heaviside(numpy.random.rand(1024, 1) * inverse_rate + (1 - inverse_rate), 0)
            self.weight1 = self.weight1 + random_matrix1 * new_weight1
            self.weight2 = self.weight2 + random_matrix2 * new_weight2
            self.weight3 = self.weight3 + random_matrix2 * new_weight3
            self.weight4 = self.weight4 + random_matrix2 * new_weight4
            self.weight5 = self.weight5 + random_matrix2 * new_weight5
            self.weight6 = self.weight6 + random_matrix2 * new_weight6
            self.weight7 = self.weight7 + random_matrix2 * new_weight7
            self.weight8 = self.weight8 + random_matrix2 * new_weight8
            self.weight9 = self.weight9 + random_matrix4 * new_weight9
            on_good_moves, on_bad_moves, good_moves, bad_moves = self.check_train()
            next_success = (on_good_moves + on_bad_moves) / (good_moves + bad_moves) * 100
            print("Test success rate :", next_success, "(on good moves :", (on_good_moves / good_moves) * 100, "% / on bad moves :", (on_bad_moves / bad_moves) * 100, "% )")
            difference = abs(((on_good_moves / good_moves) - (on_bad_moves / bad_moves)) * 100)
            if next_success < success - 0.5 * (difference - precedent_difference) or difference > precedent_difference:
                print("Reseting")
                self.weight1 = self.weight1 - random_matrix1 * new_weight1
                self.weight2 = self.weight2 - random_matrix2 * new_weight2
                self.weight3 = self.weight3 - random_matrix2 * new_weight3
                self.weight4 = self.weight4 - random_matrix2 * new_weight4
                self.weight5 = self.weight5 - random_matrix2 * new_weight5
                self.weight6 = self.weight6 - random_matrix2 * new_weight6
                self.weight7 = self.weight7 - random_matrix2 * new_weight7
                self.weight8 = self.weight8 - random_matrix2 * new_weight8
                self.weight9 = self.weight9 - random_matrix4 * new_weight9
                on_good_moves, on_bad_moves, good_moves, bad_moves = old_on_good_moves, old_on_bad_moves, old_good_moves, old_bad_moves
            else:
                diff = success - self.check_test()  # Check test take some time, but it's essential not to overtrain
            old_on_good_moves, old_on_bad_moves, old_good_moves, old_bad_moves = on_good_moves, on_bad_moves, good_moves, bad_moves
            success = (on_good_moves + on_bad_moves) / (good_moves + bad_moves) * 100
            precedent_difference = abs(((on_good_moves / good_moves) - (on_bad_moves / bad_moves)) * 100)
            print("New success rate :", success, "(on good moves :", (on_good_moves / good_moves) * 100, "% / on bad moves :", (on_bad_moves / bad_moves) * 100, "% )")
        self.save()
        print("Saved.")

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
        print("Success rate on good moves : {0}%".format(good_on_good_moves / len(file_goodmoves) * 100))
        print("Success rate on bad moves : {0}%".format(good_on_bad_moves / len(file_badmoves) * 100))

    def check_difference(self):
        """Check success rating on good moves and on bad moves and return it."""
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
        self.array_to_csv(self.weight5, "w5.csv")
        self.array_to_csv(self.weight6, "w6.csv")
        self.array_to_csv(self.weight7, "w7.csv")

    @staticmethod
    def normalisation(value):
        """Sigmoide."""
        return 1 / (1 + math.exp(-value))

    def change_files(self):
        """Change files locations."""
        self.train_good = input("Good moves training file : ")
        self.train_bad = input("Bad moves training file : ")
        self.test_good = input("Good moves test file : ")
        self.test_bad = input("Bad moves test file : ")


if __name__ == '__main__':
    test = NeuralNetwork()
