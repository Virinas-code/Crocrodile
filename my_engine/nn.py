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
        self.b1 = self.csv_to_array("b1.csv")
        self.b2 = self.csv_to_array("b2.csv")
        self.b3 = self.csv_to_array("b3.csv")
        self.b4 = self.csv_to_array("b4.csv")
        self.b5 = self.csv_to_array("b5.csv")
        self.cweight1 = self.csv_to_array("cw1.csv")
        self.cweight2 = self.csv_to_array("cw2.csv")
        self.cweight3 = self.csv_to_array("cw3.csv")
        self.cweight4 = self.csv_to_array("cw4.csv")
        self.cweight5 = self.csv_to_array("cw5.csv")
        self.cb1 = self.csv_to_array("cb1.csv")
        self.cb2 = self.csv_to_array("cb2.csv")
        self.cb3 = self.csv_to_array("cb3.csv")
        self.cb4 = self.csv_to_array("cb4.csv")
        self.cb5 = self.csv_to_array("cb5.csv")
        self.pre_input_layer = numpy.zeros(768)
        self.input_layer = numpy.zeros(64)
        self.hidden_layer_1 = numpy.zeros(64)
        self.hidden_layer_2 = numpy.zeros(64)
        self.hidden_layer_3 = numpy.zeros(64)
        self.hidden_layer_4 = numpy.zeros(1)
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
        self.pre_input_layer = numpy.array(inputs)
        # Generate piece types inputs.self.hidden_layer_9
        white_pawns = list()
        for index in range(0, 768, 12):
            white_pawns.append(self.pre_input_layer[index])
        white_knights = list()
        for index in range(1, 769, 12):
            white_knights.append(self.pre_input_layer[index])
        white_bishops = list()
        for index in range(2, 770, 12):
            white_bishops.append(self.pre_input_layer[index])
        white_rooks = list()
        for index in range(3, 771, 12):
            white_rooks.append(self.pre_input_layer[index])
        white_queens = list()
        for index in range(4, 772, 12):
            white_queens.append(self.pre_input_layer[index])
        white_king = list()
        for index in range(5, 773, 12):
            white_king.append(self.pre_input_layer[index])
        black_pawns = list()
        for index in range(6, 774, 12):
            black_pawns.append(self.pre_input_layer[index])
        black_knights = list()
        for index in range(7, 769, 12):
            black_knights.append(self.pre_input_layer[index])
        black_bishops = list()
        for index in range(8, 770, 12):
            black_bishops.append(self.pre_input_layer[index])
        black_rooks = list()
        for index in range(9, 771, 12):
            black_rooks.append(self.pre_input_layer[index])
        black_queens = list()
        for index in range(10, 772, 12):
            black_queens.append(self.pre_input_layer[index])
        black_king = list()
        for index in range(11, 773, 12):
            black_king.append(self.pre_input_layer[index])
        result = (white_pawns, white_knights, white_bishops, white_rooks,
                  white_queens, white_king, black_pawns, black_knights,
                  black_bishops, black_rooks, black_queens, black_king)
        future = list(result)
        self.input_layer = []
        self.input_layer.extend(future)
        self.input_layer.append([0]*64)
        inputs = []
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
        self.input_layer.append(inputs + [0] * 60)
        inputs = []
        cols = [0, 0, 0, 0, 0, 0, 0, 0]
        if board.has_legal_en_passant():
            cols[chess.square_file(board.ep_square)] = 1
        inputs.extend(cols)
        self.input_layer.append(inputs + [0] * 56)
        inputs = []
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
        self.input_layer.append(inputs + [0] * 32)
        self.input_layer.extend([[0] * 64] * 48)
        self.input_layer = numpy.array(self.input_layer)

    @staticmethod
    def csv_to_array(csv_path):
        """Read CSV file and return array."""
        result = list()
        with open(csv_path) as file:
            reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                result.append(row)
        return numpy.array(result)

    def calculate(self):
        """Calculate NN result."""
        normalizer = numpy.vectorize(self.normalisation)
        self.hidden_layer_1 = self.input_layer @ self.weight1 + self.b1
        self.hidden_layer_1 = normalizer(self.hidden_layer_1)
        self.hidden_layer_2 = self.hidden_layer_1 @ self.weight2 + self.b2
        self.hidden_layer_2 = normalizer(self.hidden_layer_2)
        self.hidden_layer_3 = self.hidden_layer_2 @ self.weight3 + self.b3
        self.hidden_layer_3 = normalizer(self.hidden_layer_3)
        self.hidden_layer_4 = self.weight4 @ self.hidden_layer_3 + self.b4
        self.hidden_layer_4 = normalizer(self.hidden_layer_4)
        self.output_layer = self.hidden_layer_4 @ self.weight5 + self.b5
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
            random_matrix1 = numpy.random.rand(64, 64) * (2 * mutation_change) - mutation_change
            random_matrix4 = numpy.random.rand(1, 64) * (2 * mutation_change) - mutation_change
            random_matrix5 = numpy.random.rand(64, 1) * (2 * mutation_change) - mutation_change
            random_matrixb5 = numpy.random.rand(1, 1) * (2 * mutation_change) - mutation_change
            rand1 = numpy.random.rand(64, 64) * inverse_rate + (1 - inverse_rate)
            rand2 = numpy.random.rand(64, 64) * inverse_rate + (1 - inverse_rate)
            rand3 = numpy.random.rand(64, 64) * inverse_rate + (1 - inverse_rate)
            rand4 = numpy.random.rand(1, 64) * inverse_rate + (1 - inverse_rate)
            rand5 = numpy.random.rand(64, 1) * inverse_rate + (1 - inverse_rate)
            randb1 = numpy.random.rand(64, 64) * inverse_rate + (1 - inverse_rate)
            randb2 = numpy.random.rand(64, 64) * inverse_rate + (1 - inverse_rate)
            randb3 = numpy.random.rand(64, 64) * inverse_rate + (1 - inverse_rate)
            randb4 = numpy.random.rand(1, 64) * inverse_rate + (1 - inverse_rate)
            randb5 = numpy.random.rand(1, 1) * inverse_rate + (1 - inverse_rate)
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
            next_success = (on_good_moves + on_bad_moves) / (good_moves + bad_moves) * 100
            print("Test success rate :", next_success, "(on good moves :", (on_good_moves / good_moves) * 100, "% / on bad moves :", (on_bad_moves / bad_moves) * 100, "% )")
            difference = abs(((on_good_moves / good_moves) - (on_bad_moves / bad_moves)) * 100)
            if next_success < success - 0.5 * (difference - precedent_difference) or difference > precedent_difference:
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
                self.cweight1 = self.cweight1 - 0.05 * numpy.heaviside(rand1, 0) * self.cweight1
                self.cweight2 = self.cweight2 - 0.05 * numpy.heaviside(rand2, 0) * self.cweight2
                self.cweight3 = self.cweight3 - 0.05 * numpy.heaviside(rand3, 0) * self.cweight3
                self.cweight4 = self.cweight4 - 0.05 * numpy.heaviside(rand4, 0) * self.cweight4
                self.cweight5 = self.cweight5 - 0.05 * numpy.heaviside(rand5, 0) * self.cweight5
                self.cb1 = self.cb1 - 0.05 * numpy.heaviside(randb1, 0) * self.cb1
                self.cb2 = self.cb2 - 0.05 * numpy.heaviside(randb2, 0) * self.cb2
                self.cb3 = self.cb3 - 0.05 * numpy.heaviside(randb3, 0) * self.cb3
                self.cb4 = self.cb4 - 0.05 * numpy.heaviside(randb4, 0) * self.cb4
                self.cb5 = self.cb5 - 0.05 * numpy.heaviside(randb5, 0) * self.cb5
                on_good_moves, on_bad_moves, good_moves, bad_moves = old_on_good_moves, old_on_bad_moves, old_good_moves, old_bad_moves
            else:
                diff = success - self.check_test()  # Check test take some time, but it's essential not to overtrain
                # Nouvelle matrice consolidation = Normalisation(Ancienne + 0.05 * heaviside(matrice aléatorie, 0) * ancienne)
                self.cweight1 = normalizer(self.cweight1 + 0.05 * numpy.heaviside(rand1, 0) * self.cweight1)
                self.cweight2 = normalizer(self.cweight2 + 0.05 * numpy.heaviside(rand2, 0) * self.cweight2)
                self.cweight3 = normalizer(self.cweight3 + 0.05 * numpy.heaviside(rand3, 0) * self.cweight3)
                self.cweight4 = normalizer(self.cweight4 + 0.05 * numpy.heaviside(rand4, 0) * self.cweight4)
                self.cweight5 = normalizer(self.cweight5 + 0.05 * numpy.heaviside(rand5, 0) * self.cweight5)
                self.cb1 = normalizer(self.cb1 + 0.05 * numpy.heaviside(randb1, 0) * self.cb1)
                self.cb2 = normalizer(self.cb2 + 0.05 * numpy.heaviside(randb2, 0) * self.cb2)
                self.cb3 = normalizer(self.cb3 + 0.05 * numpy.heaviside(randb3, 0) * self.cb3)
                self.cb4 = normalizer(self.cb4 + 0.05 * numpy.heaviside(randb4, 0) * self.cb4)
                self.cb5 = normalizer(self.cb5 + 0.05 * numpy.heaviside(randb5, 0) * self.cb5)
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
    def normalisation(value):
        """Normalisation."""
        if value < 0:
            return 0
        elif value > 1:
            return 1
        return value

    def change_files(self):
        """Change files locations."""
        self.train_good = input("Good moves training file : ")
        self.train_bad = input("Bad moves training file : ")
        self.test_good = input("Good moves test file : ")
        self.test_bad = input("Bad moves test file : ")


if __name__ == '__main__':
    test = NeuralNetwork()
