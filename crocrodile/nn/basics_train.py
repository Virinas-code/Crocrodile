#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Training.

Back to basics.

:author: Virinas-code and ZeBox
"""
import sys
import json
import csv
import random
import pickle
import chess
import numpy
import crocrodile
import crocrodile.nn
from crocrodile.cli import Progress

NoneType = type(None)


class BasicsTrain:
    """
    Basics train - class for training Crocrodile.

    :author: @ZeBox and Virinas-code
    """

    def __init__(self):
        """
        Initialize training.

        :param self: Current BasicsTrain object.
        """
        self.config: dict = json.loads(open("basics_train.json").read())
        self.neural_networks: list[crocrodile.nn.NeuralNetwork] = list()

    @staticmethod
    def array_to_csv(array, csv_path):
        """Write array in csv_path CSV file."""
        array = list(array)
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(array)
            file.close()
        return 0

    def ask(self) -> dict:
        """
        Ask for inputs.

        :param self: Current BasicsTrain object.
        :type self: BasicsTrain
        :return: Good moves file.
        :rtype: str
        """
        if "-i" in sys.argv or "--input" in sys.argv:
            good_moves_file: str = input("Good moves file : ")
            mutation_rate: float = float(input("Mutation rate : "))
            mutation_change: float = float(input("Mutation change : "))
            min_bad_moves: int = int(input("Minimum performance on bad moves : "))
            self.config["good_moves"] = good_moves_file
            self.config["mutation_rate"] = mutation_rate
            self.config["mutation_change"] = mutation_change
            self.config["min_bad_moves"] = min_bad_moves
            open("basics_train.json", 'w').write(json.dumps(self.config))
        return self.config

    @staticmethod
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

    @staticmethod
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

    def generate(self) -> None:
        """
        Generate random networks and save them.

        :return: None
        :rtype: None
        """
        number = int(input("Population : "))
        open("nns/population.dat", "w").write(str(number))
        progress = Progress()
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
        progress.total = number
        progress.text = "Generating random matrixes"
        for loop in range(number):
            progress.update(loop)
            tests_weight1.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_weight2.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_weight3.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_weight4.append(numpy.random.rand(1, 64) * 2 - 1)
            tests_weight5.append(numpy.random.rand(64, 1) * 2 - 1)
            tests_bias1.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_bias2.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_bias3.append(numpy.random.rand(64, 64) * 2 - 1)
            tests_bias4.append(numpy.random.rand(1, 64) * 2 - 1)
            tests_bias5.append(numpy.array(numpy.random.rand(1, 1) * 2 - 1))
        progress.done()
        progress.text = "Creating networks"
        for loop in range(number):
            progress.update(loop)
            self.neural_networks.append(crocrodile.nn.NeuralNetwork())
        progress.done()
        progress.text = "Saving random matrixes to networks"
        for loop in range(number):
            progress.update(loop)
            self.neural_networks[loop].weight1 = tests_weight1[loop]
            self.neural_networks[loop].weight2 = tests_weight2[loop]
            self.neural_networks[loop].weight3 = tests_weight3[loop]
            self.neural_networks[loop].weight4 = tests_weight4[loop]
            self.neural_networks[loop].weight5 = tests_weight5[loop]
            self.neural_networks[loop].b1 = tests_bias1[loop]
            self.neural_networks[loop].b2 = tests_bias2[loop]
            self.neural_networks[loop].b3 = tests_bias3[loop]
            self.neural_networks[loop].b4 = tests_bias4[loop]
            self.neural_networks[loop].b5 = tests_bias5[loop]
        print(self.neural_networks)
    def save(self) -> None:
        """
        Save neural networks to nns/ folder.

        :return: None
        """
        progress = Progress()
        progress.total = len(self.neural_networks)
        progress.text = "Saving networks"
        for loop in range(len(self.neural_networks)):
            progress.update(loop)
            numpy.savetxt(f"nns/{loop}-w1.csv", self.neural_networks[loop].weight1, delimiter=",")
            numpy.savetxt(f"nns/{loop}-w2.csv", self.neural_networks[loop].weight2, delimiter=",")
            numpy.savetxt(f"nns/{loop}-w3.csv", self.neural_networks[loop].weight3, delimiter=",")
            numpy.savetxt(f"nns/{loop}-w4.csv", self.neural_networks[loop].weight4, delimiter=",")
            numpy.savetxt(f"nns/{loop}-w5.csv", self.neural_networks[loop].weight5, delimiter=",")
            numpy.savetxt(f"nns/{loop}-b1.csv", self.neural_networks[loop].b1, delimiter=",")
            numpy.savetxt(f"nns/{loop}-b2.csv", self.neural_networks[loop].b2, delimiter=",")
            numpy.savetxt(f"nns/{loop}-b3.csv", self.neural_networks[loop].b3, delimiter=",")
            numpy.savetxt(f"nns/{loop}-b4.csv", self.neural_networks[loop].b4, delimiter=",")
            open(f"nns/{loop}-b5.csv", 'w').write(str(float(self.neural_networks[loop].b5)))  # patch-001
        progress.done()

    def load(self) -> None:
        """
        Load neural networks from nns/ folder.

        :return: None
        """
        progress = Progress()
        progress.text = "Loading population"
        number = int(open("nns/population.dat", "r").read())
        progress.text = "Creating network objects"
        progress.total = number
        for loop in range(number):
            progress.update(loop)
            self.neural_networks.append(crocrodile.nn.NeuralNetwork())
        progress.done()
        progress.text = "Loading networks"
        for loop in range(number):
            progress.update(loop)
            self.neural_networks[loop].weight1 = numpy.genfromtxt(f"nns/{loop}-w1.csv", delimiter=',')
            self.neural_networks[loop].weight2 = numpy.genfromtxt(f"nns/{loop}-w2.csv", delimiter=',')
            self.neural_networks[loop].weight3 = numpy.genfromtxt(f"nns/{loop}-w3.csv", delimiter=',')
            self.neural_networks[loop].weight4 = numpy.genfromtxt(f"nns/{loop}-w4.csv", delimiter=',')
            self.neural_networks[loop].weight5 = numpy.genfromtxt(f"nns/{loop}-w5.csv", delimiter=',').reshape(-1,1)
            self.neural_networks[loop].b1 = numpy.genfromtxt(f"nns/{loop}-b1.csv", delimiter=',')
            self.neural_networks[loop].b2 = numpy.genfromtxt(f"nns/{loop}-b2.csv", delimiter=',')
            self.neural_networks[loop].b3 = numpy.genfromtxt(f"nns/{loop}-b3.csv", delimiter=',')
            self.neural_networks[loop].b4 = numpy.genfromtxt(f"nns/{loop}-b4.csv", delimiter=',')
            self.neural_networks[loop].b5 = numpy.array(float(open(f"nns/{loop}-b5.csv").read()))  # patch-001
        progress.done()

    def train(self, param_good_moves: list, param_bad_moves: list) -> None:
        """
        Train neural networks.

        :return: None
        :rtype: None
        """
        def sprint(value):
            centered = value.center(18)
            print("********** {0} **********".format(centered))
        max_iters = 200
        max_success = 100
        inverse_rate = 100 / config["mutation_rate"]
        mutation_change = config["mutation_change"]
        sprint("Initialize")
        iters = 0
        tests_results = list()
        population = len(self.neural_networks)
        for loop in range(population):
            print(
                f"Testing networks... ({loop}/{population})", end="\r", flush=True)
            on_good_moves, on_bad_moves, good_moves, bad_moves = self.neural_networks[loop].test(param_good_moves, param_bad_moves)
            success = ((on_good_moves / good_moves)**2
                       * (on_bad_moves / bad_moves)) * 100
            self.neural_networks[loop].result = success
        print("Testing networks... Done.          ")
        good_moves_score = 0
        bad_moves_score = 0
        # https://stackoverflow.com/questions/16225677/get-the-second-largest-number-in-a-list-in-linear-time
        while good_moves_score < 100 or bad_moves_score < config["min_bad_moves"]:
            iters += 1
            sprint("Training #{0}".format(iters))
            print("Selecting best networks...", end=" ", flush=True)
            print(sorted(self.neural_networks, key=lambda sub: sub.result))
            print(f"Worst networks : {', '.join(liste)}")
            liste = []
            for count in range(4):
                liste.append(
                    "#" + str(maxis_indices[count]) + " (" + str(maxis[count]) + ")")
            print(f"Best networks : {', '.join(liste)}")
            for network_indice in range(4):
                print(
                    f"Coupling network #{network_indice + 1}... (selecting second network)", end="\r", flush=True)
                cont = True
                while cont:
                    cont = False
                    rand = random.randint(0, population - 1)
                    if rand in minis_indices or rand in maxis_indices:
                        cont = True
                second_network = rand
                print(
                    f"Coupling network #{network_indice + 1}... (generating coupling matrixes)", end="\r", flush=True)
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
                    f"Coupling network #{network_indice + 1}... (coupling)                    ", end="\r", flush=True)
                tests_weight1[minis_indices[network_indice]] = tests_weight1[maxis_indices[network_indice]
                                                                             ] * choose_w1 + tests_weight1[second_network] * (1 - choose_w1)
                tests_weight2[minis_indices[network_indice]] = tests_weight2[maxis_indices[network_indice]
                                                                             ] * choose_w2 + tests_weight2[second_network] * (1 - choose_w2)
                tests_weight3[minis_indices[network_indice]] = tests_weight3[maxis_indices[network_indice]
                                                                             ] * choose_w3 + tests_weight3[second_network] * (1 - choose_w3)
                tests_weight4[minis_indices[network_indice]] = tests_weight4[maxis_indices[network_indice]
                                                                             ] * choose_w4 + tests_weight4[second_network] * (1 - choose_w4)
                tests_weight5[minis_indices[network_indice]] = tests_weight5[maxis_indices[network_indice]
                                                                             ] * choose_w5 + tests_weight5[second_network] * (1 - choose_w5)
                tests_bias1[minis_indices[network_indice]] = tests_bias1[maxis_indices[network_indice]
                                                                         ] * choose_b1 + tests_bias1[second_network] * (1 - choose_b1)
                tests_bias2[minis_indices[network_indice]] = tests_bias2[maxis_indices[network_indice]
                                                                         ] * choose_b2 + tests_bias2[second_network] * (1 - choose_b2)
                tests_bias3[minis_indices[network_indice]] = tests_bias3[maxis_indices[network_indice]
                                                                         ] * choose_b3 + tests_bias3[second_network] * (1 - choose_b3)
                tests_bias4[minis_indices[network_indice]] = tests_bias4[maxis_indices[network_indice]
                                                                         ] * choose_b4 + tests_bias4[second_network] * (1 - choose_b4)
                tests_bias5[minis_indices[network_indice]] = tests_bias5[maxis_indices[network_indice]
                                                                         ] * choose_b5 + tests_bias5[second_network] * (1 - choose_b5)
                tests_weight1[minis_indices[network_indice]] += ((numpy.random.rand(64, 64) * (
                    2 * mutation_change) - mutation_change)) * numpy.heaviside(numpy.random.rand(64, 64) * inverse_rate + (1 - inverse_rate), 0)
                tests_weight2[minis_indices[network_indice]] += ((numpy.random.rand(64, 64) * (
                    2 * mutation_change) - mutation_change)) * numpy.heaviside(numpy.random.rand(64, 64) * inverse_rate + (1 - inverse_rate), 0)
                tests_weight3[minis_indices[network_indice]] += ((numpy.random.rand(64, 64) * (
                    2 * mutation_change) - mutation_change)) * numpy.heaviside(numpy.random.rand(64, 64) * inverse_rate + (1 - inverse_rate), 0)
                tests_weight4[minis_indices[network_indice]] += ((numpy.random.rand(1, 64) * (
                    2 * mutation_change) - mutation_change)) * numpy.heaviside(numpy.random.rand(1, 64) * inverse_rate + (1 - inverse_rate), 0)
                tests_weight5[minis_indices[network_indice]] += ((numpy.random.rand(64, 1) * (
                    2 * mutation_change) - mutation_change)) * numpy.heaviside(numpy.random.rand(64, 1) * inverse_rate + (1 - inverse_rate), 0)
                tests_bias1[minis_indices[network_indice]] += ((numpy.random.rand(64, 64) * (
                    2 * mutation_change) - mutation_change)) * numpy.heaviside(numpy.random.rand(64, 64) * inverse_rate + (1 - inverse_rate), 0)
                tests_bias2[minis_indices[network_indice]] += ((numpy.random.rand(64, 64) * (
                    2 * mutation_change) - mutation_change)) * numpy.heaviside(numpy.random.rand(64, 64) * inverse_rate + (1 - inverse_rate), 0)
                tests_bias3[minis_indices[network_indice]] += ((numpy.random.rand(64, 64) * (
                    2 * mutation_change) - mutation_change)) * numpy.heaviside(numpy.random.rand(64, 64) * inverse_rate + (1 - inverse_rate), 0)
                tests_bias4[minis_indices[network_indice]] += ((numpy.random.rand(1, 64) * (
                    2 * mutation_change) - mutation_change)) * numpy.heaviside(numpy.random.rand(1, 64) * inverse_rate + (1 - inverse_rate), 0)
                tests_bias5[minis_indices[network_indice]] += ((numpy.random.rand(1, 1) * (
                    2 * mutation_change) - mutation_change)) * numpy.heaviside(numpy.random.rand(1, 1) * inverse_rate + (1 - inverse_rate), 0)
                print(
                    f"Coupling network #{network_indice + 1}... (testing)                    ", end="\r", flush=True)
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
                on_good_moves, on_bad_moves, good_moves, bad_moves = self.masters_check_train()
                success = ((on_good_moves / good_moves)**2
                           * (on_bad_moves / bad_moves)) * 100
                tests_results[minis_indices[network_indice]] = success
                print(
                    f"Coupling network #{network_indice + 1}... Done.   ", end="\r", flush=True)
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

    def main(self, argv):
        """Start training."""
        self.ask()
        good_moves_file = self.config["good_moves"]
        good_moves_list = self.parse_good_moves(good_moves_file)
        good_moves_train = list()
        bad_moves_list = list()
        if "-n" in argv or "--new-networks" in argv:
            self.generate()
        else:
            self.load()
        for good_move in good_moves_list:
            good_moves_train.append(good_move)
            print(f"########## Training #{len(good_moves_train)} ##########")
            bad_moves_list.extend(self.generate_bad_moves(
                good_move, good_moves_list, bad_moves_list))
            print(
                f"Bad moves: {len(bad_moves_list)} / Good moves: {len(good_moves_train)}")
            print("Training...", end="\r")


def main(argv):
    """
    Start function called in init.

    :param argv: sys.argv
    :type argv: list
    :return: None
    :rtype: None
    """
    trainer = BasicsTrain()  # Create trainer object
    trainer.main(argv)  # Start trainer


if __name__ == '__main__':
    main(sys.argv)
