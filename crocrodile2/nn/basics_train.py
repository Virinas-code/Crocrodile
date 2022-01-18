#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Training.

Back to basics.

:author: Virinas-code and ZeBox
"""
import csv
import datetime
import json
import random
import sys

import chess
import crocrodile
import crocrodile.nn
import matplotlib.pyplot as plt
import numpy
from crocrodile.cli import Progress

NoneType = type(None)

LAYERS_COUNT = 31
MAX_ITERS = 50000
SYMETRY_MATRIX = numpy.array(
    [
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    ]
)
LAYERS = 5


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
        self.neural_networks: list[crocrodile.nn.NeuralNetwork] = []

    @staticmethod
    def array_to_csv(array, csv_path):
        """Write array in csv_path CSV file."""
        array = list(array)
        with open(csv_path, "w", newline="") as file:
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
            open("basics_train.json", "w").write(json.dumps(self.config))
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
        return good_moves_list

    @staticmethod
    def generate_bad_moves(good_move_pos: str, good_moves_list, bad_moves_list):
        """
        Generate bad moves for position.

        :param good_move_pos: Good move in position (FEN + good move)
        :type good_move_pos: str
        """
        result = list()
        position = chess.Board(good_move_pos.split("\n")[0])
        for move in position.legal_moves:
            generated_position = position.fen() + "\n" + move.uci()
            if (
                generated_position not in good_moves_list
                and generated_position not in bad_moves_list
            ):
                result.append(generated_position)
        return result

    def generate(self) -> None:
        """
        Generate empty networks and save them.

        :return: None
        :rtype: None
        """
        self.neural_networks.clear()
        number = int(input("Population : "))
        open("nns/population.dat", "w").write(str(number))
        progress = Progress()
        progress.total = number
        progress.text = "Creating networks"
        for loop in range(number):
            progress.update(loop)
            self.neural_networks.append(crocrodile.nn.NeuralNetwork())
        progress.done()
        progress.text = "Generating empty matrixes"
        for loop in range(number):
            progress.update(loop)
            self.neural_networks[loop].generate()
        progress.done()
        self.config["iterations_done"] = 0
        open("basics_train.json", "w").write(json.dumps(self.config))
        self.save()

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
            self.neural_networks[loop].save(loop)
        progress.done()

    def load(self) -> None:
        """
        Load neural networks from nns/ folder.

        :return: None
        """
        self.neural_networks.clear()
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
            self.neural_networks[loop].load_layers(loop)
        progress.done()
        for indice in range(len(self.neural_networks)):
            self.neural_networks[indice].indice = indice

    def couple_pawns(
        self, matrix1: numpy.ndarray, matrix2: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Couple two pawn matrixes.

        :param numpy.ndarray matrix1: First matrix to couple.
        :param numpy.ndarray matrix2: Second matrix to couple.
        :return: A new matrix.
        :rtype: numpy.ndarray
        """
        mutation_change = self.config["mutation_change"]
        inverse_rate = 100 / self.config["mutation_rate"]
        choose_matrix: numpy.ndarray = numpy.zeros(matrix1.shape)
        direction: bool = bool(random.getrandbits(1))
        choose: bool = bool(random.getrandbits(1))
        try:
            len2 = len(choose_matrix[0])
        except TypeError:
            matrix1 = numpy.array([matrix1])
            matrix2 = numpy.array([matrix2])
            choose_matrix: numpy.ndarray = numpy.zeros(matrix1.shape)
            len2 = len(choose_matrix[0])
        except IndexError:
            matrix1 = numpy.array([[matrix1]])
            matrix2 = numpy.array([[matrix2]])
            choose_matrix: numpy.ndarray = numpy.zeros(matrix1.shape)
            len2 = len(choose_matrix[0])
        for line in range(len(choose_matrix)):
            for column in range(len2):
                if random.random() < 0.001:
                    choose: bool = not choose
                fill: int = int(choose)
                if direction:
                    choose_matrix[line][column] = fill
                else:
                    try:
                        choose_matrix[column][line] = fill
                    except IndexError:
                        choose_matrix[line][column] = fill
        result = (matrix1 * choose_matrix + matrix2 * (1 - choose_matrix)) + (
            numpy.random.rand(*matrix1.shape) * (2 * mutation_change) - mutation_change
        ) * numpy.heaviside(
            numpy.random.rand(*matrix1.shape) * inverse_rate + (1 - inverse_rate), 0
        )
        return 0.5 * (result + result @ SYMETRY_MATRIX)

    def couple_pieces(
        self, matrix1: numpy.ndarray, matrix2: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Couple two pieces matrixes.

        :param numpy.ndarray matrix1: First matrix to couple.
        :param numpy.ndarray matrix2: Second matrix to couple.
        :return: A new matrix.
        :rtype: numpy.ndarray
        """
        mutation_change = self.config["mutation_change"]
        inverse_rate = 100 / self.config["mutation_rate"]
        choose_matrix: numpy.ndarray = numpy.zeros(matrix1.shape)
        direction: bool = bool(random.getrandbits(1))
        choose: bool = bool(random.getrandbits(1))
        try:
            len2 = len(choose_matrix[0])
        except TypeError:
            matrix1 = numpy.array([matrix1])
            matrix2 = numpy.array([matrix2])
            choose_matrix: numpy.ndarray = numpy.zeros(matrix1.shape)
            len2 = len(choose_matrix[0])
        except IndexError:
            matrix1 = numpy.array([[matrix1]])
            matrix2 = numpy.array([[matrix2]])
            choose_matrix: numpy.ndarray = numpy.zeros(matrix1.shape)
            len2 = len(choose_matrix[0])
        for line in range(len(choose_matrix)):
            for column in range(len2):
                if random.random() < 0.001:
                    choose: bool = not choose
                fill: int = int(choose)
                if direction:
                    choose_matrix[line][column] = fill
                else:
                    try:
                        choose_matrix[column][line] = fill
                    except IndexError:
                        choose_matrix[line][column] = fill
        result = (matrix1 * choose_matrix + matrix2 * (1 - choose_matrix)) + (
            numpy.random.rand(*matrix1.shape) * (2 * mutation_change) - mutation_change
        ) * numpy.heaviside(
            numpy.random.rand(*matrix1.shape) * inverse_rate + (1 - inverse_rate), 0
        )
        return 0.25 * (
            result
            + result @ SYMETRY_MATRIX
            + SYMETRY_MATRIX @ result
            + SYMETRY_MATRIX @ result @ SYMETRY_MATRIX
        )

    def couple(self, matrix1: numpy.ndarray, matrix2: numpy.ndarray) -> numpy.ndarray:
        """
        Couple two matrixes.

        :param numpy.ndarray matrix1: First matrix to couple.
        :param numpy.ndarray matrix2: Second matrix to couple.
        :return: A new matrix.
        :rtype: numpy.ndarray
        """
        mutation_change = self.config["mutation_change"]
        inverse_rate = 100 / self.config["mutation_rate"]
        choose_matrix: numpy.ndarray = numpy.zeros(matrix1.shape)
        direction: bool = bool(random.getrandbits(1))
        choose: bool = bool(random.getrandbits(1))
        try:
            len2 = len(choose_matrix[0])
        except TypeError:
            matrix1 = numpy.array([matrix1])
            matrix2 = numpy.array([matrix2])
            choose_matrix: numpy.ndarray = numpy.zeros(matrix1.shape)
            len2 = len(choose_matrix[0])
        except IndexError:
            matrix1 = numpy.array([[matrix1]])
            matrix2 = numpy.array([[matrix2]])
            choose_matrix: numpy.ndarray = numpy.zeros(matrix1.shape)
            len2 = len(choose_matrix[0])
        for line in range(len(choose_matrix)):
            for column in range(len2):
                if random.random() < 0.001:
                    choose: bool = not choose
                fill: int = int(choose)
                if direction:
                    choose_matrix[line][column] = fill
                else:
                    try:
                        choose_matrix[column][line] = fill
                    except IndexError:
                        choose_matrix[line][column] = fill
        return (matrix1 * choose_matrix + matrix2 * (1 - choose_matrix)) + (
            numpy.random.rand(*matrix1.shape) * (2 * mutation_change) - mutation_change
        ) * numpy.heaviside(
            numpy.random.rand(*matrix1.shape) * inverse_rate + (1 - inverse_rate), 0
        )

    def couple_networks(self, worst_network: int, network1: int, network2: int) -> None:
        """
        Couple two networks.

        :param int network1: First network indice
        :param int network2: Second network indice.
        :return: Nothing.
        :rtype: None.
        """
        for layer_indice in range(LAYERS):
            self.neural_networks[worst_network].w_pawns[
                layer_indice
            ] = self.couple_pawns(
                self.neural_networks[network1].w_pawns[layer_indice],
                self.neural_networks[network2].w_pawns[layer_indice],
            )
            self.neural_networks[worst_network].b_pawns[
                layer_indice
            ] = self.couple_pawns(
                self.neural_networks[network1].b_pawns[layer_indice],
                self.neural_networks[network2].b_pawns[layer_indice],
            )
            self.neural_networks[worst_network].w_pieces[
                layer_indice
            ] = self.couple_pieces(
                self.neural_networks[network1].w_pieces[layer_indice],
                self.neural_networks[network2].w_pieces[layer_indice],
            )
            self.neural_networks[worst_network].b_pieces[
                layer_indice
            ] = self.couple_pieces(
                self.neural_networks[network1].b_pieces[layer_indice],
                self.neural_networks[network2].b_pieces[layer_indice],
            )
        self.neural_networks[worst_network].w_pawns[-1] = self.couple(
            self.neural_networks[network1].w_pawns[-1],
            self.neural_networks[network2].w_pawns[-1],
        )
        self.neural_networks[worst_network].b_pawns[-1] = self.couple(
            self.neural_networks[network1].b_pawns[-1],
            self.neural_networks[network2].b_pawns[-1],
        )
        self.neural_networks[worst_network].w_last = self.couple(
            self.neural_networks[network1].w_last, self.neural_networks[network2].w_last
        )
        self.neural_networks[worst_network].b_last = self.couple(
            self.neural_networks[network1].b_last, self.neural_networks[network2].b_last
        )

    def train(
        self,
        new_good_move: str,
        new_bad_moves: str,
        param_good_moves: list,
        param_bad_moves: list,
    ) -> float:
        """
        Train neural networks.

        :return: Mean performance at end.
        :rtype: float
        """
        perf_graph = []
        dist_graph = []
        best_graph = []
        worst_graph = []

        def sprint(value):
            centered = value.center(18)
            print("********** {0} **********".format(centered))

        inverse_rate = 100 / self.config["mutation_rate"]
        mutation_change = self.config["mutation_change"]
        sprint("Initialize")
        iters = 0
        tests_results = list()
        population = len(self.neural_networks)
        # First original testing.
        for loop in range(population):
            print(f"Testing networks... ({loop}/{population})", end="\r", flush=True)
            on_good_moves, on_bad_moves = self.neural_networks[
                loop
            ].test_full_multiprocesses(param_good_moves, param_bad_moves)
            success = (
                (
                    (on_good_moves / len(param_good_moves))
                    - (1 - (on_bad_moves / len(param_bad_moves)))
                )
                * (on_good_moves / len(param_good_moves))
                * 100
            )
            self.neural_networks[loop].result = success
            self.neural_networks[loop].perfs = (on_good_moves, on_bad_moves)
        print("Testing networks... Done.          ")
        # https://stackoverflow.com/questions/16225677/get-the-second-largest-number-in-a-list-in-linear-time
        while True:
            # input("Press Enter to continue...")
            iters += 1
            print(f"########## Session #{len(param_good_moves)} ##########")
            print(
                f"Bad moves: {len(param_bad_moves)} / Good moves: {len(param_good_moves)}"
            )
            sprint("Training #{0}".format(iters))
            print("Selecting best networks...", end=" ", flush=True)
            self.neural_networks: list[crocrodile.nn.NeuralNetwork] = sorted(
                self.neural_networks, key=lambda sub: sub.result, reverse=True
            )
            for indice in range(len(self.neural_networks)):
                self.neural_networks[indice].indice = indice
            maxis_indices = [0, 1, 2, 3]
            minis_indices = [
                population - 1,
                population - 2,
                population - 3,
                population - 4,
            ]
            print("Done.")
            print(
                f"Worst networks : {', '.join(repr(nn) for nn in self.neural_networks[-4:])}"
            )
            print(
                f"Best networks : {', '.join(repr(nn) for nn in self.neural_networks[:4])}"
            )
            for network_indice in range(1):
                print(
                    f"Coupling network #{network_indice + 1}... (selecting second network)",
                    end="\r",
                    flush=True,
                )
                rand = random.randint(2, population - 3)
                second_network = rand
                # print(f"Coupling network {self.neural_networks[maxis_indices[network_indice]]} with {self.neural_networks[second_network]} to {self.neural_networks[minis_indices[network_indice]]}")
                print(
                    f"Coupling network #{network_indice + 1}... (generating coupling matrixes)",
                    end="\r",
                    flush=True,
                )
                self.couple_networks(
                    minis_indices[network_indice],
                    second_network,
                    maxis_indices[network_indice],
                )
                print(
                    f"Coupling network #{network_indice + 1}... (testing)                    ",
                    end="\r",
                    flush=True,
                )
                on_good_moves, on_bad_moves = self.neural_networks[
                    minis_indices[network_indice]
                ].test_full(param_good_moves, param_bad_moves)
                success = (
                    (
                        (on_good_moves / len(param_good_moves))
                        - (1 - (on_bad_moves / len(param_bad_moves)))
                    )
                    * (on_good_moves / len(param_good_moves))
                    * 100
                )
                self.neural_networks[loop].result = success
                self.neural_networks[loop].perfs = (on_good_moves, on_bad_moves)
                # print(f"New result: {self.neural_networks[minis_indices[network_indice]]}")
                print(
                    f"Coupling network #{network_indice + 1}... Done.   ",
                    end="\r",
                    flush=True,
                )
            print("Coupling networks... Done.                           ")
            perf_sum = 0
            for loop in range(population):
                perf_sum += self.neural_networks[loop].result
            print(f"Mean performance : {(perf_sum / population)}")
            self.neural_networks: list[crocrodile.nn.NeuralNetwork] = sorted(
                self.neural_networks, key=lambda sub: sub.result, reverse=True
            )
            for indice in range(len(self.neural_networks)):
                self.neural_networks[indice].indice = indice
            print(f"perf: {self.neural_networks[4].perfs}")
            # patch-003
            print(f"gdmvs: {self.neural_networks[4].perfs[0] / len(param_good_moves)}")
            print(
                f"bdmvs: {(self.neural_networks[4].perfs[1] / len(param_bad_moves)) * 100}"
            )  # patch-003
            if (
                self.neural_networks[4].perfs[0] / len(param_good_moves) >= 1
                and (self.neural_networks[4].perfs[1] / len(param_bad_moves)) * 100
                > self.config["min_bad_moves"]
            ):  # patch-003
                break  # :)
            if iters >= MAX_ITERS:
                break  # Prevent complex moves
            if iters % 10 == 0:
                self.save()
            best_graph.append(self.neural_networks[0].result)
            worst_graph.append(self.neural_networks[-2].result)
            perf_graph.append(perf_sum / population)
            dist_graph.append(
                self.neural_networks[0].result - self.neural_networks[-2].result
            )
            l = range(iters)
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].plot(l, perf_graph)
            axs[0, 0].set_title("Mean performance")
            axs[0, 1].plot(l, best_graph)
            axs[0, 1].set_title("Best network")
            axs[1, 0].plot(l, worst_graph)
            axs[1, 0].set_title("Worst network")
            axs[1, 1].plot(l, dist_graph)
            axs[1, 1].set_title("Difference")

            for ax in axs.flat:
                ax.set(xlabel="Iteration", ylabel="Performance")

            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()

            fig.savefig("nns/graph.png")
            del fig
            del axs
        print("Saving tests result...", end=" ", flush=True)
        saved_results = list()
        for element in tests_results:
            saved_results.append([float(element)])
        self.array_to_csv(saved_results, "nns/results.csv")
        print("Done.")
        perf_sum = 0
        for loop in range(population):
            perf_sum += self.neural_networks[loop].result
        best_graph.close()
        worst_graph.close()
        perf_graph.close()
        dist_graph.close()
        return perf_sum / population

    def main(self, argv):
        """Start training."""
        self.ask()
        performance_output_file = "nns/log/" + str(datetime.datetime.now()) + ".log"
        good_moves_file = self.config["good_moves"]
        good_moves_list = self.parse_good_moves(good_moves_file)
        good_moves_train = list()
        bad_moves_list = list()
        bad_moves_train: list = []
        if "-n" in argv or "--new-networks" in argv:
            self.generate()
            sys.exit(0)
        else:
            self.load()
        first_train = True
        for good_move in good_moves_list:
            good_moves_train.append(good_move)
            print(f"########## Session #{len(good_moves_train)} ##########")
            new_bad_moves = self.generate_bad_moves(
                good_move, good_moves_list, bad_moves_list
            )
            bad_moves_list.extend(new_bad_moves)
            bad_moves_train.extend(new_bad_moves)
            print(
                f"Bad moves: {len(bad_moves_train)} / Good moves: {len(good_moves_train)}"
            )
            print("Training...", end="\r")
            if len(good_moves_train) > self.config["iterations_done"]:
                if first_train:
                    random.shuffle(bad_moves_train)
                    while len(bad_moves_train) > self.config["max_bad_moves"]:
                        bad_moves_train.pop()
                    random.shuffle(good_moves_train)
                    while len(good_moves_train) > self.config["max_good_moves"]:
                        good_moves_train.pop()
                    """progress = Progress()
                    progress.text = "Testing networks"
                    progress.total = len(self.neural_networks)
                    for network_indice in range(len(self.neural_networks)):
                        progress.update(network_indice)
                        self.neural_networks[network_indice].test_full(good_moves_train, bad_moves_train)
                    progress.done()"""
                    first_train = False
                with open(performance_output_file, "a") as file:
                    file.write(
                        str(
                            self.train(
                                good_move,
                                new_bad_moves,
                                good_moves_train,
                                bad_moves_train,
                            )
                        )
                        + "\n"
                    )
                if len(good_moves_train) % 10 == 0:
                    self.save()
                self.config["iterations_done"] = len(good_moves_train)
                open("basics_train.json", "w").write(json.dumps(self.config))


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


if __name__ == "__main__":
    main(sys.argv)
