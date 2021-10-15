#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test NN calculation time.

Using unittest.
"""
import time
import unittest

import chess
import numpy
import crocrodile.nn

LAYERS_COUNT = 32


class TestNNCalcTime(unittest.TestCase):
    def test_nn_build(self):
        start = time.time()
        nn = crocrodile.nn.NeuralNetwork()
        end = time.time()
        print(f" (time: {end - start})")

    def test_nn_load(self):
        nn = crocrodile.nn.NeuralNetwork()
        start = time.time()
        for layer in range(LAYERS_COUNT):
            nn.layers.append(
                numpy.genfromtxt(f"nns/0-w{layer}.csv", delimiter=","))
            nn.bias.append(
                numpy.genfromtxt(f"nns/0-b{layer}.csv", delimiter=","))
        nn.layers[-1] = nn.layers[-1].reshape(
            1, nn.layers[-1].size
        )
        nn.bias[-1] = nn.bias[-1].reshape(
            1, nn.bias[-1].size
        )
        nn.last_layer = numpy.genfromtxt("nns/0-wlast.csv",
                                                                 delimiter=",")
        nn.last_layer = nn.last_layer.reshape(
            nn.last_layer.size, 1)
        nn.last_bias = numpy.genfromtxt("nns/0-blast.csv",
                                                                delimiter=",")
        nn.last_bias = nn.last_bias.reshape(
            1, 1)
        end = time.time()
        print(f" (time: {end - start})")

    def test_calc_time(self):
        nn = crocrodile.nn.NeuralNetwork()
        for layer in range(LAYERS_COUNT):
            nn.layers.append(
                numpy.genfromtxt(f"nns/0-w{layer}.csv", delimiter=","))
            nn.bias.append(
                numpy.genfromtxt(f"nns/0-b{layer}.csv", delimiter=","))
        nn.layers[-1] = nn.layers[-1].reshape(
            1, nn.layers[-1].size
        )
        nn.bias[-1] = nn.bias[-1].reshape(
            1, nn.bias[-1].size
        )
        nn.last_layer = numpy.genfromtxt("nns/0-wlast.csv",
                                                                 delimiter=",")
        nn.last_layer = nn.last_layer.reshape(
            nn.last_layer.size, 1)
        nn.last_bias = numpy.genfromtxt("nns/0-blast.csv",
                                                                delimiter=",")
        nn.last_bias = nn.last_bias.reshape(
            1, 1)
        start = time.time()
        for i in range(100):
            nn.check_move(chess.STARTING_FEN, "e2e4")
        end = time.time()
        print(f" (time: {end - start})")
        self.assertLess(end - start, 0.1)  # add assertion here


if __name__ == '__main__':
    unittest.main()
