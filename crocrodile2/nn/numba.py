#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile - NN Numba.

Use ``numba`` with Crocrodile NN.
"""
from numba import njit
import numpy

LAYERS = 5


@njit
def jit_calculate(input_layer: numpy.ndarray,
                  w_pawns: list[numpy.ndarray], w_pieces: list[numpy.ndarray],
                  b_pawns: list[numpy.ndarray], b_pieces: list[numpy.ndarray],
                  w_last: numpy.ndarray, b_last: numpy.ndarray) -> numpy.ndarray:
    """
    Calculate NN result by using :attr:NeuralNetwork.layers with multiple layers.

    :return: Output layer.
    :rtype: numpy.ndarray
    """
    hidden_layer: numpy.ndarray = input_layer
    mask_false = 16 * [16 * [False]]
    for layer_index in range(LAYERS):
        mask_pawns = numpy.ma.masked_outside(
            abs(hidden_layer), 0.08, 0.1699).mask
        mask_pieces = numpy.ma.masked_outside(
            abs(hidden_layer), 0.17, 1.16).mask
        hidden_layer = (numpy.ma.array(hidden_layer, mask=mask_pawns) @ w_pawns[layer_index]
                        + numpy.ma.array(b_pawns[layer_index], mask=mask_pawns)) + \
                       (numpy.ma.array(hidden_layer, mask=mask_pieces) @ w_pieces[layer_index]
                        + numpy.ma.array(b_pieces[layer_index], mask=mask_pieces))
        hidden_layer.mask = mask_false
    column_mask = 16 * [[False]]
    line_mask = [16 * [False]]
    hidden_layer = (hidden_layer @ numpy.ma.array(w_pawns[-1], mask=column_mask) + b_pawns[-1]) + \
                   (hidden_layer @
                    numpy.ma.array(w_pieces[-1], mask=column_mask) + b_pieces[-1])
    output_layer = numpy.ma.array(
        w_last, mask=line_mask) @ hidden_layer + b_last
    return output_layer

