#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NN Random.

Reset NN wa, wb and wc.
"""
import csv
import numpy


def array_to_csv(array, csv_path):
    """Python array to CSV file."""
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in array:
            writer.writerow(row)
        file.close()
    return 0


continuer = input("Do you REALLY want to reset all weights ? [y/N] ")

if continuer == "y":
    w1 = numpy.random.rand(64, 64) * 2 - 1
    w2 = numpy.random.rand(64, 64) * 2 - 1
    w3 = numpy.random.rand(64, 64) * 2 - 1
    w4 = numpy.random.rand(1, 64) * 2 - 1
    w5 = numpy.random.rand(64, 1) * 2 - 1
    b1 = numpy.random.rand(64, 64) * 2 - 1
    b2 = numpy.random.rand(64, 64) * 2 - 1
    b3 = numpy.random.rand(64, 64) * 2 - 1
    b4 = numpy.random.rand(1, 64) * 2 - 1
    b5 = numpy.random.rand(1, 1) * 2 - 1
    array_to_csv(w1, "w1.csv")
    array_to_csv(w2, "w2.csv")
    array_to_csv(w3, "w3.csv")
    array_to_csv(w4, "w4.csv")
    array_to_csv(w5, "w5.csv")
    array_to_csv(b1, "b1.csv")
    array_to_csv(b2, "b2.csv")
    array_to_csv(b3, "b3.csv")
    array_to_csv(b4, "b4.csv")
    array_to_csv(b5, "b5.csv")
    print("Done.")
