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


continuer = input("Do you REALLY want to reset wa, wb and wc ? [y/N] ")

if continuer == "y":
    w1 = numpy.random.rand(74, 74) * 2 - 1
    w2 = numpy.random.rand(74, 74) * 2 - 1
    w3 = numpy.random.rand(74, 74) * 2 - 1
    w4 = numpy.random.rand(74, 74) * 2 - 1
    w5 = numpy.random.rand(74, 74) * 2 - 1
    w6 = numpy.random.rand(74, 74) * 2 - 1
    w7 = numpy.random.rand(74, 1) * 2 - 1
    array_to_csv(w1, "w1.csv")
    array_to_csv(w2, "w2.csv")
    array_to_csv(w3, "w3.csv")
    array_to_csv(w4, "w4.csv")
    array_to_csv(w5, "w5.csv")
    array_to_csv(w6, "w6.csv")
    array_to_csv(w7, "w7.csv")
    print("Done.")
