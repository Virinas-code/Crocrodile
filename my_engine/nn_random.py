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


continuer = input("Do you REALLY want to reset w1, w2, w3, w4, w5, w6, w7, w8 and w9 ? [y/N] ")

if continuer == "y":
    w1 = numpy.random.rand(813, 1024)
    w2 = numpy.random.rand(1024, 1024)
    w3 = numpy.random.rand(1024, 1024)
    w4 = numpy.random.rand(1024, 1024)
    w5 = numpy.random.rand(1024, 1024)
    w6 = numpy.random.rand(1024, 1024)
    w7 = numpy.random.rand(1024, 1024)
    w8 = numpy.random.rand(1024, 1024)
    w9 = numpy.random.rand(1024, 1)
    array_to_csv(w1, "w1.csv")
    array_to_csv(w2, "w2.csv")
    array_to_csv(w3, "w3.csv")
    array_to_csv(w4, "w4.csv")
    array_to_csv(w5, "w5.csv")
    array_to_csv(w6, "w6.csv")
    array_to_csv(w7, "w7.csv")
    array_to_csv(w8, "w8.csv")
    array_to_csv(w9, "w9.csv")
    print("Done.")
