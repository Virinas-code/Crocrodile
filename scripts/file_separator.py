#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile NN Training files separation tool.

Selects a given number of elements into a NN Training file.
"""
import os
import random

os.chdir("..")

file = input("From file : ")
number = int(input("Number of elements : "))
to = input("To file : ")

print("Reading...", end=" ", flush=True)

file = open(file).read().split("\n\n")
to = open(to, 'w')

print("Done.")
print("Getting random elements...", end=" ", flush=True)

sample = random.choices(file, k=number)

print("Done.")
print("Writing in destination file...", end=" ", flush=True)

to.write("\n\n".join(sample))

print("Done.")
