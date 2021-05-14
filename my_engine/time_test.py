#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Test.

Test time for evaluation function.
"""
import time
import numpy

start = time.time()
random_matrix = numpy.random.rand(64, 64)
for a in range(1000):
    random_matrix @ random_matrix
end = time.time()
print("Time :", end - start)
