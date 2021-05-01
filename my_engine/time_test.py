#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Test.

Test time for random number.
"""
import random
import time

start = time.time()
list = list()
for a in range(1000000):
    list.append(random.random())
end = time.time()
print("Time :", end - start)
