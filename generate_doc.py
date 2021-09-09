#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crocrodile - Generate documentation."""
import os
import sys

EXIT_CODE = 0

EXIT_CODE += os.system("make html")
os.chdir("build/html")
EXIT_CODE += os.system("firefox-esr index.html")

sys.exit(EXIT_CODE)
