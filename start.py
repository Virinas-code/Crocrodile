#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile Starter.

Starts Crocrodile.
"""
import sys
import crocrodile
import crocrodile.nn
import crocrodile.nn.basics_train

print(sys.argv)

if sys.argv[1] == "uci":
    print("Starting: UCI")
    uci = crocrodile.uci.UCI()
    uci.run()
elif sys.argv[1] == "basics":
    crocrodile.nn.basics_train.main(sys.argv)
elif sys.argv[1] == "client":
    import crocrodile.client
    crocrodile.client.main(sys.argv)
else:
    print("ERROR: Target not found", file=sys.stderr)
    sys.exit(1)
