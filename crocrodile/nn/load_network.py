#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile NNs.

Load network: Load network to use it with the client.
"""
import os


def load_network(network=None):
    print("Welcome in Crocrodile Network Loader !")
    if not network:
        network = input("Select a network to load : ")

    print("Cleaning...")
    os.system('rm crocrodile/client/w1.csv crocrodile/client/w2.csv crocrodile/client/w3.csv crocrodile/client/w4.csv crocrodile/client/w5.csv crocrodile/client/b1.csv crocrodile/client/b2.csv crocrodile/client/b3.csv crocrodile/client/b4.csv crocrodile/client/b5.csv')
    print("Loading...")
    os.system(f"cp nns/{network}-w1.csv crocrodile/client/w1.csv")
    os.system(f"cp nns/{network}-w2.csv crocrodile/client/w2.csv")
    os.system(f"cp nns/{network}-w3.csv crocrodile/client/w3.csv")
    os.system(f"cp nns/{network}-w4.csv crocrodile/client/w4.csv")
    os.system(f"cp nns/{network}-w5.csv crocrodile/client/w5.csv")
    os.system(f"cp nns/{network}-b1.csv crocrodile/client/b1.csv")
    os.system(f"cp nns/{network}-b2.csv crocrodile/client/b2.csv")
    os.system(f"cp nns/{network}-b3.csv crocrodile/client/b3.csv")
    os.system(f"cp nns/{network}-b4.csv crocrodile/client/b4.csv")
    os.system(f"cp nns/{network}-b5.csv crocrodile/client/b5.csv")
    print("Network loaded !")


if __name__ == '__main__':
    load_network()
