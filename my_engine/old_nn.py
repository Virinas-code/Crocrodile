#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NN of Crocrodile.

Tools to manage Crocrodile NN.
"""
import csv
import copy
import random
import time
import math
import numpy
import chess


def csv_to_array(csv_path):
    result = []
    with open(csv_path) as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            result.append(row)
    return result

def array_to_csv(array, csv_path):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in array:
            writer.writerow(row)
        file.close()
    return 0

print("Loading weights...")
wa = numpy.array(csv_to_array("wa.csv"))
wb = numpy.array(csv_to_array("wb.csv"))
wc = numpy.array(csv_to_array("wc.csv"))
print("Done.")

def normalisation(x):
    """Sigmoïde modifiée."""
    return (1 / (1 + math.exp(-x))) * 2 - 1

def nn_opening_white_check_move(fen, move): # Move is UCI str
    board = chess.Board(fen=fen)
    pieces = board.piece_map()
    INPUTS_VALUES = {'': 0, 'P': 0.1, 'N': 0.2, 'B': 0.3, 'R': 0.5, 'Q': 0.6, 'K': 0.7, 'p': -0.1, 'n': -0.2, 'b': -0.3, 'r': -0.5, 'q': -0.6, 'k': -0.7}
    inputs = []
    for a in range(64):
        if pieces.get(a, None):
            inputs.append(INPUTS_VALUES.get(pieces[a].symbol(), 0))
        else:
            inputs.append(0)
    if board.has_kingside_castling_rights(chess.WHITE):
        inputs.append(1)
    else:
        inputs.append(0)
    if board.has_queenside_castling_rights(chess.WHITE):
        inputs.append(1)
    else:
        inputs.append(0)
    if board.has_kingside_castling_rights(chess.BLACK):
        inputs.append(1)
    else:
        inputs.append(0)
    if board.has_queenside_castling_rights(chess.BLACK):
        inputs.append(1)
    else:
        inputs.append(0)
    if board.has_legal_en_passant():
        inputs.append(chess.square_file(board.ep_square) / 10)
    else:
        inputs.append(-1)
    move = chess.Move.from_uci(move)
    from_square = move.from_square
    inputs.append(chess.square_file(from_square) / 10)
    inputs.append(chess.square_rank(from_square) / 10)
    to_square = move.to_square
    inputs.append(chess.square_file(to_square) / 10)
    inputs.append(chess.square_rank(to_square) / 10)
    inputs.append(1)
    # print("Inputs :", inputs)
    cache1 = list()
    cache2 = list()
    for a in range(38):
        cache1.append(0)
        cache2.append(0)
    output = float()
    for j in range(38):
        current = 0
        for i in range(74):
            current += inputs[i] * wa[i][j]
        cache1[j] = normalisation(current)
    # print(cache1)
    #print("Moyenne :", sum(cache1) / len(cache1))
    for j in range(38):
        current = 0
        for i in range(38):
            current += cache1[i] * wb[i][j]
        cache2[j] = normalisation(current)
    # print(cache2)
    # print("Moyenne :", sum(cache2) / len(cache2))
    for j in range(1):
        current = 0
        for i in range(38):
            current += cache2[i] * wc[i][j]
        output = current
    #print("Brut :", output)
    if output >= 0:
        output = 1
    else:
        output = -1
    # print("Output :", output)
    return output

def train():
    with open("my_engine/training_boncoups_ouverture_blancs.txt") as file:
        file1 = file.read()
        file.close()
    with open("my_engine/training_mauvaiscoups_ouverture_blancs.txt") as file:
        file2 = file.read()
        file.close()
    file1 = file1.split("\n\n")
    file2 = file2.split("\n\n")
    l = len(file1) + len(file2)
    print("==== CHECKING ====")
    errs = 0
    good = 0
    for inputs in file1:
        pos = inputs.split("\n")[0]
        mve = inputs.split("\n")[1]
        res = nn_opening_white_check_move(pos, mve)
        if res == 1:
           good += 1
        else:
            errs += 1
    for inputs in file2:
        pos = inputs.split("\n")[0]
        mve = inputs.split("\n")[1]
        res = nn_opening_white_check_move(pos, mve)
        if res == -1:
            good += 1
        else:
            errs += 1
    print("Errors : {0}/{1} tests".format(errs, l))
    default = copy.copy(errs)
    print("Good moves : {0}/{1} tests".format(good, l))
    sessions = 1  # inintialise le nombre de sessions d'optimisation
    loops = int(input("Nombre de répétitions : "))
    add = float(input("Valeur à ajouter : "))
    for loop in range(loops):
        print("==== TRAINING #{0} ====".format(loop))
        print("---- WC TRAINING ----")
        best = -1
        best_score = float('inf')
        rand = (random.randint(0, 37), 0)
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
                good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Training WC[{0}][{1}] : {2} errors".format(rand[0], rand[1], errs))
        if errs < best_score:
            best = 0
            best_score = errs
        wc[rand[0]][rand[1]] += add
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
                good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Training WC[{0}][{1}] + add : {2} errors".format(rand[0], rand[1], errs))
        if errs < best_score:
            best = 1
            best_score = errs
        wc[rand[0]][rand[1]] -= add
        wc[rand[0]][rand[1]] -= add
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
                good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Training WC[{0}][{1}] - add : {2} errors".format(rand[0], rand[1], errs))
        if errs < best_score:
            best = 2
            best_score = errs
        wc[rand[0]][rand[1]] += add
        print("#### Updating neural network... ####")
        if best == 0:
            pass  # Il ne faut rien faire...
        elif best == 1:
            wc[rand[0]][rand[1]] += add
        else:
            wc[rand[0]][rand[1]] -= add
        print("Minimum  : {0}".format(best_score))
        print("Index : {0}".format(rand))
        if best == 0:
            print("Modification : Nothing")
        elif best == 1:
            print("Modification : + add")
        else:
            print("Modification : - add")
        print("####################################")
        print("---- RESULTS ----")
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
               good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Errors : {0}/{1} tests".format(errs, l))
        default = copy.copy(errs)
        print("Good moves : {0}/{1} tests".format(good, l))
        print("---- WB TRAINING ----")
        best = -1
        best_score = float('inf')
        rand = (random.randint(0, 37), random.randint(0, 37))
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
                good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Training WB[{0}][{1}]  : {2} errors".format(rand[0], rand[1], errs))
        if errs < best_score:
            best = 0
            best_score = errs
        wb[rand[0]][rand[1]] += add
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
                good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Training WB[{0}][{1}] + add : {2} errors".format(rand[0], rand[1], errs))
        if errs < best_score:
            best = 1
            best_score = errs
        wb[rand[0]][rand[1]] -= add
        wb[rand[0]][rand[1]] -= add
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
                good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Training WB[{0}][{1}] - add : {2} errors".format(rand[0], rand[1], errs))
        if errs < best_score:
            best = 2
            best_score = errs
        wb[rand[0]][rand[1]] += add
        print("#### Updating neural network... ####")
        if best == 0:
            pass  # Il ne faut rien faire...
        elif best == 1:
            wb[rand[0]][rand[1]] += add
        else:
            wb[rand[0]][rand[1]] -= add
        print("Minimum  : {0}".format(best_score))
        print("Index : {0}".format(rand))
        if best == 0:
            print("Modification : Nothing")
        elif best == 1:
            print("Modification : + add")
        else:
            print("Modification : - add")
        print("####################################")
        print("---- RESULTS ----")
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
               good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Errors : {0}/{1} tests".format(errs, l))
        default = copy.copy(errs)
        print("Good moves : {0}/{1} tests".format(good, l))
        print("---- WA TRAINING ----")
        best = -1
        best_score = float('inf')
        rand = (random.randint(0, 73), random.randint(0, 37))
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
                good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Training WA[{0}][{1}]  : {2} errors".format(rand[0], rand[1], errs))
        if errs < best_score:
            best = 0
            best_score = errs
        wa[rand[0]][rand[1]] += add
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
                good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Training WA[{0}][{1}] + add : {2} errors".format(rand[0], rand[1], errs))
        if errs < best_score:
            best = 1
            best_score = errs
        wa[rand[0]][rand[1]] -= add
        wa[rand[0]][rand[1]] -= add
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
                good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Training WA[{0}][{1}] - add : {2} errors".format(rand[0], rand[1], errs))
        if errs < best_score:
            best = 2
            best_score = errs
        wa[rand[0]][rand[1]] += add
        print("#### Updating neural network... ####")
        if best == 0:
            pass  # Il ne faut rien faire...
        elif best == 1:
            wa[rand[0]][rand[1]] += add
        else:
            wa[rand[0]][rand[1]] -= add
        print("Minimum  : {0}".format(best_score))
        print("Index : {0}".format(rand))
        if best == 0:
            print("Modification : Nothing")
        elif best == 1:
            print("Modification : + add")
        else:
            print("Modification : - add")
        print("####################################")
        print("---- RESULTS ----")
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
               good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Errors : {0}/{1} tests".format(errs, l))
        default = copy.copy(errs)
        print("Good moves : {0}/{1} tests".format(good, l))
        """
        print("==== TRAINING #{0} ====".format(sessions))
        print("---- WB TRAINING ----")
        index = (0,0)
        mini = float('inf')
        for a in range(len(wb)):
            for b in range(len(wb[0])):
                wb[a][b] += 0.1
                errs = 0
                good = 0
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 1
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
                print("Training WB[{0}][{1}] + 0.1 : {2} errors".format(a, b, errs))
                if errs < mini:
                    index = (a, b)
                    mini = errs
                    sign = True
                wb[a][b] -= 0.1
                wb[a][b] -= 0.1
                errs = 0
                good = 0
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 1
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
                print("Training WB[{0}][{1}] - 0.1 : {2} errors".format(a, b, errs))
                if errs < mini:
                    index = (a, b)
                    mini = errs
                    sign = False
                wb[a][b] += 0.1
        print("#### Updating neural network... ####")
        if sign:
            wb[index[0]][index[1]] += 0.1
        else:
            wb[index[0]][index[1]] -= 0.1
        print("Minimum  : {0}".format(mini))
        print("Index : {0}".format(index))
        print("Saving...")
        array_to_csv(wb, "wb.csv")
        print("Saved to wb.csv")
        print("####################################")
        print("---- RESULTS ----")
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
               good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Errors : {0}/{1} tests".format(errs, l))
        default = copy.copy(errs)
        print("Good moves : {0}/{1} tests".format(good, l))
        print("==== TRAINING #{0} ====".format(sessions))
        print("---- WA TRAINING ----")
        index = (0,0)
        mini = float('inf')
        for a in range(len(wa)):
            for b in range(len(wa[0])):
                wa[a][b] += 0.1
                errs = 0
                good = 0
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 1
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
                print("Training WA[{0}][{1}] + 0.1 : {2} errors".format(a, b, errs))
                if errs < mini:
                    index = (a, b)
                    mini = errs
                    sign = True
                wa[a][b] -= 0.1
                wa[a][b] -= 0.1
                errs = 0
                good = 0
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 1
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
                print("Training WA[{0}][{1}] - 0.1 : {2} errors".format(a, b, errs))
                if errs < mini:
                    index = (a, b)
                    mini = errs
                    sign = False
                wa[a][b] += 0.1
        print("#### Updating neural network... ####")
        if sign:
            wa[index[0]][index[1]] += 0.1
        else:
            wa[index[0]][index[1]] -= 0.1
        print("Minimum  : {0}".format(mini))
        print("Index : {0}".format(index))
        print("Saving...")
        array_to_csv(wa, "wa.csv")
        print("Saved to wa.csv")
        print("####################################")
        print("---- RESULTS ----")
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
               good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Errors : {0}/{1} tests".format(errs, l))
        default = copy.copy(errs)
        print("Good moves : {0}/{1} tests".format(good, l))
        print("==== RESULTS ====")
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
               good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Errors : {0}/{1} tests".format(errs, l))
        default = copy.copy(errs)
        print("Good moves : {0}/{1} tests".format(good, l))
        # return results
        """

    print("Saving...")
    array_to_csv(wc, "wc.csv")
    print("Saved to wc.csv")
    print("Saving...")
    array_to_csv(wb, "wb.csv")
    print("Saved to wb.csv")
    print("Saving...")
    array_to_csv(wa, "wa.csv")
    print("Saved to wa.csv")



def systematic_train():
    with open("my_engine/train_data_goodmoves.txt") as file:
        file1 = file.read()
        file.close()
    with open("my_engine/train_data_badmoves.txt") as file:
        file2 = file.read()
        file.close()
    file1 = file1.split("\n\n")
    file2 = file2.split("\n\n")
    file1_count = int(input("Nombre de bon coups : "))
    file2_count = int(input("Nombre de mauvais coups : "))
    file1 = random.sample(file1, file1_count)
    file2 = random.sample(file2, file2_count)
    l = len(file1) + len(file2)
    print("==== CHECKING ====")
    errs = 0
    good = 0
    for inputs in file1:
        pos = inputs.split("\n")[0]
        mve = inputs.split("\n")[1]
        res = nn_opening_white_check_move(pos, mve)
        if res == 1:
           good += 1
        else:
            errs += 5
    for inputs in file2:
        pos = inputs.split("\n")[0]
        mve = inputs.split("\n")[1]
        res = nn_opening_white_check_move(pos, mve)
        if res == -1:
            good += 1
        else:
            errs += 1
    print("Errors : {0} ({1} tests)".format(errs, l))
    default = copy.copy(errs)
    print("Good moves : {0}/{1} tests".format(good, l))
    sessions = 0  # inintialise le nombre de sessions d'optimisation
    add = float(input("Valeur à ajouter initiale : "))
    min_add = float(input("Valeur limite pour add : "))
    multi = float(input("Multiplier : "))
    while add > min_add:
        sessions += 1
        print("==== TRAINING #{0} ====".format(sessions))
        print("---- WC TRAINING ----")
        tstart = time.time()
        best = -1
        best_score = float('inf')
        rand = (random.randint(0, 37), 0)
        errs = 0
        good = 0
        for rand0 in range(len(wc)):
            for rand1 in range(len(wc[0])):
                equals = 0
                print("Training WC[{0}][{1}]...  ".format(rand0, rand1), end="\r")
                best = -1
                best_score = float('inf')
                errs = 0
                good = 0
                rand = (rand0, rand1)
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 5
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
                # print("Training WC[{0}][{1}] : {2} errors".format(rand[0], rand[1], errs))
                if errs < best_score:
                    best = 0
                    best_score = errs
                wc[rand[0]][rand[1]] += add
                errs = 0
                good = 0
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 5
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
                # print("Training WC[{0}][{1}] + add : {2} errors".format(rand[0], rand[1], errs))
                if errs < best_score:
                    best = 1
                    best_score = errs
                if errs == best_score:
                    equals += 1
                wc[rand[0]][rand[1]] -= add
                wc[rand[0]][rand[1]] -= add
                errs = 0
                good = 0
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 5
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
                # print("Training WC[{0}][{1}] - add : {2} errors".format(rand[0], rand[1], errs))
                if errs < best_score:
                    best = 2
                    best_score = errs
                if errs == best_score:
                    equals += 1
                wc[rand[0]][rand[1]] += add
                if equals == 2:
                    wc[rand[0]][rand[1]] += min_add
                # print("#### Updating neural network... ####")
                if best == 0:
                    pass  # Il ne faut rien faire...
                elif best == 1:
                    wc[rand[0]][rand[1]] += add
                else:
                    wc[rand[0]][rand[1]] -= add
                # print("Minimum  : {0}".format(best_score))
                # print("Index : {0}".format(rand))
                """
                if best == 0:
                    # print("Modification : Nothing")
                elif best == 1:
                    # print("Modification : + add")
                else:
                    # print("Modification : - add")
                """
                # print("####################################")
                if best_score == 0:
                    print("Errors : 0 - Saving...           ")
                    array_to_csv(wc, "wc.csv")
                    print("Saved to wc.csv")
                    print("Saving...")
                    array_to_csv(wb, "wb.csv")
                    print("Saved to wb.csv")
                    print("Saving...")
                    array_to_csv(wa, "wa.csv")
                    print("Saved to wa.csv")
                    exit(0)
        print("Done after {0} s.          ".format(time.time() - tstart))
        print("---- RESULTS ----")
        errs = 0
        good = 0
        ogm = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
               good += 1
            else:
                errs += 5
                ogm += 5
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Errors : {0} ({1} tests) (on good moves : {2}/{3} tests)".format(errs, l, ogm, len(file1)))
        default = copy.copy(errs)
        print("Good moves : {0}/{1} tests".format(good, l))
        print("---- WB TRAINING ----")
        tstart = time.time()
        best = -1
        best_score = float('inf')
        rand = (random.randint(0, 37), 0)
        errs = 0
        good = 0
        for rand0 in range(len(wb)):
            for rand1 in range(len(wb[0])):
                equals = 0
                print("Training WB[{0}][{1}]...  ".format(rand0, rand1), end="\r")
                best = -1
                best_score = float('inf')
                errs = 0
                good = 0
                rand = (rand0, rand1)
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 5
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
                # print("Training WB[{0}][{1}] : {2} errors".format(rand[0], rand[1], errs))
                if errs < best_score:
                    best = 0
                    best_score = errs
                wb[rand[0]][rand[1]] += add
                errs = 0
                good = 0
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 5
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
                # print("Training WB[{0}][{1}] + add : {2} errors".format(rand[0], rand[1], errs))
                if errs < best_score:
                    best = 1
                    best_score = errs
                if errs == best_score:
                    equals += 1
                wb[rand[0]][rand[1]] -= add
                wb[rand[0]][rand[1]] -= add
                errs = 0
                good = 0
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 5
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
                # print("Training WB[{0}][{1}] - add : {2} errors".format(rand[0], rand[1], errs))
                if errs < best_score:
                    best = 2
                    best_score = errs
                if errs == best_score:
                    equals += 1
                wb[rand[0]][rand[1]] += add
                if equals == 2:
                    wb[rand[0]][rand[1]] += min_add
                # print("#### Updating neural network... ####")
                if best == 0:
                    pass  # Il ne faut rien faire...
                elif best == 1:
                    wb[rand[0]][rand[1]] += add
                else:
                    wb[rand[0]][rand[1]] -= add
                # print("Minimum  : {0}".format(best_score))
                # print("Index : {0}".format(rand))
                """
                if best == 0:
                    # print("Modification : Nothing")
                elif best == 1:
                    # print("Modification : + add")
                else:
                    # print("Modification : - add")
                """
                # print("####################################")
                if best_score == 0:
                    print("Errors : 0 - Saving...           ")
                    array_to_csv(wc, "wc.csv")
                    print("Saved to wc.csv")
                    print("Saving...")
                    array_to_csv(wb, "wb.csv")
                    print("Saved to wb.csv")
                    print("Saving...")
                    array_to_csv(wa, "wa.csv")
                    print("Saved to wa.csv")
                    exit(0)
        print("Done after {0} s.          ".format(time.time() - tstart))
        print("---- RESULTS ----")
        errs = 0
        good = 0
        ogm = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
               good += 1
            else:
                errs += 5
                ogm += 5
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Errors : {0} ({1} tests) (on good moves : {2}/{3} tests)".format(errs, l, ogm, len(file1)))
        default = copy.copy(errs)
        print("Good moves : {0}/{1} tests".format(good, l))
        print("---- WA TRAINING ----")
        tstart = time.time()
        best = -1
        best_score = float('inf')
        rand = (random.randint(0, 37), 0)
        errs = 0
        good = 0
        for rand0 in range(len(wa)):
            for rand1 in range(len(wa[0])):
                equals = 0
                print("Training WA[{0}][{1}]...  ".format(rand0, rand1), end="\r")
                best = -1
                best_score = float('inf')
                errs = 0
                good = 0
                rand = (rand0, rand1)
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 5
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
##                print("Training WA[{0}][{1}] : {2} errors".format(rand[0], rand[1], errs))
                if errs < best_score:
                    best = 0
                    best_score = errs
                wa[rand[0]][rand[1]] += add
                errs = 0
                good = 0
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 5
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
##                print("Training WA[{0}][{1}] + add : {2} errors".format(rand[0], rand[1], errs))
                if errs < best_score:
                    best = 1
                    best_score = errs
                if errs == best_score:
                    equals += 1
                wa[rand[0]][rand[1]] -= add
                wa[rand[0]][rand[1]] -= add
                errs = 0
                good = 0
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 5
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
##                print("Training WA[{0}][{1}] - add : {2} errors".format(rand[0], rand[1], errs))
                if errs < best_score:
                    best = 2
                    best_score = errs
                if errs == best_score:
                    equals += 1
                wa[rand[0]][rand[1]] += add
                if equals == 2:
                    wa[rand[0]][rand[1]] += min_add
##                print("#### Updating neural network... ####")
                if best == 0:
                    pass  # Il ne faut rien faire...
                elif best == 1:
                    wa[rand[0]][rand[1]] += add
                else:
                    wa[rand[0]][rand[1]] -= add
##                print("Minimum  : {0}".format(best_score))
##                print("Index : {0}".format(rand))
                """
                if best == 0:
##                    print("Modification : Nothing")
                elif best == 1:
##                    print("Modification : + add")
                else:
##                    print("Modification : - add")
##                print("####################################")
                """
                if best_score == 0:
                    print("Errors : 0 - Saving...           ")
                    array_to_csv(wc, "wc.csv")
                    print("Saved to wc.csv")
                    print("Saving...")
                    array_to_csv(wb, "wb.csv")
                    print("Saved to wb.csv")
                    print("Saving...")
                    array_to_csv(wa, "wa.csv")
                    print("Saved to wa.csv")
                    exit(0)
        print("Done after {0} s.          ".format(time.time() - tstart))
        print("---- RESULTS ----")
        errs = 0
        good = 0
        ogm = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
               good += 1
            else:
                errs += 5
                ogm += 5
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Errors : {0} ({1} tests) (on good moves : {2}/{3} tests)".format(errs, l, ogm, len(file1)))
        default = copy.copy(errs)
        print("Good moves : {0}/{1} tests".format(good, l))
        print("Saving...")
        array_to_csv(wc, "wc.csv")
        print("Saved to wc.csv")
        print("Saving...")
        array_to_csv(wb, "wb.csv")
        print("Saved to wb.csv")
        print("Saving...")
        array_to_csv(wa, "wa.csv")
        print("Saved to wa.csv")
        print("Saved.")
        print("Updating loop system...")
        add = add * multi
        print("New add value :", add)
        print("Updated.")
        """
        print("==== TRAINING #{0} ====".format(sessions))
        print("---- WB TRAINING ----")
        index = (0,0)
        mini = float('inf')
        for a in range(len(wb)):
            for b in range(len(wb[0])):
                wb[a][b] += 0.1
                errs = 0
                good = 0
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 1
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
                print("Training WB[{0}][{1}] + 0.1 : {2} errors".format(a, b, errs))
                if errs < mini:
                    index = (a, b)
                    mini = errs
                    sign = True
                wb[a][b] -= 0.1
                wb[a][b] -= 0.1
                errs = 0
                good = 0
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 1
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
                print("Training WB[{0}][{1}] - 0.1 : {2} errors".format(a, b, errs))
                if errs < mini:
                    index = (a, b)
                    mini = errs
                    sign = False
                wb[a][b] += 0.1
        print("#### Updating neural network... ####")
        if sign:
            wb[index[0]][index[1]] += 0.1
        else:
            wb[index[0]][index[1]] -= 0.1
        print("Minimum  : {0}".format(mini))
        print("Index : {0}".format(index))
        print("Saving...")
        array_to_csv(wb, "wb.csv")
        print("Saved to wb.csv")
        print("####################################")
        print("---- RESULTS ----")
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
               good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Errors : {0}/{1} tests".format(errs, l))
        default = copy.copy(errs)
        print("Good moves : {0}/{1} tests".format(good, l))
        print("==== TRAINING #{0} ====".format(sessions))
        print("---- WA TRAINING ----")
        index = (0,0)
        mini = float('inf')
        for a in range(len(wa)):
            for b in range(len(wa[0])):
                wa[a][b] += 0.1
                errs = 0
                good = 0
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 1
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
                print("Training WA[{0}][{1}] + 0.1 : {2} errors".format(a, b, errs))
                if errs < mini:
                    index = (a, b)
                    mini = errs
                    sign = True
                wa[a][b] -= 0.1
                wa[a][b] -= 0.1
                errs = 0
                good = 0
                for inputs in file1:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == 1:
                        good += 1
                    else:
                        errs += 1
                for inputs in file2:
                    pos = inputs.split("\n")[0]
                    mve = inputs.split("\n")[1]
                    res = nn_opening_white_check_move(pos, mve)
                    if res == -1:
                        good += 1
                    else:
                        errs += 1
                print("Training WA[{0}][{1}] - 0.1 : {2} errors".format(a, b, errs))
                if errs < mini:
                    index = (a, b)
                    mini = errs
                    sign = False
                wa[a][b] += 0.1
        print("#### Updating neural network... ####")
        if sign:
            wa[index[0]][index[1]] += 0.1
        else:
            wa[index[0]][index[1]] -= 0.1
        print("Minimum  : {0}".format(mini))
        print("Index : {0}".format(index))
        print("Saving...")
        array_to_csv(wa, "wa.csv")
        print("Saved to wa.csv")
        print("####################################")
        print("---- RESULTS ----")
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
               good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Errors : {0}/{1} tests".format(errs, l))
        default = copy.copy(errs)
        print("Good moves : {0}/{1} tests".format(good, l))
        print("==== RESULTS ====")
        errs = 0
        good = 0
        for inputs in file1:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == 1:
               good += 1
            else:
                errs += 1
        for inputs in file2:
            pos = inputs.split("\n")[0]
            mve = inputs.split("\n")[1]
            res = nn_opening_white_check_move(pos, mve)
            if res == -1:
                good += 1
            else:
                errs += 1
        print("Errors : {0}/{1} tests".format(errs, l))
        default = copy.copy(errs)
        print("Good moves : {0}/{1} tests".format(good, l))
        # return results
        """

    print("Saving...")
    array_to_csv(wc, "wc.csv")
    print("Saved to wc.csv")
    print("Saving...")
    array_to_csv(wb, "wb.csv")
    print("Saved to wb.csv")
    print("Saving...")
    array_to_csv(wa, "wa.csv")
    print("Saved to wa.csv")
# return results

def check_training():
    """Check on complete files."""
    print("Loading weights...", end=" ")
    wa = csv_to_array("wa.csv")
    wb = csv_to_array("wb.csv")
    wc = csv_to_array("wc.csv")
    print("Done.")
    with open("my_engine/train_data_goodmoves.txt") as file:
        file1 = file.read()
        file.close()
    with open("my_engine/train_data_badmoves.txt") as file:
        file2 = file.read()
        file.close()
    file1 = file1.split("\n\n")
    file2 = file2.split("\n\n")
    l = len(file1) + len(file2)
    errs = 0
    good = 0
    ogm = 0
    obm = 0
    for inputs in file1:
        pos = inputs.split("\n")[0]
        mve = inputs.split("\n")[1]
        res = nn_opening_white_check_move(pos, mve)
        if res == 1:
           good += 1
        else:
            errs += 1
            ogm += 1
    for inputs in file2:
        pos = inputs.split("\n")[0]
        mve = inputs.split("\n")[1]
        res = nn_opening_white_check_move(pos, mve)
        if res == -1:
            good += 1
        else:
            errs += 1
            obm += 1
    print("Errors : {0}/{1} tests (on good moves : {2}/{3} tests)".format(errs, l, ogm, len(file1)))
    default = copy.copy(errs)
    print("Good answers : {0:.2f}% (On good moves : {1:.2f}% | On bad moves : {2:.2f}%)".format(good/l*100, (len(file1)-ogm)/len(file1)*100, (len(file2)-obm)/len(file2)*100))


def check_test():
    """Check on complete files."""
    print("Loading weights...", end=" ")
    wa = csv_to_array("wa.csv")
    wb = csv_to_array("wb.csv")
    wc = csv_to_array("wc.csv")
    print("Done.")
    with open("my_engine/test_data_goodmoves.txt") as file:
        file1 = file.read()
        file.close()
    with open("my_engine/test_data_badmoves.txt") as file:
        file2 = file.read()
        file.close()
    file1 = file1.split("\n\n")
    file2 = file2.split("\n\n")
    l = len(file1) + len(file2)
    errs = 0
    good = 0
    ogm = 0
    obm = 0
    for inputs in file1:
        pos = inputs.split("\n")[0]
        mve = inputs.split("\n")[1]
        res = nn_opening_white_check_move(pos, mve)
        if res == 1:
           good += 1
        else:
            errs += 1
            ogm += 1
    for inputs in file2:
        pos = inputs.split("\n")[0]
        mve = inputs.split("\n")[1]
        res = nn_opening_white_check_move(pos, mve)
        if res == -1:
            good += 1
        else:
            errs += 1
            obm += 1
    print("Errors : {0}/{1} tests (on good moves : {2}/{3} tests)".format(errs, l, ogm, len(file1)))
    default = copy.copy(errs)
    print("Good answers : {0:.2f}% (On good moves : {1:.2f}% | On bad moves : {2:.2f}%)".format(good/l*100, (len(file1)-ogm)/len(file1)*100, (len(file2)-obm)/len(file2)*100))
