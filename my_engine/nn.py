import chess
import csv

def csv_to_array(csv_path):
    r = []
    with open(csv_path) as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            r.append(row)
    return r

def array_to_csv(array, csv_path):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in array:
            writer.writerow(row)
        file.close()
    return 0

print("Loading weights...", end=" ")
wa = csv_to_array("wa.csv")
wb = csv_to_array("wb.csv")
wc = csv_to_array("wc.csv")
print("Done.")

def normalisation(x):
    if x > 1:
        return 1
    elif x < -1:
        return -1
    else:
        return x

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

