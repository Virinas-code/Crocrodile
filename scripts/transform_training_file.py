import chess


def flip_file(file: str) -> str:
    files = list("abcdefgh")
    reversed_file = list(reversed(files))
    return reversed_file[files.index(file)]


def flip_move_horizontal(uci: str) -> str:
    """
    Flip move horizontally.

    :param str uci: UCI representation of move
    :return: UCI representation of flipped move
    :rtype: str
    """
    from_square = list(uci[0:2])
    to_square = list(uci[2:4])
    if len(uci) == 5:  # Promotion
        promotion = uci[4]
    else:
        promotion = ""
    from_square[0] = flip_file(from_square[0])
    to_square[0] = flip_file(to_square[0])
    return "".join(from_square) + "".join(to_square) + promotion


def flip_rank(rank: str) -> str:
    ranks = list("12345678")
    reversed_ranks = list(reversed(ranks))
    return reversed_ranks[ranks.index(rank)]


def flip_move_vertical(uci: str) -> str:
    """
    Flip move vertically.

    :param str uci: UCI representation of move
    :return: UCI representation of flipped move
    :rtype: str
    """
    from_square = list(uci[0:2])
    to_square = list(uci[2:4])
    if len(uci) == 5:  # Promotion
        promotion = uci[4]
    else:
        promotion = ""
    from_square[1] = flip_rank(from_square[1])
    to_square[1] = flip_rank(to_square[1])
    return "".join(from_square) + "".join(to_square) + promotion


input_file = open("training_files/" + input("Step 1: Input file\t")).read().split("\n\n")
print("Step 2: Transforming...", end=" ", flush=True)
result = list()
for position in input_file:
    result.append(position)
    fen = position.split("\n")[0]
    move = position.split("\n")[1]
    # Horizontal
    board = chess.Board(fen).transform(chess.flip_horizontal)
    transformed_move = flip_move_horizontal(move)
    result.append(f"{board.fen()}\n{transformed_move}")
    if not "p" in fen and not "P" in fen:
        # Vertical
        board = chess.Board(fen).transform(chess.flip_vertical)
        transformed_move = flip_move_vertical(move)
        result.append(f"{board.fen()}\n{transformed_move}")
        # Vertical + horizontal
        board = chess.Board(fen).transform(chess.flip_vertical).transform(chess.flip_horizontal)
        transformed_move = flip_move_vertical(flip_move_horizontal(move))
        result.append(f"{board.fen()}\n{transformed_move}")
print("Done.")
output_file = open("training_files/" + input("Step 3: Output file\t"), 'w').write("\n\n".join(result))