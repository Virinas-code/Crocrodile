#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile client.

Lichess client for Crocrodile
"""
import copy
import datetime
import sys
import threading
import time

import berserk
import chess
import colorama

# from Crocrodile.main import yukoo
import crocrodile
import crocrodile.engine
import crocrodile.nn

yukoo = crocrodile.engine.EngineBase("Yukoo", "Virinas-code")
minimax = yukoo.minimax_std

SPEEDS = ["classical", "rapid", "blitz", "bullet"]
VARIANTS = ["standard", "fromPosition"]

main_log = open("main.log", "w")
error_log = open("error.log", "w")
debug_log = open("debug.log", "w")

colorama.init()


def _lok(id, *args, **kwargs):
    if kwargs.get("store", True):
        main_log.write(" ".join(str(arg) for arg in args) + "\n")
    else:
        del kwargs["store"]
    try:
        client.bots.post_message(id, " ".join(args))
        client.bots.post_message(id, " ".join(args), spectator=True)
    except:
        pass
    print(
        colorama.Style.RESET_ALL
        + colorama.Fore.GREEN
        + time.asctime(time.localtime())
        + ":",
        f"Game {id} |",
        *args,
        **kwargs,
    )


def _ldebug(*args, **kwargs):
    debug_log.write(" ".join(str(arg) for arg in args) + "\n")
    print(
        colorama.Style.RESET_ALL
        + colorama.Fore.MAGENTA
        + time.asctime(time.localtime())
        + ":",
        *args,
        **kwargs,
    )


def _lerr(id, *args, **kwargs):
    error_log.write(" ".join(str(arg) for arg in args) + "\n")
    try:
        client.bots.post_message(id, "ERROR: " + " ".join(args))
        client.bots.post_message(id, "ERROR: " + " ".join(args), spectator=True)
    except:
        pass
    print(
        colorama.Style.RESET_ALL
        + colorama.Fore.RED
        + time.asctime(time.localtime())
        + ":",
        *args,
        **kwargs,
    )


def lnone(*args, **kwargs):
    """Don't log anything."""
    pass


def show_user_description(user):
    """
    Show user description <Status> <Title> <Username>.
    Status: online ● offline ○
    Title: BOT, GM, WGM, CM, WCM, IM, WIM, LM or nothing

    :param dict user: JSON parsed from Lichess API
    :return: String
    :rtype: str
    """
    statuses = {True: "●", False: "○"}
    if not user["title"]:
        return f"{statuses[user['online']]} {user['name']}"
    return f"{statuses[user['online']]} {user['title']} {user['name']}"


ldebug = lnone
lok = _lok
lerr = _lerr
CHALLENGE = False
AUTO_CHALLENGE = False

with open("lichess.token") as file:
    token = file.read()[:-1]

session = berserk.TokenSession(token)
client = berserk.Client(session)
HELP_USAGE = 'Usage : client.py [-v | -q] [-h] \
                [-c "user time increment color" | \
                --challenge "user time increment color" | -a | --auto] \
                [-n | --neural-network] [-u | --upgrade] [-d | --dev]'
HELP_CHALLENGE = "\t-c, --challenge \"user time increment color\" : \
                    Challenge user in time+increment, BOT is playing with \
                    color ('white' or 'black')"


def limit_time(total_time: float, increment: int) -> float:
    """Calculate minimum time to calculate best move."""
    limit_time: float = increment + (total_time / 40)
    return limit_time


def start_depth(total_time: float) -> int:
    if total_time > 600:
        return 3
    elif total_time > 300:
        return 2
    else:
        return 1


class Game(threading.Thread):
    def __init__(self, client, game_id, color, fen, **kwargs):
        super().__init__(**kwargs)
        lok(game_id, "Starting... (FEN", fen + ")")
        self.engine = crocrodile.engine.EngineBase("Crocrodile", "Virinas-code")
        self.game_id = game_id
        self.initial_fen = fen
        self.client = client
        self.stream = client.bots.stream_game_state(game_id)
        if color == "white":
            self.my_turn = chess.BLACK
        else:
            self.my_turn = chess.WHITE
        if self.my_turn == chess.WHITE:
            self.time_control = "wtime"
        else:
            self.time_control = "btime"
        lok(self.game_id, "Started")

    def run(self):
        for event in self.stream:
            if event["type"] == "gameState":
                self.game_state_change(event)
            elif event["type"] == "chatLine":
                self.chat(event)
            elif event["type"] == "gameFull":
                self.game_full(event)
            else:
                ldebug(event["type"], ":", event)

    def game_state_change(self, event):
        ldebug("game state change", event)
        if event["status"] == "started":
            mvs = event["moves"].split(" ")
            board = chess.Board(self.initial_fen)
            first_pos = True
            if mvs != [""]:
                first_pos = False
                for move in mvs:
                    board.push(chess.Move.from_uci(move))
            ldebug("\n" + str(board))
            if board.turn == self.my_color:
                t = event[self.time_control].time()
                time_s = (t.hour * 60 + t.minute) * 60 + t.second
                t = event.get(
                    "winc",
                    datetime.datetime(
                        1970, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
                    ),
                )
                increment = (t.hour * 60 + t.minute) * 60 + t.second
                """
                lok("depth " + str(5) + ")")
                score, best_move = minimax(board, 5, board.turn)
                """
                if first_pos:
                    depth = 2
                    limit = 10
                else:
                    depth = start_depth(time_s)
                    limit = limit_time(time_s, increment)
                score = float("-inf")
                best_move = chess.Move.null()
                lok(
                    self.game_id,
                    "Maximum " + str(int(limit)) + "s to calculate",
                )
                limit += time.time()
                nn_moves: list = list()
                for move in self.engine.nn_select_best_moves(board):
                    nn_moves.append(move.uci())
                lok(
                    self.game_id,
                    "Legal moves :",
                    len(list(board.legal_moves)),
                    "/ Selected moves :",
                    len(self.engine.nn_select_best_moves(board)),
                    "(" + ", ".join(nn_moves) + ")",
                )
                last_score, last_best_move = self.engine.search(
                    board, depth, board.turn, float("inf")
                )
                while True:
                    depth += 1
                    score, best_move = self.engine.search(
                        board, depth, board.turn, limit
                    )
                    if score == float("inf"):
                        score = last_score
                        best_move = last_best_move
                        lok(
                            self.game_id,
                            "Depth " + str(depth) + ": Not enough time",
                        )
                        break
                    elif score == 10000 or score == -10000:
                        break
                    elif depth > 10:
                        break
                    else:
                        last_score, last_best_move = (
                            copy.deepcopy(score),
                            copy.deepcopy(best_move),
                        )

                retry = 3
                while retry > 0:
                    try:
                        self.client.bots.make_move(self.game_id, best_move)
                        retry = 0
                    except Exception as e:
                        lerr(self.game_id, "Error:", e)
                        time.sleep(3)
                        pass
                    retry = retry - 1
        elif event["status"] == "draw":
            lok(self.game_id, "Draw")
        elif event["status"] == "resign":
            if event["winner"] == "white":
                lok(self.game_id, "White wins - Black resign")
            else:
                lok(self.game_id, "White resigns - Black wins")
        elif event["status"] == "mate":
            if event["winner"] == "white":
                lok(self.game_id, "White wins - Black is mate")
            else:
                lok(self.game_id, "White is mate - Black wins")
        elif event["status"] == "aborted":
            lok(self.game_id, "Aborted")
        else:
            lok(self.game_id, event["status"].capitalize())
            sys.exit(0)

    def chat(self, event):
        ldebug(
            "Game",
            self.game_id,
            "|",
            event["room"].capitalize(),
            "@" + event["username"],
            "says",
            event["text"],
        )

    def game_full(self, event):
        lok(self.game_id, "Game full")
        if event["white"]["id"] in ("crocrodile-dev", "crocrodile"):
            self.my_color = True
            self.time_control = "wtime"
            lok(self.game_id, "Playing as White")
            self.game_state_change(
                {
                    "status": "started",
                    "moves": "",
                    "btime": datetime.datetime(1970, 1, 1, 12),
                    "wtime": datetime.datetime(1970, 1, 1, 12),
                }
            )
        else:
            self.my_color = False
            self.time_control = "btime"
            lok(self.game_id, "Playing as Black")


def main(argv: list) -> None:
    global client, lok, CHALLENGE, challenge_time, challenge_user, challenge_increment, challenge_color, ldebug, AUTO_CHALLENGE
    print("Starting...")
    if len(argv) > 1:
        argc = 0
        for arg in argv:
            if argc == 0:
                if arg == "-v" or arg == "--verbose":
                    ldebug = _ldebug
                if arg == "-q" or arg == "--quiet":
                    lok = lnone
                    ldebug = lnone
                    lerr = lnone
                if arg == "-h" or arg == "--help":
                    print(HELP_USAGE)
                    print("Description : Crocrodile Lichess client")
                    print("Commands :")
                    print("\t-h, --help : Show this message and exit")
                    print("\t-v, --verbose : Show debug logs")
                    print("\t-q, --quiet : Don't show any logs")
                    print(HELP_CHALLENGE)
                    print("\t-a, --auto : Auto challenge BOTs")
                    print("\t-n, --neural-network : Enable Neural Network")
                    print("\t-u, --upgrade: Upgrade to bot account")
                    print("\t-d, --dev : Dev account")
                    sys.exit(0)
                if arg == "-c" or arg == "--challenge":
                    argc = 1
                if arg in ("-a", "--auto"):
                    AUTO_CHALLENGE = True
                if arg in ("-n", "--neural-network"):
                    minimax = yukoo.search
                if arg in ("-u", "--upgrade"):
                    client.account.upgrade_to_bot()
                if arg in ("-d", "--dev"):
                    session = berserk.TokenSession(open("dev.token").read())
                    client = berserk.Client(session)
            else:
                arg_list = arg.split(" ")
                print(arg_list)
                if len(arg) > 3:
                    CHALLENGE = True
                    challenge_user = arg_list[0]
                    challenge_time = arg_list[1]
                    challenge_increment = arg_list[2]
                    challenge_color = arg_list[3]
                argc = 0
    else:
        ldebug = lnone
    lok(None, "Token is", token, store=False)

    lok(
        None,
        "Connected to",
        client.account.get().get("title", "USER"),
        client.account.get().get("username", "Anonymous"),
        store=False,
    )
    lok(None, "Waiting for challenges")
    continue_loop = True
    colors = {}
    fens = {}
    if CHALLENGE:
        print(challenge_time)
        challenge = client.challenges.create(
            challenge_user,
            True,
            clock_limit=int(float(challenge_time)) * 60,
            clock_increment=int(challenge_increment),
            color=challenge_color,
        )
        ldebug(challenge)
        if challenge["challenge"]["color"] == "white":
            colors[challenge["challenge"]["id"]] = "black"
        else:
            colors[challenge["challenge"]["id"]] = "white"
        fens[challenge["challenge"]["id"]] = chess.STARTING_FEN
    while continue_loop:
        for event in client.bots.stream_incoming_events():
            ldebug(event)
            if event["type"] == "challenge":
                lok(
                    event["challenge"]["id"],
                    "From",
                    show_user_description(event["challenge"]["challenger"]),
                    "- received",
                )
                if (
                    event["challenge"]["speed"] in SPEEDS
                    and event["challenge"]["variant"]["key"] in VARIANTS
                    and not event["challenge"]["id"] in colors
                    and event["challenge"]["challenger"]["id"] != "crocrodile"
                ):  # patch-002
                    client.bots.accept_challenge(event["challenge"]["id"])
                    lok(event["challenge"]["id"], "Accepted")
                    colors[event["challenge"]["id"]] = event["challenge"]["color"]
                    if event["challenge"]["variant"]["key"] == "fromPosition":
                        fens[event["challenge"]["id"]] = event["challenge"][
                            "initialFen"
                        ]
                    else:
                        fens[event["challenge"]["id"]] = chess.STARTING_FEN
                else:
                    if event["challenge"]["challenger"]["id"] != "crocrodile":
                        if event["challenge"]["id"] in colors:
                            lok(
                                event["challenge"]["id"],
                                "Declining because this is a rematch",
                            )
                        elif event["challenge"]["speed"] not in SPEEDS:
                            lok(
                                event["challenge"]["id"],
                                "Declining because the bot doesn't play this \
                                    speed ("
                                + event["challenge"]["speed"].capitalize()
                                + ")",
                            )
                        elif event["challenge"]["variant"]["key"] not in VARIANTS:
                            lok(
                                event["challenge"]["id"],
                                "Declining because the bot doesn't play this \
                                    variant ("
                                + event["challenge"]["variant"]["name"]
                                + ")",
                            )
                        else:
                            lok(
                                event["challenge"]["id"],
                                "Declining",
                            )
                        client.bots.decline_challenge(event["challenge"]["id"])
                        if event["challenge"]["id"] in colors:
                            client.bots.post_message(
                                event["challenge"]["id"],
                                "I don't accept rematches (lot of bugs)",
                            )
                    else:
                        lok(
                            event["challenge"]["id"],
                            "Challenging "
                            + show_user_description(event["challenge"]["destUser"]),
                        )
            elif event["type"] == "gameStart":
                game = Game(
                    client,
                    event["game"]["id"],
                    colors[event["game"]["id"]],
                    fens[event["game"]["id"]],
                )
                game.start()
            elif event["type"] == "gameFinish":
                if AUTO_CHALLENGE:
                    challenge = client.challenges.create(
                        challenge_user,
                        True,
                        clock_limit=int(float(challenge_time)) * 60,
                        clock_increment=int(challenge_increment),
                        color=challenge_color,
                    )
                    if challenge["challenge"]["color"] == "white":
                        colors[challenge["challenge"]["id"]] = "black"
                    else:
                        colors[challenge["challenge"]["id"]] = "white"
                    fens[challenge["challenge"]["id"]] = chess.STARTING_FEN
                lok(event["game"]["id"], "Finished")
            elif event["type"] == "challengeDeclined":
                if event["challenge"]["challenger"]["id"] == "crocrodile":
                    lok(
                        event["challenge"]["id"],
                        "Declined by "
                        + show_user_description(event["challenge"]["destUser"]),
                    )
                else:
                    lok(event["challenge"]["id"], "Declined")
            else:
                ldebug(event["type"], ":", event)


if __name__ == "__main__":
    main(sys.argv)
