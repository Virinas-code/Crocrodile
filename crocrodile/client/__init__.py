#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crocrodile client.

Lichess client for Crocrodile
"""
import threading
import datetime
import time
import sys
import berserk
import colorama
import chess
# from Crocrodile.main import yukoo
import crocrodile
from crocrodile.nn.load_network import load_network  # Network loader
import crocrodile.nn
import crocrodile.engine
import copy
yukoo = crocrodile.engine.EngineBase("Yukoo", "Virinas-code")
minimax = yukoo.minimax_std

SPEEDS = ['classical', 'rapid', 'blitz', 'bullet']
VARIANTS = ['standard', "fromPosition"]

main_log = open("main.log", 'w')
error_log = open("error.log", 'w')
debug_log = open("debug.log", 'w')

colorama.init()

# Load best network
load_network()


def _lok(*args, **kwargs):
    main_log.write(" ".join(str(arg) for arg in args) + "\n")
    print(colorama.Style.RESET_ALL + colorama.Fore.GREEN
          + time.asctime(time.localtime()) + ":", *args, **kwargs)


def _ldebug(*args, **kwargs):
    debug_log.write(" ".join(str(arg) for arg in args) + "\n")
    print(colorama.Style.RESET_ALL + colorama.Fore.MAGENTA
          + time.asctime(time.localtime()) + ":", *args, **kwargs)


def _lerr(*args, **kwargs):
    error_log.write(" ".join(str(arg) for arg in args) + "\n")
    print(colorama.Style.RESET_ALL + colorama.Fore.RED
          + time.asctime(time.localtime()) + ":", *args, **kwargs)


def lnone(*args, **kwargs):
    """Don't log anything."""
    pass


ldebug = lnone
lok = _lok
lerr = _lerr
CHALLENGE = False
AUTO_CHALLENGE = False

with open("lichess.token") as file:
    token = file.read()[:-1]

session = berserk.TokenSession(token)
client = berserk.Client(session)

if len(sys.argv) > 1:
    argc = 0
    for arg in sys.argv:
        if argc == 0:
            if arg == "-v" or arg == "--verbose":
                ldebug = _ldebug
            if arg == "-q" or arg == "--quiet":
                lok = lnone
                ldebug = lnone
                lerr = lnone
            if arg == "-h" or arg == "--help":
                print(
                    "Usage : client.py [-v | -q] [-h] [-c \"user time increment color\" | --challenge \"user time increment color\" | -a | --auto] [-n | --neural-network] [-u | --upgrade] [-d | --dev]")
                print("Description : Crocrodile Lichess client")
                print("Commands :")
                print("\t-h, --help : Show this message and exit")
                print("\t-v, --verbose : Show debug logs")
                print("\t-q, --quiet : Don't show any logs")
                print("\t-c, --challenge \"user time increment color\" : Challenge user in time+increment, BOT is playing with color ('white' or 'black')")
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
                minimax = yukoo.minimax_nn
            if arg in ("-u", "--upgrade"):
                client.account.upgrade_to_bot()
            if arg in ("-d", "--dev"):
                session = berserk.TokenSession(open("dev.token").read())
                client = berserk.Client(session)
        else:
            arg_list = arg.split(" ")
            if len(arg) > 3:
                CHALLENGE = True
                challenge_user = arg_list[0]
                challenge_time = arg_list[1]
                challenge_increment = arg_list[2]
                challenge_color = arg_list[3]
            argc = 0
else:
    ldebug = lnone


def limit_time(total_time: float, increment: int) -> float:
    """Calculate minimum time to calculate best move."""
    limit_time: float = increment + (total_time / 40)
    return limit_time


def start_depth(total_time: float) -> int:
    if total_time > 600:
        return 4
    elif total_time > 300:
        return 3
    else:
        return 2


class Game(threading.Thread):
    def __init__(self, client, game_id, color, fen, **kwargs):
        super().__init__(**kwargs)
        lok("Game", game_id, "| Starting... (FEN", fen + ")")
        self.game_id = game_id
        self.initial_fen = fen
        self.client = client
        self.stream = client.bots.stream_game_state(game_id)
        if color == 'white':
            self.my_turn = chess.BLACK
        else:
            self.my_turn = chess.WHITE
        if self.my_turn == chess.WHITE:
            self.time_control = "wtime"
        else:
            self.time_control = "btime"
        lok("Game", self.game_id, "| Started")

    def run(self):
        for event in self.stream:
            if event['type'] == 'gameState':
                self.game_state_change(event)
            elif event['type'] == 'chatLine':
                self.chat(event)
            elif event["type"] == "gameFull":
                self.game_full(event)
            else:
                ldebug(event["type"], ":", event)

    def game_state_change(self, event):
        ldebug("game state change", event)
        if event['status'] == "started":
            mvs = event['moves'].split(" ")
            board = chess.Board(self.initial_fen)
            first_pos = True
            if mvs != ['']:
                first_pos = False
                for move in mvs:
                    board.push(chess.Move.from_uci(move))
            ldebug("\n" + str(board))
            if board.turn == self.my_color:
                t = event[self.time_control].time()
                time_s = (t.hour * 60 + t.minute) * 60 + t.second
                t = event.get("winc", datetime.datetime(
                    1970, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc))
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
                score = float('-inf')
                best_move = chess.Move.null()
                lok("Game", self.game_id, "| Maximum "
                    + str(int(limit)) + "s to calculate")
                limit += time.time()
                lok("Game", self.game_id, "| Depth "
                    + str(depth) + ": Calculating...", end="\r")
                last_score, last_best_move = minimax(
                    board, depth, board.turn, float('inf'))
                lok("Game", self.game_id, "| Depth " + str(depth) + ": Score "
                    + str(last_score) + " (best move " + last_best_move.uci() + ")")
                while True:
                    depth += 1
                    lok("Game", self.game_id, "| Depth "
                        + str(depth) + ": Calculating...", end="\r")
                    score, best_move = minimax(board, depth, board.turn, limit)
                    if score == float('inf'):
                        score = last_score
                        best_move = last_best_move
                        lok("Game", self.game_id, "| Depth "
                            + str(depth) + ": Not enough time")
                        break
                    elif score == 10000 or score == -10000:
                        break
                    else:
                        lok("Game", self.game_id, "| Depth " + str(depth) + ": Score "
                            + str(score) + " (best move " + best_move.uci() + ")")
                        last_score, last_best_move = copy.deepcopy(
                            score), copy.deepcopy(best_move)

                retry = 3
                while retry > 0:
                    try:
                        self.client.bots.make_move(self.game_id, best_move)
                        retry = 0
                    except Exception as e:
                        lerr("Game", self.game_id, "| Error:", e)
                        time.sleep(3)
                        pass
                    retry = retry - 1
        elif event["status"] == "draw":
            lok("Game", self.game_id, "| Draw")
        elif event["status"] == "resign":
            if event["winner"] == "white":
                lok("Game", self.game_id, "| White wins - Black resign")
            else:
                lok("Game", self.game_id, "| White resigns - Black wins")
        else:
            lok("Game", self.game_id, "|", event['status'].capitalize())
            sys.exit(0)

    def chat(self, event):
        lok("Game", self.game_id, "|", event['room'].capitalize(
        ), "@" + event['username'], "says", event['text'])

    def game_full(self, event):
        lok("Game", self.game_id, "| Game full")
        if event["white"]["id"] in ("crocrodile-dev", "Crocrodile"):
            self.my_color = True
            self.time_control = "wtime"
            lok("Game", self.game_id, "| Playing as White")
            self.game_state_change({'status': 'started', 'moves': '', 'btime': datetime.datetime(
                1970, 1, 1, 12), 'wtime': datetime.datetime(1970, 1, 1, 12)})
        else:
            self.my_color = False
            self.time_control = "btime"
            lok("Game", self.game_id, "| Playing as Black")


lok("Token is", token)

lok("Connected to", client.account.get().get("title", "USER"),
    client.account.get().get("username", "Anonymous"))
lok("Waiting for challenges")
continue_loop = True
colors = {}
fens = {}
if CHALLENGE:
    print(challenge_time)
    challenge = client.challenges.create(challenge_user, True, clock_limit=int(float(
        challenge_time)) * 60, clock_increment=int(challenge_increment), color=challenge_color)
    ldebug(challenge)
    if challenge["challenge"]["color"] == "white":
        colors[challenge["challenge"]["id"]] = "black"
    else:
        colors[challenge["challenge"]["id"]] = "white"
    fens[challenge["challenge"]["id"]] = chess.STARTING_FEN
while continue_loop:
    for event in client.bots.stream_incoming_events():
        ldebug(event)
        if event['type'] == 'challenge':
            if event['challenge']['speed'] in SPEEDS and event['challenge']['variant']['key'] in VARIANTS and not event['challenge']['id'] in colors and event['challenge']['challenger']['id'] != "crocrodile" and event['challenge']['color'] != 'random':
                client.bots.accept_challenge(event['challenge']['id'])
                lok("Challenge", event["challenge"]["id"], "| Accepted")
                colors[event['challenge']['id']] = event['challenge']['color']
                if event["challenge"]["variant"]["key"] == "fromPosition":
                    fens[event["challenge"]["id"]
                         ] = event["challenge"]["initialFen"]
                else:
                    fens[event["challenge"]["id"]] = chess.STARTING_FEN
            else:
                if event["challenge"]["challenger"]["id"] != "crocrodile":
                    if event["challenge"]["color"] == "random":
                        lok("Challenge", event["challenge"]["id"],
                            "| Declining because this is a random color challenge")
                    elif event['challenge']['id'] in colors:
                        lok("Challenge", event["challenge"]["id"],
                            "| Declining because this is a rematch")
                    elif event['challenge']['speed'] not in SPEEDS:
                        lok("Challenge", event["challenge"]["id"], "| Declining because the bot doesn't play this speed ("
                            + event["challenge"]["speed"].capitalize() + ")")
                    elif event['challenge']['variant']['key'] not in VARIANTS:
                        lok("Challenge", event["challenge"]["id"],
                            "| Declining because the bot doesn't play this variant (" + event["challenge"]["variant"]["name"] + ")")
                    else:
                        lok("Challenge", event["challenge"]
                            ["id"], "| Declining")
                    client.bots.decline_challenge(event['challenge']['id'])
                    if event['challenge']['id'] in colors:
                        client.bots.post_message(
                            event['challenge']['id'], "I don't accept rematches (lot of bugs)")
                else:
                    lok("Challenge", event["challenge"]["id"],
                        "| Challenging @" + event["challenge"]["destUser"]["name"])
        elif event['type'] == 'gameStart':
            game = Game(
                client, event['game']['id'], colors[event['game']['id']], fens[event["game"]["id"]])
            game.start()
        elif event['type'] == 'gameFinish':
            if AUTO_CHALLENGE:
                challenge = client.challenges.create(challenge_user, True, clock_limit=int(float(
                    challenge_time)) * 60, clock_increment=int(challenge_increment), color=challenge_color)
                if challenge["challenge"]["color"] == "white":
                    colors[challenge["challenge"]["id"]] = "black"
                else:
                    colors[challenge["challenge"]["id"]] = "white"
                fens[challenge["challenge"]["id"]] = chess.STARTING_FEN
            lok("Game", event["game"]["id"], "| Finished")
        elif event["type"] == "challengeDeclined":
            if event["challenge"]["challenger"]["id"] == "crocrodile":
                lok("Challenge", event["challenge"]["id"],
                    "| Declined by @" + event["challenge"]["destUser"]["name"])
            else:
                lok("Challenge", event["challenge"]["id"], "| Declined")
        else:
            ldebug(event["type"], ":", event)
