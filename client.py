import berserk
import colorama
import time
import sys
import chess
import threading
import datetime
# from Crocrodile.main import yukoo
import my_engine
yukoo = my_engine.EngineBase("Yukoo", "Virinas-code")

SPEEDS = ['classical', 'rapid', 'blitz']
VARIANTS = ['standard', "fromPosition"]

main_log = open("main.log", 'w')
error_log = open("error.log", 'w')
debug_log = open("debug.log", 'w')

colorama.init()
def _lok(*args):
    main_log.write(" ".join(str(arg) for arg in args) + "\n")
    print(colorama.Style.RESET_ALL + colorama.Fore.GREEN + time.asctime(time.localtime()) + ":", *args)

def _ldebug(*args):
    debug_log.write(" ".join(str(arg) for arg in args) + "\n")
    print(colorama.Style.RESET_ALL + colorama.Fore.MAGENTA + time.asctime(time.localtime()) + ":", *args)

def _lerr(*args):
    error_log.write(" ".join(str(arg) for arg in args) + "\n")
    print(colorama.Style.RESET_ALL + colorama.Fore.RED + time.asctime(time.localtime()) + ":", *args)

def lnone(*args):
    pass

ldebug = lnone
lok = _lok
lerr = _lerr
challenge = False

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
                print("Usage : client.py [-v | -q] [-h] [-c \"user time increment color\" | --challenge \"user time increment color\"]")
                print("Description : Crocrodile Lichess client")
                print("Commands :")
                print("\t-h, --help : Show this message and exit")
                print("\t-v, --verbose : Show debug logs")
                print("\t-q, --quiet : Don't show any logs")
                print("\t-c, --challenge \"user time increment color\" : Challenge user in time+increment, BOT is playing with color ('white' or 'black')")
                exit(0)
            if arg == "-c" or arg == "--challenge":
                argc = 1
        else:
            arg = arg.split(" ")
            if len(arg) > 3:
                challenge = True
                challenge_user = arg[0]
                challenge_time = arg[1]
                challenge_increment = arg[2]
                challenge_color = arg[3]
            argc = 0
else:
    ldebug = lnone

class Game(threading.Thread):
    def __init__(self, client, game_id, color, fen, **kwargs):
        super().__init__(**kwargs)
        lok("Game", game_id, ": initial FEN", fen)
        self.game_id = game_id
        self.initial_fen = fen
        self.client = client
        self.stream = client.bots.stream_game_state(game_id)
        self.current_state = next(self.stream)
        if color == 'white':
            self.my_turn = chess.WHITE
        else:
            self.my_turn = chess.BLACK
        if self.my_turn != chess.WHITE:
            self.time_control = "wtime"
            self.game_state_change({'status':'started', 'moves':'0000 0000', 'btime':datetime.datetime(1970, 1, 1, 12), 'wtime': datetime.datetime(1970, 1, 1, 12)})
        else:
            self.time_control = "btime"
        lok("Game", self.game_id, "start")
    def run(self):
        for event in self.stream:
            if event['type'] == 'gameState':
                self.game_state_change(event)
            elif event['type'] == 'chatLine':
                self.chat(event)
            else:
                ldebug(event["type"], ":", event)
    def game_state_change(self, event):
        ldebug("game state change", event)
        if event['status'] == "started":
            lok("Game", self.game_id, ": moves", event['moves'])
            mvs = event['moves'].split(" ")
            board = chess.Board(self.initial_fen)
            for move in mvs:
                board.push(chess.Move.from_uci(move))
            ldebug("\n" + str(board))
            if board.turn != self.my_turn:
                t = event[self.time_control].time()
                time = (t.hour * 60 + t.minute) * 60 + t.second
                lok("Game", self.game_id, ": Calculating (time", str(time) + ")...")
                if True:  # time > 600 and len(mvs) % 4 == 0 or len(mvs) % 4 == 1
                    lok("Game", self.game_id, ": depth", 3)  # depth 4
                    score, best_move = yukoo.minimax(board, 3, board.turn)
                elif time < 120:
                    lok("Game", self.game_id, ": depth", 3)
                    score, best_move = yukoo.minimax(board, 3, board.turn, False)
                else:
                    if len(mvs) % 8 == 0 or len(mvs) % 8 == 1:
                        lok("Game", self.game_id, ": depth", 4)
                        score, best_move = yukoo.minimax(board, 4, board.turn, False)
                    else:
                        lok("Game", self.game_id, ": depth", 3)
                        score, best_move = yukoo.minimax(board, 3, board.turn, False)
                lok("Game", self.game_id, ": score", score, "(best move", str(best_move) + ")")
                retry = 3
                while retry > 0:
                    try:
                        self.client.bots.make_move(self.game_id, best_move)
                        retry = 0
                    except Exception as e:
                        lerr(type(e), e)
                        pass
                    retry = retry - 1
        else:
            lok("Game", self.game_id, ":", event['status'])
            sys.exit(0)
    def chat(self, event):
        lok("Game", self.game_id, ":", event['room'].capitalize(), event['username'], "says", event['text'])
lok("Initialized.")

with open("lichess.token") as file:
    token = file.read()[:-1]

lok("Token is", token)

session = berserk.TokenSession(token)
client = berserk.Client(session)

lok("Connected to", client.account.get().get("title", "USER"), client.account.get().get("username", "Anonymous"))
lok("Waiting for challenges")
continue_loop = True
colors = {}
fens = {}
if challenge:
    print(challenge_time)
    challenge = client.challenges.create(challenge_user, True, clock_limit=int(challenge_time), clock_increment=int(challenge_increment), color=challenge_color)
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
            if event['challenge']['speed'] in SPEEDS and event['challenge']['variant']['key'] in VARIANTS and not event['challenge']['id'] in colors:  #  and event['challenge']['color'] != 'random'
                client.bots.accept_challenge(event['challenge']['id'])
                colors[event['challenge']['id']] = event['challenge']['color']
                if event["challenge"]["variant"]["key"] == "fromPosition":
                    fens[event["challenge"]["id"]] = event["challenge"]["initialFen"]
                else:
                    fens[event["challenge"]["id"]]  = chess.STARTING_FEN
            else:
                client.bots.decline_challenge(event['challenge']['id'])
                lok("Don't accept challenge in", event['challenge']['speed'].capitalize(), ("because it's a rematch" if event['challenge']['id'] in colors else "because the bot don't play this speed"))
                if event['challenge']['id'] in colors:
                    client.bots.post_message(event['challenge']['id'], "I don't aceppt rematches (lot of bugs)")
        elif event['type'] == 'gameStart':
            game = Game(client, event['game']['id'], colors[event['game']['id']], fens[event["game"]["id"]])
            game.start()
        else:
            ldebug(event["type"], ":", event)
