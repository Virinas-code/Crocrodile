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
print(yukoo)

SPEEDS = ['classical', 'correspondence']

colorama.init()
def lok(*args):
    print(colorama.Style.RESET_ALL + colorama.Fore.GREEN + time.asctime(time.localtime()) + ":", *args)

def ldebug(*args):
    print(colorama.Style.RESET_ALL + colorama.Fore.MAGENTA + time.asctime(time.localtime()) + ":", *args)

def lerr(*args):
    print(colorama.Style.RESET_ALL + colorama.Fore.RED + time.asctime(time.localtime()) + ":", *args)

class Game(threading.Thread):
    def __init__(self, client, game_id, color, **kwargs):
        super().__init__(**kwargs)
        self.game_id = game_id
        self.client = client
        self.stream = client.bots.stream_game_state(game_id)
        self.current_state = next(self.stream)
        if color == 'white':
            self.my_turn = chess.WHITE
        else:
            self.my_turn = chess.BLACK
        if self.my_turn != chess.WHITE:
            self.game_state_change({'status':'started', 'moves':'0000 0000', 'btime':datetime.datetime(1970, 1, 1, 12)})
    def run(self):
        for event in self.stream:
            if event['type'] == 'gameState':
                self.game_state_change(event)
            elif event['type'] == 'chatLine':
                self.chat(event)
    def game_state_change(self, event):
        ldebug("game state change", event)
        if event['status'] == "started":
            lok("Game", self.game_id, ": moves", event['moves'])
            mvs = event['moves'].split(" ")
            board = chess.Board()
            for move in mvs:
                board.push(chess.Move.from_uci(move))
            ldebug("\n" + str(board))
            if board.turn != self.my_turn:
                lok("Calculating...")
                t = event['btime'].time()
                if (t.hour * 60 + t.minute) * 60 + t.second > 60:
                    score, best_move = yukoo.minimax(board, 3, board.turn, False)
                else:
                    score, best_move = yukoo.minimax(board, 2, board.turn, False)
                lok("Best move", best_move)
                lok("Score", score)
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
    token = file.read()

lok("Token is", token)

session = berserk.TokenSession(token)
client = berserk.Client(session)

lok("Connected to", client.account.get().get("title", "USER"), client.account.get().get("username", "Anonymous"))
lok("Waiting for challenges")
continue_loop = True
colors = {}
while continue_loop:
    for event in client.bots.stream_incoming_events():
        ldebug(event)
        if event['type'] == 'challenge':
            if event['challenge']['speed'] in SPEEDS:
                client.bots.accept_challenge(event['challenge']['id'])
                colors[event['challenge']['id']] = event['challenge']['color']
            else:
                client.bots.decline_challenge(event['challenge']['id'])
                lok("Don't accept challenge in", event['challenge']['speed'].capitalize())
        elif event['type'] == 'gameStart':
            game = Game(client, event['game']['id'], colors[event['game']['id']])
            game.start()
