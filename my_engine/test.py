import engine
import chess
import time
test_engine = engine.EngineBase("test","test")
board = chess.Board()
start = time.time()
for a in range(1):
    test_engine.minimax(board, 3, True, False)
end = time.time()
print(end-start)
