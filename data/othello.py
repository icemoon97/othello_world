import os
import pgn
import numpy as np
import random
from tqdm import tqdm
import time
import multiprocessing
import pickle
import psutil
import seaborn as sns
import itertools
from copy import copy, deepcopy
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

rows = list("abcdefgh")
columns = [str(_) for _ in range(1, 9)]

mask = np.zeros(64).reshape(8, 8)
mask[3, 3] = 1
mask[3, 4] = 1
mask[4, 3] = 1
mask[4, 4] = 1
mask = mask.astype(bool)

# Othello is a strategy board game for two players (Black and White), played on an 8 by 8 board. 
# The game traditionally begins with four discs placed in the middle of the board as shown below. Black moves first.
# W (27) B (28)
# B (35) W (36)

def permit(s):
    s = s.lower()
    if len(s) != 2:
        return -1
    if s[0] not in rows or s[1] not in columns:
        return -1
    return rows.index(s[0]) * 8 + columns.index(s[1])

def permit_reverse(integer):
    r, c = integer // 8, integer % 8
    return "".join([rows[r], columns[c]])

start_hands = [permit(_) for _ in ["d5", "d4", "e4", "e5"]]
eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

default_data = "othello_synthetic"

class Othello:
    def __init__(self, data_root=None, championship=False, wthor=False, n_games=1000, test_split=0.2, deduplicate=True, quiet=False):
        # data_root: folder of games dataset
        # championship: True if loading from .pgn files, otherwise assume .pickle sequences
        # wthor: True for WTHOR dataset, false for liveothello
        # n_games: -1 to load all games at data_root
        # test_split: test/train split for loaded games
        # deduplicate: True if games should be deduplicated (not necessary if games have been pre processed)
        # quiet: print statements
        self.board_size = 8 * 8
        self.total_games = n_games
        self.data_root = data_root if data_root is not None else default_data
        self.sequences = []
        self.results = []

        if not n_games:
            return
        if championship:
            self.load_championship(wthor)
            return
        
        bar = tqdm(os.listdir(f"./data/{self.data_root}"), disable=quiet)
        files_loaded = 0
        games_loaded = 0
        # loop through all files in data_root and load pickle sequences
        for f in bar:
            if not f.endswith(".pickle"):
                continue

            path = os.path.join(f"./data/{self.data_root}", f)
            with open(path, 'rb') as handle:
                b = pickle.load(handle)
                files_loaded += 1
                games_loaded += len(b)
                # if len(b) < 9e4:  # should be 1e5 each
                #     print(f"Warning: {path} only contains {len(b)} sequences")
                self.sequences.extend(b)

            process = psutil.Process(os.getpid())
            mem_gb = process.memory_info().rss / 2 ** 30
            bar.set_description(f"Mem Used: {mem_gb:.4} GB")

            # break if target number of games is reached
            if n_games > 0 and games_loaded > n_games:
                break

        if n_games > 0:
            self.sequences = self.sequences[:n_games]
        if not quiet: print(f"Loaded {games_loaded} from {files_loaded} files")

        if (deduplicate):
            seq = sorted(self.sequences)
            self.sequences = [k for k, _ in itertools.groupby(seq)]
            # reshuffle sequences after deduplicating
            random.shuffle(self.sequences)
            if not quiet: print(f"Deduplicating finished with {len(self.sequences)} games left")

        test_size = int(test_split * len(self.sequences))
        self.val = self.sequences[:test_size]
        self.sequences = self.sequences[test_size:]
        if not quiet: print(f"Using {len(self.sequences)} for training, {len(self.val)} for validation")

    def load_championship(self, wthor):
        criteria = lambda fn: fn.endswith("pgn") if wthor else fn.startswith("liveothello")

        if self.data_root is None:
            print("Must provide data_root with championship=True")
            return

        for fn in os.listdir(f"./data/{self.data_root}"):
            if not criteria(fn):
                continue

            with open(os.path.join(f"./data/{self.data_root}", fn), "r") as f:
                pgn_text = f.read()
            games = pgn.loads(pgn_text)
            num_ldd = len(games)
            processed = []
            res = []
            for game in games:
                tba = []
                for move in game.moves:
                    x = permit(move)
                    if x != -1:
                        tba.append(x)
                    else:
                        break
                if len(tba) != 0:
                    try:
                        rr = [int(s) for s in game.result.split("-")]
                    except:
                        # print(game.result)
                        # break
                        rr = [0, 0]
                    res.append(rr)
                    processed.append(tba)

            num_psd = len(processed)
            print(f"Loaded {num_psd}/{num_ldd} (qualified/total) sequences from {fn}")
            self.sequences.extend(processed)
            self.results.extend(res)

    def __len__(self, ):
        return len(self.sequences)
    
    def __getitem__(self, i):
        return self.sequences[i]

player_type_sorts = {
    # type 0: top left bias
    0: lambda x: (x // 8, x % 8),
    # type 1: top right bias
    1: lambda x: (x // 8, -(x % 8)),
    # type 2: bottom left bias
    2: lambda x: (-(x // 8), x % 8),
    # type 3: bottom right bias
    3: lambda x: (-(x // 8), -(x % 8)),
}

def get_synthetic_game(_):
    player_type = random.randint(0, 3)

    moves = []
    ob = OthelloBoardState()
    legal_moves = ob.get_valid_moves()
    while legal_moves:
        # choose based on player type
        if random.random() < 0.8:
            legal_moves.sort(key=player_type_sorts[player_type])
            next_step = legal_moves[0]
        # uniform random selection
        else:
            next_step = random.choice(legal_moves)

        # next_step = random.choice(legal_moves)
        moves.append(next_step)
        ob.update([next_step, ])
        legal_moves = ob.get_valid_moves()

    return (player_type, moves)

def generate_synthetic(n, data_root=None):
    data_root = data_root if data_root is not None else default_data
    data_root = f"./data/{data_root}"
    # create dir if it doesn't already exist
    if not os.path.exists(data_root):
        os.makedirs(data_root)
        print(f"Created dir {data_root}")

    seq = []
    num_proc = multiprocessing.cpu_count() # use all processors
    p = multiprocessing.Pool(num_proc)
    for possible in tqdm(p.imap(get_synthetic_game, range(n)), total=n):
        seq.append(possible)
    p.close()
    
    print(f"generated initial {len(seq)} games. Now deduplicating...")
    seq = [k for k, _ in itertools.groupby(sorted(seq))]
    # reshuffle sequences after deduplicating
    random.shuffle(seq)

    t_start = time.strftime("_%Y%m%d_%H%M%S")
    path = f"{data_root}/gen10e5_{t_start}.pickle"
    print(f"deduplicating done. saving {len(seq)} synthetic games to {path}")
    with open(path, 'wb') as handle:
        pickle.dump(seq, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

class OthelloBoardState():
    # 1 is black, -1 is white
    def __init__(self, board_size = 8):
        self.board_size = board_size * board_size
        board = np.zeros((8, 8))
        board[3, 4] = 1
        board[3, 3] = -1
        board[4, 3] = 1
        board[4, 4] = -1
        self.initial_state = board
        self.state = self.initial_state
        self.age = np.zeros((8, 8))
        self.next_hand_color = 1
        self.history = []

    def get_occupied(self, ):
        board = self.state
        tbr = board.flatten() != 0
        return tbr.tolist()
    def get_state(self, ):
        board = self.state + 1  # white 0, blank 1, black 2
        tbr = board.flatten()
        return tbr.tolist()
    def get_age(self, ):
        return self.age.flatten().tolist()
    def get_next_hand_color(self, ):
        return (self.next_hand_color + 1) // 2
    
    def update(self, moves, prt=False):
        # takes a new move or new moves and update state
        if prt:
            self.__print__()
        for _, move in enumerate(moves):
            self.umpire(move)
            if prt:
                self.__print__()

    def umpire(self, move):
        r, c = move // 8, move % 8
        assert self.state[r, c] == 0, f"{r}-{c} is already occupied!"
        occupied = np.sum(self.state != 0)
        color = self.next_hand_color
        tbf = []
        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        if len(tbf) == 0:  # means one hand is forfeited
            # print(f"One {color} move forfeited")
            color *= -1
            self.next_hand_color *= -1
            for direction in eights:
                buffer = []
                cur_r, cur_c = r, c
                while 1:
                    cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                    if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                        break
                    if self.state[cur_r, cur_c] == 0:
                        break
                    elif self.state[cur_r, cur_c] == color:
                        tbf.extend(buffer)
                        break
                    else:
                        buffer.append([cur_r, cur_c])
        if len(tbf) == 0:
            valids = self.get_valid_moves()
            if len(valids) == 0:
                assert 0, "Both color cannot put piece, game should have ended!"
            else:
                assert 0, "Illegal move!"
                
        self.age += 1
        for ff in tbf:
            self.state[ff[0], ff[1]] *= -1
            self.age[ff[0], ff[1]] = 0
        self.state[r, c] = color
        self.age[r, c] = 0
        self.next_hand_color *= -1
        self.history.append(move)
        
    def __print__(self, ):
        print("-"*20)
        print([permit_reverse(_) for _ in self.history])
        a = "abcdefgh"
        for k, row in enumerate(self.state.tolist()):
            tbp = []
            for ele in row:
                if ele == -1:
                    tbp.append("O")
                elif ele == 0:
                    tbp.append(" ")
                else:
                    tbp.append("X")
            # tbp.append("\n")
            print(" ".join([a[k]] + tbp))
        tbp = [str(k) for k in range(1, 9)]
        print(" ".join([" "] + tbp))
        print("-"*20)
        
    def tentative_move(self, move):
        # tentatively put a piece, do nothing to state
        # returns 0 if this is not a move at all: occupied or both player have to forfeit
        # return 1 if regular move
        # return 2 if forfeit happens but the opponent can drop piece at this place
        r, c = move // 8, move % 8
        if not self.state[r, c] == 0:
            return 0
        occupied = np.sum(self.state != 0)
        color = self.next_hand_color
        tbf = []
        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        if len(tbf) != 0:
            return 1
        else:  # means one hand is forfeited
            # print(f"One {color} move forfeited")
            color *= -1
            # self.next_hand_color *= -1
            for direction in eights:
                buffer = []
                cur_r, cur_c = r, c
                while 1:
                    cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                    if cur_r < 0  or cur_r > 7 or cur_c < 0 or cur_c > 7:
                        break
                    if self.state[cur_r, cur_c] == 0:
                        break
                    elif self.state[cur_r, cur_c] == color:
                        tbf.extend(buffer)
                        break
                    else:
                        buffer.append([cur_r, cur_c])
            if len(tbf) == 0:
                return 0
            else:
                return 2
        
    def get_valid_moves(self, ):
        regular_moves = []
        forfeit_moves = []
        for move in range(64):
            x = self.tentative_move(move)
            if x == 1:
                regular_moves.append(move)
            elif x == 2:
                forfeit_moves.append(move)
            else:
                pass
        if len(regular_moves):
            return regular_moves
        elif len(forfeit_moves):
            return forfeit_moves
        else:
            return []
 
    def get_gt(self, moves, func, prt=False):
        # takes a new move or new moves and update state
        container = []
        if prt:
            self.__print__()
        for _, move in enumerate(moves):
            self.umpire(move)
            container.append(getattr(self, func)())  
            # to predict first y, we need already know the first x
            if prt:
                self.__print__()
        return container

if __name__ == "__main__":
    # o = Othello(data_root="othello_championship", championship=True)
    # o = Othello(data_root="othello_synthetic", n_games=-1)
    
    pass

