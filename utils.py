
from constants import *
import pygame
import copy
import sys
import random
import numpy as np
from collections import deque
import heapq

#pygame settings 
pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Smart XO")
screen.fill(BG_color)


class Board:
    def __init__(self):
        self.squares = np.zeros((ROWS, COLS))
        self.empty_sqrs = self.squares
        self.marked_sqrs = 0

    def final_state(self, show = False):
        """
        return --> 0 if no win 
        return --> 1 if player 1 wins 
        return --> 2 if player 2 wins 

        """

        #vertical wins 
        for col in range(COLS):
            if self.squares[0][col] == self.squares[1][col] == self.squares[2][col] != 0:
                if show:
                    color = CIRC_COLOR if self.squares[0][col] == 2 else CROSS_COLOR
                    iPos = (col * SQSIZE + SQSIZE // 2, 20)
                    fPos = (col * SQSIZE + SQSIZE // 2, HEIGHT - 20)
                    pygame.draw.line(screen, color, iPos, fPos, LINE_WIDTH) 
                return self.squares[0][col]
            
        #horizontal wins 
        for row in range(ROWS):
            if self.squares[row][0] == self.squares[row][1] == self.squares[row][2] != 0:
                if show:
                    color = CIRC_COLOR if self.squares[row][0] == 2 else CROSS_COLOR
                    iPos = (20, row * SQSIZE + SQSIZE // 2)
                    fPos = (WIDTH - 20, row * SQSIZE + SQSIZE // 2)
                    pygame.draw.line(screen, color, iPos, fPos, LINE_WIDTH)
                return self.squares[row][0]

        # desc diagonal 
        if self.squares[0][0] == self.squares[1][1] == self.squares[2][2] != 0:
            if show:
                color = CIRC_COLOR if self.squares[1][1] == 2 else CROSS_COLOR
                iPos = (20, 20)
                fPos = (WIDTH - 20, HEIGHT - 20)
                pygame.draw.line(screen, color, iPos, fPos, CROSS_WIDTH)

            return self.squares[1][1]
        
        if self.squares[2][0] == self.squares[1][1] == self.squares[0][2] != 0:
            if show:
                color = CIRC_COLOR if self.squares[1][1] == 2 else CROSS_COLOR
                iPos = (20, HEIGHT - 20)
                fPos = (WIDTH - 20, 20)
                pygame.draw.line(screen, color, iPos, fPos, CROSS_WIDTH)
            return self.squares[1][1]
        
        # no win yet
        return 0


    def mark_sqr(self, row, col, player):
        self.squares[row][col] = player    
        self.marked_sqrs += 1

    def empty_sqr(self, row, col):
        return self.squares[row][col] == 0
    
    def get_empty_sqrs(self):
        empty_sqrs = []
        for row in range(ROWS):
            for col in range(COLS):
                if self.empty_sqr(row, col):
                    empty_sqrs.append((row, col))
        return empty_sqrs            

    def isfull(self):
        return self.marked_sqrs == 9
    
    def isempty(self):
        return self.empty_sqrs == 0
    
    def is_empty(self, row, col):
        return self.squares[row][col]

    def clear_sqr(self, row, col):
        self.squares[row][col] = 0

    def evaluate(self, player):
        """
        Evaluate the board state for the given player.

        Return a higher score if the board state is favorable for the player and
        a lower score if it's unfavorable.

        The score can be positive for favorable states, negative for unfavorable states,
        and 0 for neutral states.
        """
        score = 0

        # Check for winning positions
        for row in range(3):
            if np.array_equal(self.squares[row, :], np.array([player, player, player])):
                score += 10  # Favorable for the player

            if np.array_equal(self.squares[:, row], np.array([player, player, player])):
                score += 10  # Favorable for the player

        if np.array_equal(np.diag(self.squares), np.array([player, player, player])) or np.array_equal(np.diag(np.fliplr(self.squares)), np.array([player, player, player])):
            score += 10  # Favorable for the player

        # Check for blocking the opponent
        opponent = 3 - player  # Assuming players are represented by 1 and 2

        for row in range(3):
            if np.count_nonzero(self.squares[row, :] == opponent) == 2 and np.count_nonzero(self.squares[row, :] == 0) == 1:
                score += 5  # Block the opponent

            if np.count_nonzero(self.squares[:, row] == opponent) == 2 and np.count_nonzero(self.squares[:, row] == 0) == 1:
                score += 5  # Block the opponent

        if np.count_nonzero(np.diag(self.squares) == opponent) == 2 and np.count_nonzero(np.diag(self.squares) == 0) == 1:
            score += 5  # Block the opponent

        if np.count_nonzero(np.diag(np.fliplr(self.squares)) == opponent) == 2 and np.count_nonzero(np.diag(np.fliplr(self.squares)) == 0) == 1:
            score += 5  # Block the opponent

        # Control the center
        if self.squares[1, 1] == player:
            score += 2  # Favorable for the player

        return score    
    
    def __lt__(self, other):
        # Define how to compare instances of the Board class
        # For example, you can compare based on some evaluation score or heuristic
        # Return True if this instance is "less than" the other instance

        # Replace the following with your actual comparison logic
        return self.evaluate(1) < other.evaluate(1)

class AI:
    def __init__(self, level = 1, player = 2) -> None:
        self.level = level
        self.player = player
        self.algo = 2

    def rnd(self, board):
        empty_sqrs = board.get_empty_sqrs()
        idx = random.randrange(0, len(empty_sqrs))
        return empty_sqrs[idx]
    
    def minimax(self, board, maximizing):
        #terminal case
        case = board.final_state()

        #player 1 wins
        if case == 1:
            return 1, None
        
        #player 2 wins 
        if case == 2:
            return -1, None
        
        #draw
        elif board.isfull():
            return 0, None

        if maximizing:
            max_eval = -100
            best_move = None
            empty_sqrs = board.get_empty_sqrs()

            for (row, col) in empty_sqrs:
                temp_baord = copy.deepcopy(board)
                temp_baord.mark_sqr(row, col, self.player)
                eval = self.minimax(temp_baord, False)[0]
                if eval > max_eval:
                    max_eval = eval
                    best_move = (row, col)
            return  max_eval, best_move

        elif not maximizing:
            min_eval = 100
            best_move = None
            empty_sqrs = board.get_empty_sqrs()

            for (row, col) in empty_sqrs:
                temp_baord = copy.deepcopy(board)
                temp_baord.mark_sqr(row, col, self.player)
                eval = self.minimax(temp_baord, True)[0]
                if eval < min_eval:
                    min_eval = eval
                    best_move = (row, col)
        return  min_eval, best_move      
    
    def dfs1(self, board, maximizing):
        stack = []
        stack.append((board, None, None))

        while stack:
            current_board, move, _ = stack.pop()

            case = current_board.final_state()
            if case == 1:
                return 1, move
            if case == 2:
                return -1, move
            if current_board.isfull():
                return 0, move

            empty_sqrs = current_board.get_empty_sqrs()

            for (row, col) in empty_sqrs:
                new_board = copy.deepcopy(current_board)
                new_board.mark_sqr(row, col, self.player)
                stack.append((new_board, (row, col), None))

        return 0, move  # Handle the case where no terminal state is reached
    
    def choose_best_move(self, board, maximizing):
        stack = []
        stack.append((board, None, None))

        best_move = None
        best_score = float('-inf') if maximizing else float('inf')

        while stack:
            current_board, move, _ = stack.pop()

            case = current_board.final_state()
            if case == 1:
                return 1, move
            if case == 2:
                return -1, move
            if current_board.isfull():
                return 0, move

            empty_sqrs = current_board.get_empty_sqrs()

            for (row, col) in empty_sqrs:
                new_board = copy.deepcopy(current_board)
                new_board.mark_sqr(row, col, self.player)
                stack.append((new_board, (row, col), None))

        return 0, best_move  # Handle the case where no terminal state is reached

    def dfs(self, board, maximizing):
        _, best_move = self.choose_best_move(board, maximizing)
        return 1 if maximizing else -1, best_move

    def bfs(self, board, maximizing):
        queue = deque()
        queue.append((board, None, None))

        best_move = None
        best_score = float('-inf') if maximizing else float('inf')

        while queue:
            current_board, move, _ = queue.popleft()

            case = current_board.final_state()
            if case == 1:
                return 1, move
            if case == 2:
                return -1, move
            if current_board.isfull():
                return 0, move

            empty_sqrs = current_board.get_empty_sqrs()

            for (row, col) in empty_sqrs:
                new_board = copy.deepcopy(current_board)
                new_board.mark_sqr(row, col, self.player)
                queue.append((new_board, (row, col), None))

        return 0, best_move  # Handle the case where no terminal state is reached
    
    def ucs(self, board, maximizing):
        priority_queue = []
        heapq.heappush(priority_queue, (0, board, None))

        best_move = None
        best_score = float('-inf') if maximizing else float('inf')

        while priority_queue:
            current_cost, current_board, move = heapq.heappop(priority_queue)

            case = current_board.final_state()
            if case == 1:
                return 1, move
            if case == 2:
                return -1, move
            if current_board.isfull():
                return 0, move

            empty_sqrs = current_board.get_empty_sqrs()

            for (row, col) in empty_sqrs:
                new_board = copy.deepcopy(current_board)
                new_board.mark_sqr(row, col, self.player)
                move_cost = 1  # You can adjust the cost as needed
                evaluation_score = new_board.evaluate(self.player)  # Calculate the evaluation score

                # Push onto the priority queue with evaluation score
                heapq.heappush(priority_queue, (current_cost + move_cost + evaluation_score, new_board, (row, col)))

        return 0, best_move  # Handle the case where no terminal state is reached
    

    def dfs_ids(self, board, maximizing, depth_limit):
        case = board.final_state()
        if case == 1:
            return 1.0, None
        if case == 2:
            return -1.0, None
        if board.isfull() or depth_limit == 0:
            return 0.0, None

        best_move = None
        best_score = float('-inf') if maximizing else float('inf')

        empty_sqrs = board.get_empty_sqrs()
        for (row, col) in empty_sqrs:
            new_board = copy.deepcopy(board)
            new_board.mark_sqr(row, col, self.player)
            move_score, _ = self.dfs_ids(new_board, not maximizing, depth_limit - 1)

            if (maximizing and move_score > best_score) or (not maximizing and move_score < best_score):
                best_score = move_score
                best_move = (row, col)

        return best_score, best_move
    
    def ids(self, board, maximizing, depth_limit):
        best_move = None
        best_score = float('-inf') if maximizing else float('inf')

        empty_sqrs = board.get_empty_sqrs()
        for (row, col) in empty_sqrs:
            new_board = copy.deepcopy(board)
            new_board.mark_sqr(row, col, self.player)
            move_score, _ = self.dfs_ids(new_board, not maximizing, depth_limit - 1)
            if (maximizing and move_score > best_score) or (not maximizing and move_score < best_score):
                best_score = move_score
                best_move = (row, col)
                
        return best_score, best_move
    
    def greedy_search(self, board, maximizing):
        priority_queue = []
        heapq.heappush(priority_queue, (0, board, None))

        best_move = None
        best_score = float('-inf') if maximizing else float('inf')

        while priority_queue:
            _, current_board, move = heapq.heappop(priority_queue)

            case = current_board.final_state()
            if case == 1:
                return 1, move
            if case == 2:
                return -1, move
            if current_board.isfull():
                return 0, move

            empty_sqrs = current_board.get_empty_sqrs()

            for (row, col) in empty_sqrs:
                new_board = copy.deepcopy(current_board)
                new_board.mark_sqr(row, col, self.player)
                evaluation_score = new_board.evaluate(self.player)  # Calculate the evaluation score

                # Push onto the priority queue with evaluation score
                heapq.heappush(priority_queue, (evaluation_score, new_board, (row, col)))

        return 0, best_move  # Handle the case where no terminal state is reached
    
    def a_star_search(self, board, maximizing):
        priority_queue = []
        heapq.heappush(priority_queue, (0, 0, board, None))

        best_move = None
        best_score = float('-inf') if maximizing else float('inf')

        while priority_queue:
            _, current_cost, current_board, move = heapq.heappop(priority_queue)

            case = current_board.final_state()
            if case == 1:
                return 1, move
            if case == 2:
                return -1, move
            if current_board.isfull():
                return 0, move

            empty_sqrs = current_board.get_empty_sqrs()

            for (row, col) in empty_sqrs:
                new_board = copy.deepcopy(current_board)
                new_board.mark_sqr(row, col, self.player)
                move_cost = 1  # You can adjust the cost as needed
                evaluation_score = new_board.evaluate(self.player)  # Calculate the evaluation score

                # Push onto the priority queue with both evaluation score and total cost
                heapq.heappush(priority_queue, (evaluation_score + current_cost + move_cost, current_cost + move_cost, new_board, (row, col)))

        return 0, best_move  # Handle the case where no terminal state is reached

    def eval(self, main_board):
        if self.level == 0:
            print("level is zer0")
            eval = "random"
            move = self.rnd(main_board)
        else:
            print("Smart AI is playing .....")
            if self.algo == 2:
                print("minimax is used")
                eval, move = self.minimax(main_board, False)
            elif self.algo == 3:    
                print("dfs is used ")
                eval, move = self.dfs(main_board, False)
            elif self.algo == 4:    
                print("bfs is used")
                eval, move = self.bfs(main_board, False)    
            elif self.algo == 5:    
                print("ucs is used")
                eval, move = self.ucs(main_board, False)    
            elif self.algo == 6:    
                print("ids is used")
                eval, move = self.ids(main_board, False, 9)    
            elif self.algo == 7:    
                print("greedy is used")
                eval, move = self.greedy_search(main_board, False)          
            elif self.algo == 8:    
                print("greedy is used")
                eval, move = self.a_star_search(main_board, False)         
        
        print(f"AI marked a square in pos {move} and the eval is {eval}")
        return move

class Game:
    def __init__(self): 
        self.board = Board()
        self.ai = AI()
        self.player = 1 
        self.gamemode = 'ai'
        self.running = True
        self.show_lines()

    def make_move(self, row, col):
        self.board.mark_sqr(row, col, self.player)
        self.draw_fig(row, col)
        self.next_turn()


    def show_lines(self):
        screen.fill(BG_color)
        #vertical lines
        pygame.draw.line(screen, LINE_COLOR, (SQSIZE,0) , (SQSIZE, HEIGHT), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (WIDTH-SQSIZE,0) , (WIDTH-SQSIZE, HEIGHT), LINE_WIDTH)

        #horizontal lines
        pygame.draw.line(screen, LINE_COLOR, (0, SQSIZE) , (WIDTH, SQSIZE), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (0, HEIGHT-SQSIZE) , (WIDTH, HEIGHT-SQSIZE), LINE_WIDTH)
    
    def draw_fig(self, row, col):
        if self.player == 1:
            #draw cross
            start_desc = (col * SQSIZE + OFFSET, row * SQSIZE + OFFSET)
            end_desc = (col * SQSIZE + SQSIZE - OFFSET, row * SQSIZE + SQSIZE - OFFSET)
            pygame.draw.line(screen, CROSS_COLOR, start_desc, end_desc, CROSS_WIDTH)
            # asc line
            start_asc = (col * SQSIZE + OFFSET, row * SQSIZE + SQSIZE - OFFSET)
            end_asc = (col * SQSIZE + SQSIZE - OFFSET, row * SQSIZE + OFFSET)
            pygame.draw.line(screen, CROSS_COLOR, start_asc, end_asc, CROSS_WIDTH)

        elif self.player == 2:
            #draw circle
            center = (col * SQSIZE + SQSIZE //2, row * SQSIZE + SQSIZE //2)  
            pygame.draw.circle(screen, CIRC_COLOR, center, RADIUS, CIRC_WIDTH)

    def change_gamemode(self):
        self.gamemode = "ai" if self.gamemode == "pvp" else "pvp"

    def reset(self):
        self.__init__()    

    def isover(self):
        return self.board.final_state(show=True) != 0 or self.board.isfull()

    def next_turn(self):
        self.player = self.player % 2 + 1    
        