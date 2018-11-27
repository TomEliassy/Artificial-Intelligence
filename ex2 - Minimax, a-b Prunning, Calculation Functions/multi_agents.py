import random

import numpy as np
import abc
import math
import util
import copy
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score

        "*** YOUR CODE HERE ***"
        if action == Action.UP:
            return 0
        return score + np.count_nonzero(board==0) + max_tile


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        score, action = min_max(game_state, self.evaluation_function, self.depth, 0)
        print("The Minimax value for tree with depth " + str(self.depth) + " is:" + str(score))
        return action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        score_and_action = self.alpha_beta(game_state, self.depth, -math.inf , math.inf, 0 ,Action.STOP)
        return score_and_action.action

    def alpha_beta(self, game_state, depth, alpha, beta, agent, action):
        legal_moves = game_state.get_legal_actions(agent)
        if depth == 0 or len(legal_moves) == 0:
            return ScoreAndAction(self.evaluation_function(game_state), action)

        if agent == 0:
            v = ScoreAndAction(-math.inf, Action.STOP)
            for move in legal_moves:
                son = self.alpha_beta(game_state.generate_successor(agent, move), depth, alpha, beta, math.fabs(agent -1), move)
                if son > v:
                    v = ScoreAndAction(son.score, move)
                alpha = max(alpha, v.score)
                if beta <= alpha:
                    break
            return v
        if agent == 1:
            v = ScoreAndAction(math.inf, Action.STOP)
            for move in legal_moves:
                son = self.alpha_beta(game_state.generate_successor(agent, move), depth -1, alpha, beta, math.fabs(agent -1), move)
                if son < v:
                    v = ScoreAndAction(son.score, move)
                beta = min(beta, v.score)
                if beta <= alpha:
                    break
            return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        score, action = self.expecti_max(game_state, self.depth, 0)
        #score, action = self.perform_expectimax(game_state, self.evaluation_function, self.depth, 0)
        return action

    def expecti_max(self,game_state, depth, agent):
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions(agent)
        if depth == 0 or len(legal_moves) == 0:
            return self.evaluation_function(game_state), None

        else:
            if agent == 0:
                best = (-1, Action.STOP)
                for action in legal_moves:
                    val, ret = self.expecti_max(game_state.generate_successor(agent, action), depth, math.fabs(agent - 1))
                    if val > best[0]:
                        best = (val, action)
                return best[0], best[1]
            else:
                acc_val = 0
                for action in legal_moves:
                    val, ret = self.expecti_max(game_state.generate_successor(agent, action), depth - 1, math.fabs(agent - 1))
                    acc_val += val
                mean = acc_val/len(legal_moves)
                return mean, None

    def perform_expectimax (self,game_state,evaluation_function, depth, agent):
        # Stop condition
        if depth == 0:
            return evaluation_function(game_state), None

        # Recursive call
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions(agent)
        best = (-1, Action.STOP)
        for action in legal_moves:
            # Consider the opponent react to the moves
            temp_state = game_state.generate_successor(agent , action)
            opponent_moves = temp_state.get_opponent_legal_actions()
            #num_of_opp_moves = len(opponent_moves)
            #rndm = random.randint(0 , num_of_opp_moves - 1)
            #opponent_move = opponent_moves[rndm]
            #best_for_opponent = (math.inf, Action.STOP) # saves the best move for the opp. REGARDING the current action
            acc_val = 0
            for opp_move in opponent_moves:
                val, act = self.perform_expectimax(temp_state.generate_successor(math.fabs(agent -1), opp_move), evaluation_function, depth -1, agent)
                acc_val += val
            mean = acc_val/len(opponent_moves)
                #if val < best_for_opponent[0]:
                    #best_for_opponent = (val , opp_move)
            if mean > best[0]:
                best = (mean, action)

        return best[0], best[1]


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    return perform_evaluate(current_game_state)


def find_smoothness (current_game_state):
     penalty = 0
     for item in neighbors_map.items():
        for neighbor in item[1]:
            penalty += math.fabs(current_game_state.board[neighbor[0] , neighbor[1]] - current_game_state.board[item[0][0] , item[0][1]])

     return penalty


neighbors_map = {(0,0):[(0,1) , (1,0)] , (0,1):[(0,2), (1,1) , (0,0)] ,(0,2):[(1,2), (0,1) , (0,3)], (0,3):[(0,2), (1,3)],
                 (1,0):[(0,0) , (1,1) , (2,0)] , (1,1):[(0,1) , (1,2), (2,1) , (1,0)] , (1,2):[(1,1) , (0,2), (1,3), (2,2)] , (1,3):[(0,3) , (2,3) , (1,2)] ,
                 (2,0): [(1,0), (2,1), (3,0),],(2,1):[(2,0),(2,2),(1,1),(3,1)],(2,2):[(2,1),(2,3),(1,2),(3,2)], (2,3): [(2,2),(1,3),(3,3)] ,
                 (3,0) : [(2,0),(3,1)] , (3,1) : [(3,0), (2,1), (3,2)], (3,2): [(3,1),(2,2),(3,3)], (3,3): [(3,2), (2,3)]}


gradient_mat = np.array([[6,5,4,3],[5,4,3,2],[4,3,2,1],[3,2,1,0]])

#gradient_mat = np.array([[4**6,4**5,4**4,4**3],[4**5,4**4,4**3,4**2],[4**4,4**3,4**2,4**1],[4**3,4**2,4**1,4**0]])

#gradient_mat = np.array([[4**15,4**14, 4**13,4**12],[4**8, 4**9, 4**10, 4**11],[4**7, 4**6, 4**5, 4**4],[4**0,4**1,4**2,4**3]])

def is_mono_increasing_right_left(board, side, up):
    board_to_check = copy.deepcopy(board)
    if up:
        board_to_check = np.swapaxes(board_to_check, 0, 1)
    res = 1
    for array in board_to_check:
        arr = array[::side]
        i = 0
        while i < len(arr) -2:
            if arr[i] > arr[i+1]:
                res -= 0.25
                i = len(arr)
            i += 1
    return res

def min_max(game_state, evaluation_function, depth, agent):
    # Stop condition
    if depth == 0:
        return evaluation_function(game_state), None

    # Recursive call
    # Collect legal moves and successor states
    legal_moves = game_state.get_legal_actions(agent)
    best = (-1, Action.STOP)
    for action in legal_moves:
        # Consider the opponent react to the moves
        temp_state = game_state.generate_successor(agent , action)
        opponent_moves = temp_state.get_opponent_legal_actions()
        best_for_opponent = (math.inf, Action.STOP) # saves the best move for the opp. REGARDING the current action
        for opp_move in opponent_moves:
            val, act = min_max(temp_state.generate_successor(math.fabs(agent -1), opp_move), evaluation_function, depth -1, agent)
            if val < best_for_opponent[0]:
                best_for_opponent = (val , opp_move)
        if best_for_opponent[0] > best[0]:
            best = (best_for_opponent[0], action)

    return best[0], best[1]


class ScoreAndAction:
    def __init__(self, score, action):
        self.score = score
        self.action = action

    def __gt__(self, other):
        return self.score > other.score

    def __lt__(self, other):
        return self.score < other.score


# Abbreviation
better = better_evaluation_function
gradient_mat = np.array([6,5,4,3,5,4,3,2,4,3,2,1,3,2,1,0])
#gradient_mat = np.array([2**6,2**5,2**4,2**3,2**5,2**4,2**3,2**2,2**4,2**3,2**2,2**1,2**3,2**2,2**1,2**0])
def perform_evaluate (state):
    # #of non-zero tiles
    non_zero = 100 * np.count_nonzero(state.board==0) #8

    # smoothness(monotonicity)
    smooth_val = find_smoothness(state)

    # max tile - 0.125

    # merges
    merges =  100 * find_merges(state)

    #shape
    shape_val = 10 * eval_grad(state)

    punish = 0
    if state.board[0][0] != state.max_tile :
        punish = -100000000

    #print ("empty: ", non_zero, "snake val: " , shape_val , "mono: ", smooth_val , "merges: ", merges , "SCORE: " , state.score , "max tile: " , state.max_tile)

    return    non_zero -smooth_val + 0.5 * state.max_tile + merges + shape_val + punish


def find_merges(state):
    merges = 0
    for item in neighbors_map.items():
        for neighbor in item[1]:
            if (state.board[item[0][0] , item[0][1]]) == (state.board[neighbor[0] , neighbor[1]]):
                merges += 1
    return merges

def eval_smoothness(state):
    '''
    Sum the difference between each pair of adjacent tiles. Smaller is better so we take its reciprocal.
    '''
    def row_smoothness(board):
        return sum([abs(r[c] - r[c+1]) for r in board for c in range(len(r) - 1)])

    return 1 / (row_smoothness(state.board) + row_smoothness(zip(*state.board)) + 1)

SNAKE_WEIGHTS = [4 ** 15, 4 ** 14, 4 ** 13, 4 ** 12,
                 4 ** 8, 4 ** 9, 4 ** 10, 4 ** 11,
                 4 ** 7, 4 ** 6, 4 ** 5, 4 ** 4,
                 4 ** 0, 4 ** 1, 4 ** 2, 4 ** 3]
def eval_snake(state):
    '''
    Linear combination of all squares' values.
    Inspired by Hadi Pouransari & Saman Ghili's "AI algorithms for the game 2048".
    '''
    arr = (np.array(state.board)).flatten()
    res = arr.dot(SNAKE_WEIGHTS)
    return res

def eval_grad(state):
    '''
    Linear combination of all squares' values.
    Inspired by Hadi Pouransari & Saman Ghili's "AI algorithms for the game 2048".
    '''
    arr = (np.array(state.board)).flatten()
    res = arr.dot(gradient_mat)
    return res
