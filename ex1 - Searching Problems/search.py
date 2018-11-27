"""
In search.py, you will implement generic search algorithms
"""

import util
import heapq
import copy
import math


Empty = 0


Player1 = 1

move = 1


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal stateif problem is not None:
            return problem.get_cost_of_actions(problem.moves_map[state])
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()




def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    if problem.is_goal_state(problem.get_start_state()):
        return []

    # The start state is not a goal state
    successors = problem.get_successors(problem.get_start_state())

    # Fringe
    fringe = util.Stack()

    for stat in successors:
        fringe.push((stat[0], [stat[move]]))

    # Explored Set
    explored_set = []
    # Loop
    while not fringe.isEmpty():

        stat = fringe.pop()

        explored_set.append(stat[0])

        # Case of goal state
        if problem.is_goal_state(stat[0]):
            return stat[move]  # we need to return only the moves

        #Else
        for next_stat in problem.get_successors(stat[0]):
            moves_so_far = copy.deepcopy(stat[move])
            moves_so_far.append(next_stat[move])
            next_stat = (next_stat[0], moves_so_far)
            # Make sure that next ISN'T  in Explored_Set and ISN'T  in the fringe
            if (next_stat[0] not in explored_set) and (next_stat not in fringe.list):
                fringe.push(next_stat)

    # In case that the fringe is empty
    return []



def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    if problem.is_goal_state(problem.get_start_state()):
        return []

    # The start state is not a goal state
    successors = problem.get_successors(problem.get_start_state())

    # Fringe
    fringe = util.Queue()

    for stat in successors:
        fringe.push((stat[0], [stat[move]]))

    # Explored Set - contains board states
    explored_set = []

    #Loop
    while not fringe.isEmpty():

        stat = fringe.pop()
        explored_set.append(stat[0])
        # Case of goal state
        if problem.is_goal_state(stat[0]):
            return stat[move]
        #Else
        for next_stat in problem.get_successors(stat[0]):
            moves_so_far = copy.deepcopy(stat[move])
            moves_so_far.append(next_stat[move])
            next_stat = (next_stat[0], moves_so_far)
            # Make sure that next ISN'T  in Explored_Set and ISN'T  in the fringe
            if (next_stat[0] not in explored_set) and (next_stat not in fringe.list):
                if problem.is_goal_state(next_stat[0]):
                    return moves_so_far
                fringe.push(next_stat)

    # In case that the fringe is empty
    return []

def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    start = problem.get_start_state()

    #In case that the start state is also the goal state
    if problem.is_goal_state(start):
        return []

    # Fringe
    fringe = util.PriorityQueue()
    moves_map = {}

    successors = problem.get_successors(start)

    for stat in successors:
        moves_map.update({stat[0]: [stat[move]]})
        fringe.push(stat[0], stat[2])

    #Explored Set - contains states of the board
    explored_set = []

    #Loop
    while not fringe.isEmpty():
        node = fringe.pop()

        # Case of goal state
        if problem.is_goal_state(node):
            return moves_map[node]

        # Else
        for next_stat in problem.get_successors(node):
            moves_so_far = copy.deepcopy(moves_map[node])
            moves_so_far.append(next_stat[move])
            childs_cost = problem.get_cost_of_actions(moves_so_far)
            # Make sure that next ISN'T  in Explored_Set and ISN'T  in the fringe
            stat_in_fringe = [item for item in fringe.heap if item[1] == next_stat[0]]
            if (next_stat[0] not in explored_set) and len(stat_in_fringe) == 0:
                fringe.push(next_stat[0], childs_cost)
                explored_set.append(next_stat[0])
                moves_map.update({next_stat[0]: moves_so_far})
            # In Case that  next_stat is in the fringe already with higher cost
            elif len(stat_in_fringe) > 0 and stat_in_fringe[0][0] > childs_cost:
                fringe.heap.remove(stat_in_fringe)
                fringe.push(next_stat[0], childs_cost)
                moves_map.update({next_stat[0]: moves_so_far})

            heapq.heapify(fringe.heap)

    # In case that the fringe is empty
    return []



def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    def huristic(state):
        if problem is not None:
            return problem.moves_map[state][2]
        return 0
    return huristic


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    "*** YOUR CODE HERE ***"
    start = problem.get_start_state()
    moves_map = problem.moves_map

    # In case that the start state is also the goal state
    if problem.is_goal_state(start):
        return []

    moves_map.update({start: None})
    # Fringe
    fringe = util.PriorityQueueWithFunction(heuristic(0, problem))

    successors = problem.get_successors(start)

    for stat in successors:
        moves_map.update({stat[0]: (stat[1], start, stat[2])})
        fringe.push(stat[0])

    # Explored Set
    explored_set = []

    # Loop
    while not fringe.isEmpty():
        node = fringe.pop()
        print(problem.expanded)

        # Case of goal state
        if problem.is_goal_state(node):
            return return_backtrace(node, start, moves_map)

        # Else
        else:
            for next_stat in problem.get_successors(node):
                cost = moves_map[node][2] + next_stat[2]
                #Make sure that next ISN'T  in Explored_Set and ISN'T  in the fringe
                stat_in_fringe = [item for item in fringe.heap if next_stat[0] == item[1]]
                if (next_stat[0] not in explored_set) and len(stat_in_fringe) == 0:
                    moves_map.update({next_stat[0]: (next_stat[1], node, cost)})
                    fringe.push(next_stat[0])
                    explored_set.append(next_stat[0])

    return []


def return_backtrace(begin_board, start, game_tree):
    backtrace = []

    curr_board = begin_board

    while curr_board != start:
        backtrace.append(game_tree[curr_board][0])
        curr_board = game_tree[curr_board][1]  # curr_board = prev_state[0]

    backtrace.reverse()
    return backtrace

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
