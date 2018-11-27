import copy
import random

import numpy as np

from board import Board
from search import SearchProblem
import search
import util
import math

Free = -1

Empty = 0

DEST = 1

ORIGIN = 2

DISTANCE = 0

PLAYER_1 = 0


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)



#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.starting_point = starting_point
        self.board_w = board_w - 1
        self.board_h = board_h - 1
        self.corners = [(0, 0), (0, self.board_h), (self.board_w, self.board_h), (self.board_w, 0)]
        self.moves_map = {}

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        return (state.state[0][self.board_h] != Free
                and state.state[self.board_w][self.board_h] != Free
                and state.state[self.board_w][0] != Free)

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        total_cost = 0
        for action in actions:
            total_cost += action.piece.num_tiles
        return total_cost


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    "*** YOUR CODE HERE ***"
    converted_problem = BlokusCoverProblem(problem.board.board_w,
                                           problem.board.board_h,
                                           problem.board.piece_list,
                                           problem.starting_point,
                                           problem.corners)
    converted_problem.moves_map = problem.moves_map
    return blokus_cover_heuristic(state, converted_problem)


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.starting_point = starting_point
        self.moves_map = {}

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        for target in self.targets:
            if state.state[target[0]][target[1]] == Free:
                return False
        return True


    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        total_cost = 0
        for action in actions:
            total_cost += action.piece.get_num_tiles()
        return total_cost


def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    def heuristic(state):
        targets = [target for target in problem.targets if state.state[target[0]][target[1]] == Free]
        if len(get_optional_targets(copy.deepcopy(targets), state, 0)) != len(targets):
            return math.inf
        acc_chess_dist = 0
        available_locations = get_optional_locations(state, 0)
        chess_distances = get_minimal_chess_distances(available_locations, targets)

        for dist in chess_distances:
            acc_chess_dist += dist[DISTANCE]

        total_cost = acc_chess_dist + problem.moves_map[state][2]
        return total_cost

    return heuristic


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0), dest=(0, 0)):
        self.expanded = 0
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.dest = dest
        self.starting_point = starting_point
        self.moves_map = {}

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        return state.state[self.dest[0]][self.dest[1]] != Free

    def get_successors(self, state):
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]


    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        total_cost = 0
        for action in actions:
            total_cost += action.piece.get_num_tiles()
        return total_cost

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace

        return backtrace
        """
        "*** YOUR CODE HERE ***"
        current_state = self.board.__copy__()
        backtrace = []
        targets_cpy = self.targets.copy()

        while len(targets_cpy) != 0:
            # ASSUMING PLAYER 1 ONLY!!!
            closest_tuple = calculate_min_chess_dist_to_tiles(targets_cpy, get_optional_locations(self.board, PLAYER_1))
            dest = closest_tuple[DEST]

            targets_cpy.remove(dest)
            appropriate_blokus_problem = ClosestLocationSearch(self.board.board_w, self.board.board_h,
                                                                  current_state.piece_list, self.starting_point,
                                                                  targets_cpy, dest)
            appropriate_blokus_problem.board = current_state
            appropriate_blokus_problem.moves_map = self.moves_map
            actions = search.a_star_search(appropriate_blokus_problem, suboptimal_heuristic)
            backtrace += actions
            self.expanded += appropriate_blokus_problem.expanded

            # This loop performes actions on current_state
            for action in actions:
                current_state.add_move(PLAYER_1 , action)


            # Try to eliminate another target(s)
            for target in targets_cpy:
                if current_state.state[target[0]][target[1]] != -1:
                    targets_cpy.remove(target)

        return backtrace


def suboptimal_heuristic(state, problem):
    dest = problem.dest
    targets = problem.targets

    def huristic(state):
        min_chess_dist = math.inf

        for available in get_optional_locations(state , PLAYER_1):
            chess_dist = get_chess_dist(available , dest) + 1
            if chess_dist < min_chess_dist :
                min_chess_dist = chess_dist

        if state.state[dest[0]][dest[1]] == PLAYER_1:
            min_chess_dist = 0

        targets_to_check = copy.deepcopy(targets)
        if len(get_optional_targets(targets_to_check, state, PLAYER_1)) != len(targets):
            if set(covered_by_mistake(state, copy.deepcopy(targets))) != set(targets_to_check):
                min_chess_dist = math.inf


        return min_chess_dist

    return huristic

class MiniContestSearch :
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        self.expanded = 0
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.starting_point = starting_point
        self.moves_map = {}

    def get_successors(self, state):
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def get_cost_of_actions(self, action):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return action.piece.get_num_tiles()

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        for target in self.targets:
            if state.state[target[0]][target[1]] == Free:
                return False
        return True

    def mini_contest_heuristic(self,state, problem):
        def heuristic(state):
            targets = [target for target in problem.targets if state.state[target[0]][target[1]] == Free]
            if len(get_optional_targets(copy.deepcopy(targets), state, 0)) != len(targets):
                return math.inf
            acc_chess_dist = 0
            available_locations = get_optional_locations(problem.board, 0)
            minimal_chess_dist = math.inf
            chess_distances = []
            for target in targets:
                minimal_chess_dist = math.inf
                for location in (available_locations + targets):
                    dist_chess = get_chess_dist(location , target)
                    if dist_chess > minimal_chess_dist:
                        minimal_chess_dist = dist_chess
                chess_distances.append(minimal_chess_dist)
            #chess_distances = get_minimal_chess_distances(available_locations, targets)

            for dist in chess_distances:
                acc_chess_dist += dist

            total_cost = acc_chess_dist #+ problem.moves_map[state][2]
            return total_cost
            #return max_chess_dist

        return heuristic
    def return_backtrace(begin_board, start, game_tree):
        backtrace = []

        curr_board = begin_board

        while curr_board != start:
            backtrace.append(game_tree[curr_board][0])
            curr_board = game_tree[curr_board][1]  # curr_board = prev_state[0]

        backtrace.reverse()
        return backtrace

    def a_star_for_contest(self, problem):
        start = problem.get_start_state()
        moves_map = problem.moves_map

        # In case that the start state is also the goal state
        if problem.is_goal_state(start):
            return []

        moves_map.update({start: None})
        # Fringe
        fringe = util.PriorityQueueWithFunction(self.mini_contest_heuristic(0, problem))

        successors = problem.get_successors(start)

        for stat in successors:
            moves_map.update({stat[0]: (stat[1], start, stat[2])})
            fringe.push(stat[0])

        # Explored Set
        explored_set = []

        # Loop
        while not fringe.isEmpty():
            node = fringe.pop()
            print("exctract sol",problem.expanded)

            # Case of goal state
            if problem.is_goal_state(node):
                return self.return_backtrace(node, start, moves_map)

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

    def solve(self):
        "*** YOUR CODE HERE ***"
        appropriate_blokus_problem = MiniContestSearch(self.board.board_w, self.board.board_h,
                                                           self.board.piece_list, self.starting_point,
                                                           self.targets)
        backtrace = self.a_star_for_contest(appropriate_blokus_problem)
        self.expanded = appropriate_blokus_problem.expanded
        return backtrace




def get_maximal_chess_distances (available_locations , targets):
    """
    :param available_locations: A list of all the free and legal locations which a move can begin at.
    :param targets: The squares in the board which have to be covered.
    :return: A list of tuples (manhattan distance , origin, destination) which measures the manhattan distances
    between the targets to the tiles of the CURRENT STATE of the board.
    """
    # if there are no available locations to place tiles in them the minimal distances are infinite
    if len(available_locations) == 0:
        return [[math.inf] for target in targets]
    # finds the minimal manhattan distance to every target from an oprional
    # tile placing location or another target
    maximal_distances = [0 for x in targets]
    idx = 0
    for target in targets:
        for orig in available_locations:
            chess_dist = get_chess_dist(target, orig)
            if chess_dist > maximal_distances[idx]:
                maximal_distances[idx] = chess_dist
        idx += 1

    return max(maximal_distances)

# Returns the chess distance between the two given points
def get_chess_dist (point1 , point2):
    return max(math.fabs(point1[0] - point2[0]) , math.fabs(point1[1] - point2[1]))


def get_optional_locations(board , player):
    """
    :param board: Contains a state
    :param player: The player to check for
    :return: A list of all the free and legal locations which a move can begin at.
    """
    optional_locations = []
    for i in range(board.board_h):
        for j in range(board.board_w):
            if (board._legal[player][i][j]) and (board.connected[player][i][j]):
                optional_locations.append((i, j))
    return optional_locations


def get_minimal_chess_distances (available_locations , targets):
    """
    :param available_locations: A list of all the free and legal locations which a move can begin at.
    :param targets: The squares in the board which have to be covered.
    :return: A list of tuples (manhattan distance , origin, destination) which measures the manhattan distances
    between the targets to the tiles of the CURRENT STATE of the board.
    """
    # if there are no available locations to place tiles in them the minimal distances are infinite
    if len(available_locations) == 0:
        return [[math.inf] for target in targets]
    # finds the minimal manhattan distance to every target from an oprional
    # tile placing location or another target
    minimal_distances = [(math.inf, x, x) for x in targets]
    all_orig = available_locations + targets
    idx = 0
    for target in targets:
        for orig in all_orig:
            chess_dist = get_chess_dist(target, orig)
            if chess_dist < minimal_distances[idx][DISTANCE] and orig != target:
                minimal_distances[idx] = (chess_dist, target, orig)
        idx += 1

    # Adapting the tuples in "minimal_distances" to represent a reasonable path problem (from the tiles to the targets).
    # First, creates a Data structure which contains all the "problematic" tuples
    problematic_targets = []
    for i in range (len(minimal_distances)):
        if minimal_distances[i][ORIGIN] in targets:
            problematic_targets.append(minimal_distances[i])

    # Second, this loop updates the tuples in "minimal_distances" to represent a reasonable path problem
    finished = False
    if len(problematic_targets) > 0:
        while not finished:
            removed = False
            for dest_to_check in problematic_targets:
                # Checks whether the destination in dest_to_check close to the player's tiles regarding minimal_distances
                item_in_minimal = [item for item in minimal_distances if item[DEST] == dest_to_check[ORIGIN]]
                problematic_dest = []
                for problem_dest in problematic_targets:
                    problematic_dest.append(problem_dest[DEST])
                if item_in_minimal[0][ORIGIN] in (available_locations + (list(set(targets) - set(problematic_dest)))):
                    cyclic_delete(problematic_targets, dest_to_check)
                    removed = True

            # In case that need to calculate the shortest manhattan distance from the group to the tiles
            if not removed:
                new_targets = [item[DEST] for item in problematic_targets]
                tuple_to_add = calculate_min_chess_dist_to_tiles(new_targets , available_locations)
                changed_item = [item for item in minimal_distances if item[DEST] == tuple_to_add[DEST]][0]
                minimal_distances[minimal_distances.index(changed_item)] = tuple_to_add

            if len(problematic_targets) == Empty:
                finished = True

    return minimal_distances


def cyclic_delete(problematic_targets, beginning):
    """

    :param problematic_targets: the data- structure (list) to delete from
    :param beginning: the first tuple (manhattan distance , origin, destination) to delete
    :return: updated list
    """

    orig = []
    orig.append(beginning[DEST]) # saves the origin in the tuple to delete

    # First, delete the beginning which has to be deleted
    problematic_targets.remove(beginning)

    # Second, tries to delete tuples where their destination is in orig
    deleted = True

    while deleted:

        deleted = False

        for tuple in problematic_targets:
             if tuple[ORIGIN] in orig:
                orig.append(tuple[DEST])
                problematic_targets.remove(tuple)
                deleted = True


def calculate_min_chess_dist_to_tiles(targets, available_locations):
    """

    :param available_locations: A list of all the free and legal locations which a move can begin at.
    :param targets: The squares in the board which have to be covered.
    :return: A tuple (manhattan distance , origin, destination) with minimal manhatttan distance
    """
    min_dest = math.inf
    for target in targets:
        for square in available_locations:
            if get_chess_dist(target , square) < min_dest:
                min_dest = get_chess_dist(target , square)
                ret_tuple = (min_dest, target, square)

    return ret_tuple
    # Data structure- contains all the problematic tuples
        #for each tuple (*, target, curr ) in the data structure:
            # if (*, curr , y ) as y in available_locations => REMOVE (*, target, curr ) from the data structure

    #for all tuples that have curr in targets:
        #if (* , curr, y) in minimal_distances and y in available_locations =>  O.K

def get_optional_targets(targets, board, player):
    """

    :param targets: The locations to cover
    :param board: The current board
    :param player: The player to check for
    :return:
    """

    # First, make sure that the targets' tiles are free
    for target in targets:
        #remove the targets which connected VERTICALLY to the player's tiles
        squares_to_check = []
        # general case: 1 < i < 8 , 1 < j < 8
        if (0 < target[0] and target[0] < board.board_h -1) and (0 < target[1] and target[1] < board.board_w -1):
            squares_to_check.append((target[0] + 1 , target[1]))
            squares_to_check.append((target[0] - 1 , target[1]))
            squares_to_check.append((target[0]  , target[1] + 1))
            squares_to_check.append((target[0]  , target[1] - 1))
        # first row(without corners): i = 1 , 1 < j < 8
        elif (target[0] == 0) and (0 < target[1] and target[1] < board.board_w -1):
            squares_to_check.append((target[0] + 1 , target[1]))
            squares_to_check.append((target[0]  , target[1] + 1))
            squares_to_check.append((target[0]  , target[1] - 1))
        # last row(without corners): i = 8 , 1 < j < 8
        elif (target[0] == board.board_h -1) and (0 < target[1] and target[1] < board.board_w -1 ):
            squares_to_check.append((target[0] - 1 , target[1]))
            squares_to_check.append((target[0]  , target[1] + 1))
            squares_to_check.append((target[0]  , target[1] - 1))
        # first col(without corners): j = 1 , 1 < j < 8
        elif (target[1] == 0) and (0 < target[0] and target[0] < board.board_h -1):
            squares_to_check.append((target[0] + 1 , target[1]))
            squares_to_check.append((target[0] - 1 , target[1]))
            squares_to_check.append((target[0]  , target[1] + 1))
        # last col(without corners): j = 8 , 1 < j < 8
        elif (target[1] == board.board_w -1) and (0 < target[0] and target[0] < board.board_h -1):
            squares_to_check.append((target[0] + 1 , target[1]))
            squares_to_check.append((target[0] - 1 , target[1]))
            squares_to_check.append((target[0]  , target[1] - 1))
        # corners
        else:
            if (target[0] == 0) and (target[1] == 0):
                squares_to_check.append((1 , 0))
                squares_to_check.append((0 , 1))
            elif (target[0] == 0) and (target[1] == board.board_w -1):
                squares_to_check.append((1 , board.board_w -1))
                squares_to_check.append((0 , board.board_w - 2))
            elif (target[0] == board.board_h - 1) and (target[1] == 0):
                squares_to_check.append((board.board_h - 2 , 0))
                squares_to_check.append((board.board_h -1 , 1))
            else: # the last corner (board_h , board_w)
                squares_to_check.append((board.board_h -1 , board.board_w - 2))
                squares_to_check.append((board.board_h -2 , board.board_w -1))

        removed = False
        for square in squares_to_check:
            if board.state[square[0]][square[1]] == player and not removed:
                targets.remove(target)
                removed = True

    return targets


def covered_by_mistake(board, targets):
    for target in targets:
        if board.state[target[0]][target[1]] != -1:
            targets.remove(target)
    return targets