from util import Pair
import math
import copy
from proposition_layer import PropositionLayer
from plan_graph_level import PlanGraphLevel
from pgparser import PgParser
from action import Action

try:
    from search import SearchProblem
    from search import a_star_search

except:
    from CPF.search import SearchProblem
    from CPF.search import a_star_search


class PlanningProblem:
    def __init__(self, domain_file, problem_file):
        """
        Constructor
        """
        p = PgParser(domain_file, problem_file)
        self.actions, self.propositions = p.parse_actions_and_propositions()
        # list of all the actions and list of all the propositions

        initial_state, goal = p.parse_problem()
        # the initial state and the goal state are lists of propositions

        self.initialState = frozenset(initial_state)
        self.goal = frozenset(goal)

        self.create_noops()
        # creates noOps that are used to propagate existing propositions from one layer to the next

        PlanGraphLevel.set_actions(self.actions)
        PlanGraphLevel.set_props(self.propositions)
        self.expanded = 0

    def get_start_state(self):
        "*** YOUR CODE HERE ***"
        return self.initialState

    def is_goal_state(self, state):
        """
        Hint: you might want to take a look at goal_state_not_in_prop_payer function
        """
        "*** YOUR CODE HERE ***"
        for goal in self.goal:
            if goal not in state:
                return False
        return True

    def get_successors(self, state):
        """
        For a given state, this should return a list of triples,
        (successor, action, step_cost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'step_cost' is the incremental
        cost of expanding to that successor, 1 in our case.
        You might want to this function:
        For a list / set of propositions l and action a,
        a.all_preconds_in_list(l) returns true if the preconditions of a are in l

        Note that a state *must* be hashable!! Therefore, you might want to represent a state as a frozenset
        """
        self.expanded += 1
        "*** YOUR CODE HERE ***"
        ret_lst = []
        for act in self.actions:
            if act.all_preconds_in_list(state):
                succ = [prop for prop in state]
                for prop_to_add in act.get_add():
                    succ.append(prop_to_add)
                for prod_to_del in act.get_delete():
                    succ.remove(prod_to_del)
                froz_succ = frozenset(succ)
                tup_to_add = (froz_succ , act , 1)

                ret_lst.append(tup_to_add)
        return ret_lst


    @staticmethod
    def get_cost_of_actions( actions):
        return len(actions)

    def goal_state_not_in_prop_layer(self, propositions):
        """
        Helper function that returns true if all the goal propositions
        are in propositions
        """
        for goal in self.goal:
            if goal not in propositions:
                return True
        return False

    def create_noops(self):
        """
        Creates the noOps that are used to propagate propositions from one layer to the next
        """
        for prop in self.propositions:
            name = prop.name
            precon = []
            add = []
            precon.append(prop)
            add.append(prop)
            delete = []
            act = Action(name, precon, add, delete, True)
            self.actions.append(act)


def max_level(state, planning_problem):
    """
    The heuristic value is the number of layers required to expand all goal propositions.
    If the goal is not reachable from the state your heuristic should return float('inf')
    A good place to start would be:
    prop_layer_init = PropositionLayer()          #create a new proposition layer
    for prop in state:
        prop_layer_init.add_proposition(prop)        #update the proposition layer with the propositions of the state
    pg_init = PlanGraphLevel()                   #create a new plan graph level (level is the action layer and the propositions layer)
    pg_init.set_proposition_layer(prop_layer_init)   #update the new plan graph level with the the proposition layer
    """
    "*** YOUR CODE HERE ***"
    prop_layer_init = PropositionLayer()
    for prop in state:
        prop_layer_init.add_proposition(prop)


    prev_pg_lev = PlanGraphLevel()
    prev_pg_lev.set_proposition_layer(prop_layer_init)

    counter = 0

    while not planning_problem.is_goal_state(prev_pg_lev.get_proposition_layer().get_propositions()):
        curr_pg_lev = PlanGraphLevel()
        curr_pg_lev.expand_without_mutex(prev_pg_lev)
        counter += 1

        # checks for changes by the expansion - if hadn't occur => return INF
        if len(curr_pg_lev.get_proposition_layer().get_propositions()) == len(prev_pg_lev.get_proposition_layer().get_propositions()):
            return math.inf

        prev_pg_lev = curr_pg_lev

    return counter

def level_sum(state, planning_problem):
    """
    The heuristic value is the sum of sub-goals level they first appeared.
    If the goal is not reachable from the state your heuristic should return float('inf')
    """
    "*** YOUR CODE HERE ***"
    goals = [goal for goal in planning_problem.goal if goal not in state]
    prop_layer_init = PropositionLayer()
    for prop in state:
        prop_layer_init.add_proposition(prop)

    prev_pg_lev = PlanGraphLevel()
    prev_pg_lev.set_proposition_layer(prop_layer_init)

    counter , sum = 0 , 0

    while not planning_problem.is_goal_state(prev_pg_lev.get_proposition_layer().get_propositions()):
        curr_pg_lev = PlanGraphLevel()
        curr_pg_lev.expand_without_mutex(prev_pg_lev)
        counter += 1

        sub_goals_curr_lev = [goal for goal in goals if goal in curr_pg_lev.get_proposition_layer().get_propositions()]
        if len(sub_goals_curr_lev) != 0:
            for subgoal in sub_goals_curr_lev:
                goals.remove(subgoal)
                sum += counter

        # checks for changes by the expansion - if hadn't occur => return INF
        if len(curr_pg_lev.get_proposition_layer().get_propositions()) == len(
                prev_pg_lev.get_proposition_layer().get_propositions()):
            return math.inf

        prev_pg_lev = curr_pg_lev

    return sum

def is_fixed(graph, level):
    """
    Checks if we have reached a fixed point,
    i.e. each level we'll expand would be the same, thus no point in continuing
    """
    if level == 0:
        return False
    return len(graph[level].get_proposition_layer().get_propositions()) == len(
        graph[level - 1].get_proposition_layer().get_propositions())


def null_heuristic(*args, **kwargs):
    return 0


if __name__ == '__main__':
    import sys
    import time

    if len(sys.argv) != 1 and len(sys.argv) != 4:
        print("Usage: PlanningProblem.py domainName problemName heuristicName(max, sum or zero)")
        exit()
    domain = 'dwrDomain.txt'
    problem = 'dwrProblem.txt'
    heuristic = null_heuristic
    if len(sys.argv) == 4:
        domain = str(sys.argv[1])
        problem = str(sys.argv[2])
        if str(sys.argv[3]) == 'max':
            heuristic = max_level
        elif str(sys.argv[3]) == 'sum':
            heuristic = level_sum
        elif str(sys.argv[3]) == 'zero':
            heuristic = null_heuristic
        else:
            print("Usage: planning_problem.py domain_name problem_name heuristic_name[max, sum, zero]")
            exit()

    prob = PlanningProblem(domain, problem)
    start = time.clock()
    plan = a_star_search(prob, heuristic)
    elapsed = time.clock() - start
    if plan is not None:
        print("Plan found with %d actions in %.2f seconds" % (len(plan), elapsed))
    else:
        print("Could not find a plan in %.2f seconds" % elapsed)
    print("Search nodes expanded: %d" % prob.expanded)
