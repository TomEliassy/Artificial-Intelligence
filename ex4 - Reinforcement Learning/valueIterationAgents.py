# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util, math

from learningAgents import ValueEstimationAgent

PROB = 1
STAT = 0
class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    "*** YOUR CODE HERE ***"
    self.values_list = util.Counter() # Store all the intermediate values

    # Init - Saves for each state a list ...
    for stat in self.mdp.getStates():
        self.values_list[stat] = []

    # Performs Iteration
    for i in range (iterations):
        # In each Iteration - computes for all the possible states
        for stat in self.mdp.getStates():
            # As was written in the forum: the value of a terminal state is 0
            if not self.mdp.isTerminal(stat):
                vals_for_acts = util.Counter() # contains the values of the possible actions in this state
                for act in self.mdp.getPossibleActions(stat):
                    for stat_prob in self.mdp.getTransitionStatesAndProbs(stat, act):
                        if ( i != 0):
                            vals_for_acts[act] += (discount * stat_prob[PROB] * self.values_list[stat_prob[STAT]][i-1])
                        else:
                            vals_for_acts[act] = 0
                self.values_list[stat].append(self.mdp.getReward(stat, None, stat) + max (vals_for_acts.values()))
            else:
                self.values_list[stat].append(0)

    # Copies the final values to self.values
    for state in self.values_list:
        if len(self.values_list[state]) == 0:
            self.values[state] = 0
        else:
            self.values[state] = self.values_list[state][-1]
    # Saves the relevant values
    #for state in self.values_list:
        #self.values[state] = self.values_list[state][iterations - 1]
    # def findMaxNeighbor(self, state , k):
    #   if k < 0 :
    #     return 0
    #   neighbors_utilities = util.Counter()
    #   for act in self.mdp.getPossibleActions(state):
    #     neighbors_utilities[act] = 0
    #     next_states_and_probs = self.mdp.getTransitionStatesAndProbs(state, act)
    #     for stat_prob in next_states_and_probs:
    #       neighbors_utilities[act] += (stat_prob[PROB] * self.values_list[stat_prob[STAT]][k])
    #   return neighbors_utilities[neighbors_utilities.argMax()]
    #
    # # Init - Saves for each state a list ...
    # for stat in self.mdp.getStates():
    #   self.values_list[stat] = []
    #
    # # Loop
    # for i in range (iterations):
    #   for stat in self.values_list:
    #     self.values_list[stat].append(self.mdp.getReward(stat, None, stat) + ((discount) * findMaxNeighbor(self, stat , i - 1)))
    #
    # for state in self.values_list:
    #     self.values[state] = self.values_list[state][iterations - 1]
    # return None

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    next_states_and_probs , acc_val = self.mdp.getTransitionStatesAndProbs(state, action) , 0

    for state_and_prob in next_states_and_probs:
      acc_val += (self.mdp.getReward(state,action,state_and_prob[STAT]) + self.discount * state_and_prob[PROB] * self.values[state_and_prob[STAT]])

    return acc_val
    util.raiseNotDefined()

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    # Case of terminal state
    acts = self.mdp.getPossibleActions(state)
    # Case of terminal state
    if self.mdp.isTerminal(state) or len(acts) == 0:
        return None

    vals = util.Counter()

    for act in acts:
        vals[act] = self.getQValue(state , act)
    return vals.argMax()

    util.raiseNotDefined()

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
