# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from collections import defaultdict
from typing import Dict

from cs188.my_utils import getIndexOfMax
from cs188.p3_reinforcement.mdp import MarkovDecisionProcess

from learningAgents import ValueEstimationAgent


def _computeQValueFromValues(
        mdp: MarkovDecisionProcess,
        values: Dict, discount: float,
        state, action):
    """
    $ Run a step of Q-value iteration: it computes Q[k+1](s, a) given
    the last computed values v[k].
    """
    return sum(
        prob * (mdp.getReward(state, action, nextState) + discount * values[nextState])
        for nextState, prob in mdp.getTransitionStatesAndProbs(state, action)
    )


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp: MarkovDecisionProcess, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = defaultdict(int)   # § don't need util.Counter
        self.policy = dict()             # § cached policy
        self.runValueIteration()

    def runValueIteration(self):
        """
        § Runs value iteration for self.iterations iterations:

            v[k](s) = max[a] Q[k](s, a)
            Q[k+1](s, a) = sum[s'] T(s, a, s') * [r(s, a, s') + discount*v[k](s')]

        and returns the resulting policy.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        discount = self.discount
        states = mdp.getStates()
        policy = dict()
        prevValues = defaultdict(int)
        currValues = self.values
        for k in range(self.iterations):
            prevValues, currValues = currValues, prevValues
            for state in states:
                possibleActions = mdp.getPossibleActions(state)
                if not possibleActions:
                    currValues[state] = 0   # $ we could just "continue"
                else:
                    qValues = [
                        _computeQValueFromValues(mdp, prevValues, discount, state, action)
                        for action in possibleActions
                    ]
                    maxActionIndex = getIndexOfMax(qValues)
                    currValues[state] = qValues[maxActionIndex]
                    policy[state] = possibleActions[maxActionIndex]
        self.values = currValues
        self.policy = policy

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # § We could cache Q-values as well
        return _computeQValueFromValues(
            self.mdp, self.values, self.discount, state, action)

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        return self.policy.get(state)

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp: MarkovDecisionProcess, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        discount = self.discount
        states = mdp.getStates()
        policy = dict()
        values = self.values
        for k in range(self.iterations):
            state = states[k % len(states)]
            possibleActions = mdp.getPossibleActions(state)
            if not possibleActions:
                continue
            else:
                qValues = [
                    _computeQValueFromValues(mdp, values, discount, state, action)
                    for action in possibleActions
                ]
                maxActionIndex = getIndexOfMax(qValues)
                values[state] = qValues[maxActionIndex]
                policy[state] = possibleActions[maxActionIndex]
        self.values = values
        self.policy = policy


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
