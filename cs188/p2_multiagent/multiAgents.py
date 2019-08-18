# multiAgents.py
# --------------
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
import math
import random
from typing import NamedTuple, Union, Any, Sequence

import util
from cs188.p2_multiagent.game import Game, AgentState
from cs188.p2_multiagent.pacman import GameState

from game import Agent
from util import manhattanDistance

from game import Directions

from cs188.p1_search.searchAgents import mazeDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide. You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generateSuccessor(0, action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()

        "*** YOUR CODE HERE ***"
        ghostDistances = [manhattanDistance(newPos, ghost.configuration.pos)
                          for ghost in newGhostStates
                          if ghost.scaredTimer == 0]

        minGhostDist = min(ghostDistances, default=100)
        if minGhostDist == 0:
            return -math.inf
        numFood = successorGameState.getNumFood()
        if numFood == 0:
            return math.inf

        food = currentGameState.getFood()
        if food[newPos[0]][newPos[1]]:
            minFoodDist = 0
        else:
            foodDistances = [
                manhattanDistance(newPos, (x, y))
                for x in range(food.width)
                for y in range(food.height)
                if food[x][y]
            ]
            minFoodDist = min(foodDistances, default=0)

        danger = 1 / (minGhostDist - 0.8)
        profit = 1 / (minFoodDist + 0.5)
        score = -danger + profit
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class ScoredAction(NamedTuple):
    score: Union[int, float]
    action: Any


def getIndexOfMax(values: Sequence, default=-1):
    return max(range(len(values)), key=values.__getitem__, default=default)


def getIndexOfMin(values: Sequence, default=-1):
    return min(range(len(values)), key=values.__getitem__, default=default)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions()
        successors = (gameState.generateSuccessor(0, action) for action in legalActions)
        scores = [self._minimaxScore(successor, 1, self.depth) for successor in successors]
        i = getIndexOfMax(scores)
        return legalActions[i]

    def _minimaxScore(self, state, agentIndex: int, depth: int) -> ScoredAction:
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if agentIndex == 0:  # pacman turn
            selectBestScore = max
            nextAgent = 1
            nextDepth = depth
        else:  # ghost turn
            selectBestScore = min
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = (depth - 1) if nextAgent == 0 else depth

        legalActions = state.getLegalActions(agentIndex)
        successors = (state.generateSuccessor(agentIndex, action) for action in legalActions)
        scores = [self._minimaxScore(successor, nextAgent, nextDepth)
                  for successor in successors]
        return selectBestScore(scores)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self._minimax(gameState, 0, self.depth,
                             alpha=-math.inf, beta=math.inf).action

    def _minimax(self, state, agentIndex: int, depth: int, alpha, beta) -> ScoredAction:
        if depth == 0 or state.isWin() or state.isLose():
            return ScoredAction(self.evaluationFunction(state), action=None)
        if agentIndex == 0:
            return self._searchMax(state, agentIndex, depth, alpha, beta)
        else:
            return self._searchMin(state, agentIndex, depth, alpha, beta)

    def _searchMax(self, state, agentIndex, depth, alpha, beta):
        maxScore = -math.inf
        maxAction = None
        legalActions = state.getLegalActions(agentIndex)
        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action)
            score = self._minimax(successor, 1, depth, alpha, beta).score
            if score > beta:
                return ScoredAction(score, action)
            if score > maxScore:
                maxScore = score
                maxAction = action
                if score > alpha:
                    alpha = score
        return ScoredAction(maxScore, maxAction)

    def _searchMin(self, state, agentIndex, depth, alpha, beta):
        minScore = math.inf
        minAction = None
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth if nextAgent > 0 else depth - 1
        legalActions = state.getLegalActions(agentIndex)
        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action)
            score = self._minimax(successor, nextAgent, nextDepth, alpha, beta).score
            if score < alpha:
                return ScoredAction(score, action)
            if score < minScore:
                minScore = score
                minAction = action
                if score < beta:
                    beta = score
        return ScoredAction(minScore, minAction)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions()
        successors = (gameState.generateSuccessor(0, action) for action in legalActions)
        scores = [self._expectedScore(successor, 1, self.depth) for successor in successors]
        i = getIndexOfMax(scores)
        return legalActions[i]

    def _expectedScore(self, state, agentIndex: int, depth: int):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        legalActions = state.getLegalActions(agentIndex)
        successors = (state.generateSuccessor(agentIndex, action) for action in legalActions)
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth if nextAgent > 0 else depth - 1
        scores = [self._expectedScore(successor, nextAgent, nextDepth)
                  for successor in successors]
        if agentIndex == 0:  # pacman turn
            return max(scores)
        else:  # ghost turn
            return sum(scores) / len(legalActions)


def avg(nums):
    return sum(nums) / len(nums)


def betterEvaluationFunction(state: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Nothing complex here, I can think of many better more elaborate
    ways with a more complex implementation but, let's be honest, who cares about
    Pacman?
    """
    "*** YOUR CODE HERE ***"
    currentScore = state.getScore()
    if state.isWin() or state.isLose():
        return currentScore
    pacman = state.getPacmanPosition()
    ghosts: AgentState = state.getGhostStates()
    ghostDistances = [mazeDistance(pacman, tuple(map(int, ghost.configuration.pos)), state)
                      for ghost in ghosts]
    scaredTimers = [ghost.scaredTimer for ghost in ghosts]
    distFromUnscared = [dist for dist, timer in zip(ghostDistances, scaredTimers) if timer == 0]
    distFromScared = [dist for dist, timer in zip(ghostDistances, scaredTimers) if timer > 2]
    ghostPenalty = sum((300 / dist ** 2 for dist in distFromUnscared), 0)
    ghostBonus = sum((190 / dist for dist in distFromScared), 0)

    foods = state.getFood().asList()
    manhattanDistances = [(manhattanDistance(pacman, food), food) for food in foods]
    manhattanNearestFood = [food for dist, food in sorted(manhattanDistances)[:5]]
    mazeNearestFood = sorted(mazeDistance(pacman, food, state) for food in manhattanNearestFood)
    foodBonus = sum(9 / d for d in mazeNearestFood)

    score = currentScore - ghostPenalty + ghostBonus + foodBonus # + capsuleBonus
    return score


# Abbreviation
better = betterEvaluationFunction
