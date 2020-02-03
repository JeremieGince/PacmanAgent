# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    # trivial case
    if problem.isGoalState(problem.getStartState()):
        return []

    path = list()  # the as a list that we want to return
    to_visit = util.Stack()  # nodes that have to be visited
    already_visited = util.Stack()  # nodes already visited
    node_to_neighbours = dict()  # map nodes to their neighbours
    actions_v_to_v_prime = dict()  # action for the transition v --> v' (v and v' are nodes here)

    # stack all node until we find the goal
    state = problem.getStartState()
    to_visit.push(state)
    while not to_visit.isEmpty():
        state = to_visit.pop()
        if problem.isGoalState(state):
            already_visited.push(state)
            node_to_neighbours[state] = []
            break
        if state not in already_visited.list:
            already_visited.push(state)
            node_to_neighbours[state] = list()
            for nextState, action, cost in problem.getSuccessors(state):
                actions_v_to_v_prime[(state, nextState)] = action
                to_visit.push(nextState)
                node_to_neighbours[state].append(nextState)

    # unstack to make the path
    state = already_visited.pop()
    while not already_visited.isEmpty():
        previousState = already_visited.pop()
        if state in node_to_neighbours[previousState]:
            path.append(actions_v_to_v_prime[(previousState, state)])
            state = previousState
    path.reverse()  # cause we stack the path in reverse, starting by the goal
    return path


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # trivial case
    if problem.isGoalState(problem.getStartState()):
        return []

    path = list()  # the as a list that we want to return
    to_visit = util.Queue()  # nodes that have to be visited
    already_visited = util.Stack()  # nodes already visited
    node_to_neighbours = dict()  # map nodes to their neighbours
    actions_v_to_v_prime = dict()  # action for the transition v --> v' (v and v' are nodes here)

    # Queue all node until we find the goal
    state = problem.getStartState()
    to_visit.push(state)
    already_visited.push(state)
    while not to_visit.isEmpty():
        state = to_visit.pop()
        node_to_neighbours[state] = list()  # initiate the neighbours list
        if problem.isGoalState(state):  # we found the goal state, so we break
            already_visited.push(state)
            node_to_neighbours[state] = []
            break
        for nextState, action, cost in problem.getSuccessors(state):
            actions_v_to_v_prime[(state, nextState)] = action
            if nextState not in already_visited.list:
                to_visit.push(nextState)
                already_visited.push(nextState)
                node_to_neighbours[state].append(nextState)

    # unstack to make the path
    state = already_visited.pop()
    while not already_visited.isEmpty():
        previousState = already_visited.pop()
        if previousState in node_to_neighbours and state in node_to_neighbours[previousState]:
            path.append(actions_v_to_v_prime[(previousState, state)])
            state = previousState
    path.reverse()  # cause we stack the path in reverse, starting by the goal
    return path


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # trivial case
    if problem.isGoalState(problem.getStartState()):
        return []

    already_visited = set()  # nodes already visited
    path_to_cost = {(problem.getStartState(),): 0}  # map of paths and their cost
    coordsPath_to_actionsPath = {(problem.getStartState(),): []}  # map of paths ans their actions to make it
    paths = util.PriorityQueue()  # paths as priority queue with cost for priority
    paths.push((problem.getStartState(),), path_to_cost[(problem.getStartState(),)])

    # Making paths with their cumulative cost. We start with the root node.
    while not paths.isEmpty():
        path = paths.pop()  # get the lowest cost path
        if problem.isGoalState(path[-1]):  # we found the lowest cost path with goal as end
            return coordsPath_to_actionsPath[path]
        if path[-1] in already_visited:
            continue

        # looking for successors to continue to build the path
        for nextState, action, cost in problem.getSuccessors(path[-1]):
            if nextState not in path:
                new_path = path + (nextState,)
                new_cost = path_to_cost.get(path) + cost
                path_to_cost[new_path] = new_cost
                coordsPath_to_actionsPath[new_path] = coordsPath_to_actionsPath[path] + [action]
                paths.push(new_path, new_cost)

        already_visited.add(path[-1])


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # trivial case
    if problem.isGoalState(problem.getStartState()):
        return []

    already_visited = set()  # nodes already visited

    # map of paths and their cost
    path_to_cost = {(problem.getStartState(),): heuristic(problem.getStartState(), problem)}

    coordsPath_to_actionsPath = {(problem.getStartState(),): []}  # map of paths ans their actions to make it
    paths = util.PriorityQueue()  # paths as priority queue with cost for priority
    paths.push((problem.getStartState(),), path_to_cost[(problem.getStartState(),)])

    # Making paths with their cumulative cost. We start with the root node.
    while not paths.isEmpty():
        path = paths.pop()  # get the lowest cost path
        if problem.isGoalState(path[-1]):  # we found the lowest cost path with goal as end
            return coordsPath_to_actionsPath[path]

        if path[-1] in already_visited:
            continue

        # looking for successors to continue to build the path
        for nextState, action, cost in problem.getSuccessors(path[-1]):
            if nextState not in path:
                new_path = path + (nextState,)
                g = path_to_cost.get(path) + cost - heuristic(path[-1], problem)
                new_cost = g + heuristic(nextState, problem)
                path_to_cost[new_path] = new_cost
                coordsPath_to_actionsPath[new_path] = coordsPath_to_actionsPath[path] + [action]
                paths.push(new_path, new_cost)

        already_visited.add(path[-1])


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
