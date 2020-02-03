# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Q value container as a Counter -> {(state, action): Qvalue}
        self.Qvalues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.Qvalues[(state, action)] if self.isKnownInMemory(state, action) else 0.0

    def isKnownInMemory(self, state, action):
        return (state, action) in self.Qvalues

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        action = self.computeActionFromQValues(state)
        return self.getQValue(state, action) if action is not None else 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actionsToQvalue = {a: self.getQValue(state, a) for a in self.getLegalActions(state)}
        return max(actionsToQvalue, key=actionsToQvalue.get) if actionsToQvalue else None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        # action = None
        "*** YOUR CODE HERE ***"
        if len(legalActions) == 0:
            action = None
        elif util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # we update the Q value
        lastQvalue = self.getQValue(state, action)

        currentQvalue = (1-self.alpha)*lastQvalue + self.alpha*(reward + self.discount * self.getValue(nextState))

        self.Qvalues[(state, action)] = currentQvalue

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def isKnownFeature(self, feature):
        return feature in self.weights

    def getWeights(self):
        return self.weights

    def getWeight(self, feature):
        return self.weights[feature] if self.isKnownFeature(feature) else 0.0

    def getFeatures(self, state, action):
        return self.featExtractor.getFeatures(state, action)

    def getFeatureValue(self, state, action, feature):
        return self.featExtractor.getFeatures(state, action)[feature]

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        return sum([self.getWeight(feature) * self.getFeatureValue(state, action, feature)
                    for feature in self.getFeatures(state, action)])

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        error = self.computeError(state, action, nextState, reward)

        # we update the weights
        for feat, feat_value in self.getFeatures(state, action).items():
            self.weights[feat] = self.getWeight(feat) + self.alpha*error*feat_value

    def computeError(self, state, action,  nextState, reward):
        return (reward + self.discount*self.getValue(nextState)) - self.getQValue(state, action)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass


if __name__ == '__main__':
    # for debug and to use the Profiler of PyCharm
    import pacman, os
    cmd = "pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 25 -n 30 -l trickyClassic"
    # cmd = "pacman.py -p PacmanQAgent -x 200 -n 200 -l smallGrid"
    # os.system("pacman.py -p PacmanQAgent -x 200 -n 200 -l smallGrid")
    argv = cmd.split(' ')[1:]
    args = pacman.readCommand(argv)
    pacman.runGames(**args)
