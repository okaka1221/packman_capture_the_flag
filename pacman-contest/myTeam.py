# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    def __init__(self, index, epsillon=0.5, alpha=0.2, gamma=0.8, **args):
        CaptureAgent.__init__(self, index)
        self.epsillon = epsillon
        self.alpha = alpha
        self.discout = gamma
        self.lastState = None
        self.lastAction = None
        self.targetPos = None
        self.mazeSize = None
        self.border = None
        self.initialFoodList = None
        self.specificPath = []
 
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        self.initialFoodList = self.getFood(gameState).asList()

        walls = gameState.getWalls()
        self.mazeSize = walls.height * walls.width

        boader = []
        x = walls.width // 2 - 1
        for y in range(1, walls.height - 1):
            if not walls[x][y] and (x, y) != self.start:
                boader.append((x, y))
        self.border = boader
        
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def doAction(self, gameState, action):
        """
        update last state and action
        """
        self.lastState = gameState
        self.lastAction = action

    def observationFunction(self, gameState):
        """
        This is where we ended up after our last action.
        The simulation should somehow ensure this is called
        """
        if self.lastState:
            reward = self.getReward(gameState)
            self.update(self.lastState, self.lastAction, gameState, reward)
            self.updateTarget(gameState)
        return gameState

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        values = [self.getQValue(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        bestAction = random.choice(bestActions)
        
        self.doAction(gameState, bestAction)

        return bestAction

    def getQValue(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        return features * self.weights
    
    def update(self, gameState, action, nextState, reward):
        actions = nextState.getLegalActions(self.index)
        values = [self.getQValue(nextState, a) for a in actions]
        maxValue = max(values)
        features = self.getFeatures(gameState, action)
        
        diff = (reward + self.discout * maxValue) - self.getQValue(gameState, action)

        for feature in features:
            self.weights[feature] += self.alpha * diff * features[feature]

    def getGhosts(self, gameState):
        enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
        ghosts = [e for e in enemies if not e.isPacman]
        return ghosts

    def getInvadors(self, gameState):
        enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
        invaders = [e for e in enemies if e.isPacman]
        return invaders
    
    def getSafeActions(self, gameState):
        safeActions = []
        myPos = gameState.getAgentPosition(self.index)
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')

        for action in actions:
            successor = self.getSuccessor(gameState, action)
            myNextPos = successor.getAgentPosition(self.index)

            finalNode = self.aStarSearch(successor, self.border, [myPos])
            if finalNode[2] < self.mazeSize:
                safeActions.append(action)

        return safeActions

    def getAlternativePath(self, gameState, minPathLength=5, penaltyDist=2, exploreRange=5):
        walls = gameState.getWalls()
        myPos = gameState.getAgentPosition(self.index)
        ghosts = self.getGhosts(gameState)
        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        targetList = foodList + capsuleList
        
        penaltyPos = []
        
        for ghost in ghosts:
            for x in range(max(1, myPos[0] - exploreRange), min(myPos[0] + exploreRange, walls.width)):
                for y in range(max(1, myPos[1] - exploreRange), min(myPos[1] + exploreRange, walls.height)):
                    pos = (int(x), int(y)) 
                    if not pos in walls.asList():
                        distToGhost = self.getMazeDistance(pos, ghost.getPosition())
                        if distToGhost <= penaltyDist:
                            penaltyPos.append(pos)
                    if pos in targetList:
                        targetList.remove(pos)

        if len(targetList) == 0:
            return [], None

        finalNode = self.aStarSearch(gameState, targetList, penaltyPos)
        pathLength = min(minPathLength, len(finalNode[1]))
        return finalNode[1][0:pathLength], finalNode[0]
                        
    def aStarSearch(self, gameState, goals, penaltyPos=[], avoidGhost=False):
        walls = gameState.getWalls().asList()
        ghosts = self.getGhosts(gameState)
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        startPos = gameState.getAgentPosition(self.index)
        currentNode = (startPos, [], 0)
        pQueue = util.PriorityQueueWithFunction(lambda item: item[2] + min(self.getMazeDistance(item[0], goal) for goal in goals))
        pQueue.push(currentNode)
        closed = set()

        while currentNode[0] not in goals and not pQueue.isEmpty():
            currentNode = pQueue.pop()
            successors = [((currentNode[0][0] + v[0], currentNode[0][1] + v[1]), a) for v, a in zip(actionVectors ,actions)]
            legalSuccessors = [s for s in successors if s[0] not in walls]

            for successor in legalSuccessors:
                if successor[0] not in closed:
                    closed.add(successor[0])

                    position = successor[0]
                    path = currentNode[1] + [successor[1]]
                    cost = currentNode[2] + 1
                    wallCount = 0

                    if successor[0] in penaltyPos:
                        cost += self.mazeSize

                    if avoidGhost:
                        distToGhost = min([self.getMazeDistance(successor[0], a.getPosition()) for a in ghosts])
                        if distToGhost > 0:
                            cost += (self.mazeSize / 4) / distToGhost

                    pQueue.push((position, path, cost))

        return currentNode
    
    def isOppoentsScared(self, gameState, timer=3):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

        for a in enemies:
            if not a.isPacman and a.scaredTimer <= timer:
                return False

        return True

    def isStucking(self, gameState, stuckingCount=4):
        history = self.observationHistory
        count = 0
        myPos = gameState.getAgentPosition(self.index)
        
        if len(history) > 0:
            for i in range(min(10, len(history))):
                myPastPos = history[-i - 1].getAgentPosition(self.index)
                if myPastPos == myPos:
                    count += 1
            
        return count >= stuckingCount

    def isChased(self, gameState, chasedCount=2):
        history = self.observationHistory
        myState = gameState.getAgentState(self.index)
        ghosts = self.getGhosts(gameState)
        
        if len(history) == 0 or len(ghosts) == 0 or not myState.isPacman:
            return False

        myPos = myState.getPosition()
        distToGhost = min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts])
        
        if distToGhost >= 4:
            return False

        for i in range(min(chasedCount, len(history))):
            pastState = history[-i - 1]
            myPastPos = pastState.getAgentPosition(self.index)
            pastGhosts = self.getGhosts(pastState)
            if len(pastGhosts) == 0:
                return False
            
            pastDistToGhost = min([self.getMazeDistance(myPastPos, a.getPosition()) for a in pastGhosts])
            if pastDistToGhost != distToGhost:
                return False
            
        return True

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    def __init__(self, index, **args):
        ReflexCaptureAgent.__init__(self, index, **args)
        self.weights = util.Counter({
            'bias': 1.0,
            'distToTarget': -10.0,
            'distToGhost': 5.0,
            'distToBorder': -1.0,
            'eatFood': 1.0,
            '#-of-ghosts-2-step-away': -1,
        })
        self.numFoodCarrying = 0
        self.wasAlternative = False

    def registerInitialState(self, gameState):
        ReflexCaptureAgent.registerInitialState(self, gameState)

        myPos = gameState.getAgentPosition(self.index)
        foodList = self.getFood(gameState).asList()
        if len(foodList) > 0:
            self.targetPos = max(foodList, key=lambda x: self.getMazeDistance(myPos, x))

    def updateTarget(self, gameState):
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        myLastState = self.lastState.getAgentState(self.index)
        foodList = self.getFood(gameState).asList()
        lastFoodList = self.getFood(self.lastState).asList()
        capsuleList = self.getCapsules(gameState)
        ghosts = self.getGhosts(gameState)
        closestBorder = min(self.border, key=lambda x: self.getMazeDistance(myPos, x))
        minDistToBorder = min([self.getMazeDistance(myPos, b) for b in self.border])

        if myPos == self.start:
            self.numFoodCarrying = 0
            self.specificPath = []
            if len(foodList) > 0:
                self.targetPos = min(foodList, key=lambda x: self.getMazeDistance(myPos, x))
            print("RETURN TO START")

        if len(lastFoodList) > len(foodList):
            self.numFoodCarrying += 1
            print("EAT FOOD")

        if not myState.isPacman:
            self.numFoodCarrying = 0

        timeLeft = gameState.data.timeleft / gameState.getNumAgents()
        if timeLeft - minDistToBorder <= 4:
            self.targetPos = closestBorder
            print("TIMER UPDATE")

        if len(foodList) == 0:
            self.targetPos = closestBorder
            print("UPDATE 1")

        if len(foodList) > 0 and len(ghosts) == 0:
                if myPos == self.targetPos or myPos in lastFoodList:
                    self.targetPos = min(foodList, key=lambda x: self.getMazeDistance(myPos, x))
                    print("UPDATE 2")
            
        if len(ghosts) > 0:
            minDistToFood =  min([self.getMazeDistance(myPos, f) for f in foodList])
            minDistToGhost =  min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts])

            if len(foodList) > 0:
                if myPos == self.targetPos or myPos in lastFoodList and self.numFoodCarrying == 0:
                    if minDistToGhost <= 3 and not self.isOppoentsScared(gameState):
                        path, target = self.getAlternativePath(gameState, minPathLength=6)
                        self.specificPath = path
                        self.targetPos = target
                        print("UPDATE 3")
                    else:
                        self.targetPos = min(foodList, key=lambda x: self.getMazeDistance(myPos, x))
                        print("UPDATE 4")
                elif not myState.isPacman and myLastState.isPacman and minDistToGhost <= 3:
                    path, target = self.getAlternativePath(gameState, minPathLength=6)
                    self.specificPath = path
                    self.targetPos = target
                    print("UPDATE 5")

            if not self.isOppoentsScared(gameState) and self.isChased(gameState):
                self.targetPos = closestBorder
                print("UPDATE 6")

            if  self.numFoodCarrying >= 3 and not self.isOppoentsScared(gameState) and minDistToBorder < minDistToFood:
                self.targetPos = closestBorder
                print("UPDATE 7")
                    
            if self.numFoodCarrying >= 5 and not self.isOppoentsScared(gameState) and minDistToGhost <= 5:
                self.targetPos = closestBorder
                print("UPDATE 8")
            
            if len(capsuleList) > 0:
                minDistToCapsule =  min([self.getMazeDistance(myPos, c) for c in capsuleList])
                
                if not self.isOppoentsScared(gameState) and self.isChased(gameState) and minDistToCapsule < minDistToBorder:
                    self.targetPos = min(capsuleList, key=lambda x: self.getMazeDistance(myPos, x))
                    print("UPDATE 9")

        print("TARGET", self.targetPos)
        
    def observationFunction(self, gameState):
        """
        This is where we ended up after our last action.
        The simulation should somehow ensure this is called
        """
        if self.lastState:
            reward = self.getReward(gameState)
            self.update(self.lastState, self.lastAction, gameState, reward)
            self.updateTarget(gameState)
        return gameState

    def getFeatures(self, gameState, action):
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        
        successor = self.getSuccessor(gameState, action)
        myNextState = successor.getAgentState(self.index)
        myNextPos = myNextState.getPosition()

        ghosts = self.getGhosts(gameState)
        distToGhost = 0.0

        if len(ghosts) > 0:
            distToGhost = min([self.getMazeDistance(myNextPos, a.getPosition()) for a in ghosts])
            if not self.isOppoentsScared and distToGhost <= 1:
                distToGhost = -999999
            if self.isOppoentsScared(successor) and myNextState.isPacman:
                if distToGhost > 0:
                    distToGhost *= 0.8
                else:
                    distToGhost = 999999

        capsuleList = self.getCapsules(gameState)
        distToCapsule = 0.0
        if len(capsuleList) > 0:
            distToCapsule = min([self.getMazeDistance(myNextPos, capsule) for capsule in capsuleList])
        
        features = util.Counter()
        features['bias'] = 1.0
        features['distToTarget'] = self.getMazeDistance(myNextPos, self.targetPos) / self.mazeSize
        features['distToGhost'] = distToGhost / self.mazeSize
        features['distToCapsule'] = distToCapsule / self.mazeSize
        
        features['#-of-ghosts-2-step-away'] = len([ghost for ghost in ghosts if self.getMazeDistance(myNextPos, ghost.getPosition()) <= 2])
        
        foodList = self.getFood(gameState).asList()
        if not features['#-of-ghosts-2-step-away'] and myNextPos in foodList:
            features['eatFood'] = 1.0
        
        if self.numFoodCarrying > 0: 
            features['distToBorder'] = min([self.getMazeDistance(myPos, b) for b in self.border]) / (self.mazeSize / 4)
        
        return features

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        ghosts = self.getGhosts(gameState)
        
        if len(self.specificPath) > 0:
            return self.specificPath.pop(0)
        elif self.isStucking(gameState):
            print("IS STUCKING")
            actions, target = self.getAlternativePath(gameState, minPathLength=3)
            if len(actions) > 0:
                self.specificPath = actions
                self.targetPos = target
                return self.specificPath.pop(0)
            else:
                actions = gameState.getLegalActions(self.index)
                return random.choice(actions)
                
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')

        if len(ghosts) > 0:
            distToGhost = min([self.getMazeDistance(myPos, a.getPosition()) for a in ghosts])
            if not self.isOppoentsScared(gameState) and myState.isPacman and distToGhost <= 6:
                safeActions = self.getSafeActions(gameState)
                if len(safeActions) > 0:
                    actions = safeActions

        values = [self.getQValue(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        bestAction = random.choice(bestActions)
        
        self.doAction(gameState, bestAction)

        return bestAction

    def getReward(self, gameState):
        reward = 0
        myPos = gameState.getAgentPosition(self.index)
        myLastPos = self.lastState.getAgentPosition(self.index)
        foodList = self.getFood(self.lastState).asList()
        capsuleList = self.getCapsules(self.lastState)

        if myPos != self.targetPos:
            reward -= 1
        else:
            if myPos in foodList:
                reward += 1
            elif myPos in capsuleList:
                reward += 2
            elif self.numFoodCarrying > 0 and gameState.getAgentState(self.index).isPacman:
                reward += 3
            else:
                reward += self.getScore(gameState) - self.getScore(self.lastState)

            distToPrevPos = self.getMazeDistance(myPos, myLastPos)
            if distToPrevPos > 1:
                reward -= distToPrevPos / self.mazeSize

        return reward


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def __init__(self, index, **args):
        ReflexCaptureAgent.__init__(self, index, **args)
        self.weights = util.Counter({
            'bias': 1.0,
            'distToTarget': -1.0,
            'distToInvader': -1.0,
            'numOfInvaders': -1.0,
            'isOppoentsScared(gameState)': 1.0,
            'onDefense': 10.0,
            'stop': -1.0,
        })

    def registerInitialState(self, gameState):
        ReflexCaptureAgent.registerInitialState(self, gameState)
        
        distCounter = util.Counter()
        for b in self.border:
            dist = 0
            for food in self.getFoodYouAreDefending(gameState).asList():
                dist = self.getMazeDistance(b, food)
            distCounter[b] = dist

        self.targetPos = max(distCounter, key=distCounter.get)

    def getFeatures(self, gameState, action):
        successor = self.getSuccessor(gameState, action)

        myNextState = successor.getAgentState(self.index)
        myNextPos = myNextState.getPosition()

        invaders = self.getInvadors(gameState)
        distToInvader = 0.0
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myNextPos, a.getPosition()) for a in invaders]
            distToInvader = min(dists)


        features = util.Counter()
        features['bias'] = 1.0
        features['numOfInvaders'] = len(invaders) / 2.0
        features['distToInvader'] = distToInvader / self.mazeSize
        features['distToTarget'] = self.getMazeDistance(myNextPos, self.targetPos) / self.mazeSize
        
        if myNextState.scaredTimer > 0:
            features['isOppoentsScared'] = (distToInvader - myNextState.scaredTimer) / self.mazeSize
        else:         
            features['isOppoentsScared'] = 0.0

        if not myNextState.isPacman: features['onDefense'] = 1.0

        if action == Directions.STOP: features['stop'] = 1.0

        return features

    def getReward(self, gameState):
        reward = 0
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        myLastState = self.lastState.getAgentState(self.index)
        myLastPos = myLastState.getPosition()
        foodList = self.getFoodYouAreDefending(gameState).asList()
        lastFoodList = self.getFoodYouAreDefending(self.lastState).asList()
        capsuleList = self.getCapsulesYouAreDefending(gameState)
        lastCapsuleList = self.getCapsulesYouAreDefending(self.lastState)

        if myPos != self.targetPos:
            reward -= 1
        else:
            if len(foodList) < len(lastFoodList):
                reward -= 1
            elif len(capsuleList) < len(lastCapsuleList):
                reward -= 2
            else:
                reward += self.getScore(gameState) - self.getScore(self.lastState)
            
            distToPrevPos = self.getMazeDistance(myPos, myLastPos)
            if distToPrevPos > 1:
                reward -= distToPrevPos / self.mazeSize

        return reward

    def updateTarget(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        foodList = self.getFoodYouAreDefending(gameState).asList()
        lastFoodList = self.getFoodYouAreDefending(self.lastState).asList()
        invaders = self.getInvadors(gameState)

        if len(foodList) - len(lastFoodList) < 0:
            eatenFood = [f for f in lastFoodList if f not in foodList]
            self.targetPos = eatenFood[0]

        if len(invaders) == 0:
            distCounter = util.Counter()
            for b in self.border:
                dist = 0
                for food in self.getFoodYouAreDefending(gameState).asList():
                    dist = self.getMazeDistance(b, food)
                distCounter[b] = dist

            self.targetPos = min(distCounter, key=distCounter.get)


