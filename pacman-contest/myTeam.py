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
               first = 'MyTeamAgent', second = 'MyTeamAgent'):
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

TEAM_STATE_DICT = {}

class MyTeamAgent(CaptureAgent):
    def __init__(self, index, epsillon=0.5, alpha=0.2, gamma=0.8, **args):
        CaptureAgent.__init__(self, index)
        self.epsillon = epsillon
        self.alpha = alpha
        self.discout = gamma

        self.isDefence = False
        self.mateIndex = None
        self.lastState = None
        self.lastAction = None
        self.targetPos = None
        self.mazeSize = None
        self.border = None
        self.oppoentBorder = None
        self.initialFoodNum = None
        self.specificPath = []
        self.weights = util.Counter({
            'distToTarget': -20,
            'distToOpponent': 1,
        })

    def registerInitialState(self, gameState):
        """
        This is initialization phase.
        Read weight file
        Get maze size
        Get border coordinates
        Set first target position
        """

        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        teamList = self.getTeam(gameState)
        teamList.remove(self.index)
        self.mateIndex = teamList[0]

        TEAM_STATE_DICT[self.mateIndex] = {
            'targetPos': None,
            'isDefence': False
        }

        # set initial num of foods to eat
        self.initialFoodNum = len(self.getFood(gameState).asList())

        # get maze size
        walls = gameState.getWalls()
        self.mazeSize = walls.height * walls.width

        # get border
        border = []
        x = walls.width // 2
        if self.red:
            x -= 1
        for y in range(1, walls.height - 1):
            if not walls[x][y] and (x, y) != self.start:
                border.append((x, y))
        self.border = border

        # get border
        opponentBorder = []
        x = walls.width // 2
        if not self.red:
            x -= 1
        for y in range(1, walls.height - 1):
            if not walls[x][y] and (x, y) != self.start:
                opponentBorder.append((x, y))
        self.oppoentBorder = opponentBorder
        
        # set first target position
        myPos = gameState.getAgentPosition(self.index)
        y_half = walls.height // 2
        foodList = self.getFood(gameState).asList()

        if len(foodList):
            if self.index < self.mateIndex:
                # if index is less than mate index, then target position is closest food in half upper region
                foodHalf = [f for f in foodList if f[1] > y_half]
                
                if len(foodHalf) > 0:
                    mateTarget = TEAM_STATE_DICT[self.mateIndex]['targetPos']
                    if mateTarget:
                        # get food list which is more than or equal to 8 steps away from teammate
                        foodListOption = [f for f in foodHalf if self.getMazeDistance(mateTarget, f) >= 8]

                        if len(foodListOption) > 0:
                            # set closest food from foodListOption1 as target
                            self.targetPos = min(foodListOption, key=lambda x: self.getMazeDistance(myPos, x))
                        else:
                            # set closest food among all food as target
                            self.targetPos = min(foodHalf, key=lambda x: self.getMazeDistance(myPos, x))
                    else:
                        # set closest food among all food as target
                        self.targetPos = min(foodHalf, key=lambda x: self.getMazeDistance(myPos, x))
                else:
                    self.targetPos = min(foodList, key=lambda x: self.getMazeDistance(myPos, x))
            else:
                # if index is no less than mate index, then target position is closest food in half lower region
                foodHalf = [f for f in foodList if f[1] <= y_half]
                
                if len(foodHalf) > 0:
                    mateTarget = TEAM_STATE_DICT[self.mateIndex]['targetPos']
                    if mateTarget:
                        # get food list which is more than or equal to 8 steps away from teammate
                        foodListOption = [f for f in foodHalf if self.getMazeDistance(mateTarget, f) >= 8]

                        if len(foodListOption) > 0:
                            # set closest food from foodListOption1 as target
                            self.targetPos = min(foodListOption, key=lambda x: self.getMazeDistance(myPos, x))
                        else:
                            # set closest food among all food as target
                            self.targetPos = min(foodHalf, key=lambda x: self.getMazeDistance(myPos, x))
                    else:
                        # set closest food among all food as target
                        self.targetPos = min(foodHalf, key=lambda x: self.getMazeDistance(myPos, x))
                else:
                    self.targetPos = min(foodList, key=lambda x: self.getMazeDistance(myPos, x))


        stateDict = {
            'targetPos': self.targetPos,
            'isDefence': self.isDefence
        }

        TEAM_STATE_DICT[self.index] = stateDict
    

    def observationFunction(self, gameState):
        """
        This is where we ended up after our last action.
        - Update target position if necessary.
        - Calculate reward
        - Update weights
        """

        if self.lastState:
            myState = gameState.getAgentState(self.index)
            myPos = myState.getPosition()
            foodList = self.getFood(gameState).asList()
            invaderList = self.getInvaderList(gameState)

            if myPos == self.start:
                self.specificPath = []

                # if food is remaining
                if len(foodList) > 0:
                    # set food or capsule as target
                    self.targetPos = self.getNextTargetFoodOrCapsule(gameState)
                
                # else if there is invader
                elif len(invaderList) > 0:
                    # set closest invader as target
                    self.targetPos = min(invaderList, key=lambda x: self.getMazeDistance(myPos, x))

                else:
                    # target is center point
                    self.targetPos = min(self.border, key=lambda x: self.getMazeDistance(myPos, x))

            # if current state is not starting position
            else:

                # if agent is on attack
                if not self.isDefence:
                    target = self.getNextTargetForOffence(gameState)

                    # if target is not None, update target
                    if target != None:
                        self.targetPos = target

                # if agent is on defence
                else:
                    target = self.getNextTargetForDefence(gameState)

                    # if target is not None, update target
                    if target != None:
                        self.targetPos = target

            # update team state dictionary
            stateDict = {
                'targetPos': self.targetPos,
                'isDefence': self.isDefence
            }
            TEAM_STATE_DICT[self.index] = stateDict

            # calculate reword and update weights
            if len(self.specificPath) == 0:
                reward = self.getReward(gameState)
                self.update(self.lastState, self.lastAction, gameState, reward)

        return gameState

    
    def getNextTargetForOffence(self, gameState):
        targetPos = None
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        myLastState = self.lastState.getAgentState(self.index)
        mateState = gameState.getAgentState(self.mateIndex)
        matePos = mateState.getPosition()
        distToTarget = self.getMazeDistance(myPos, self.targetPos)
        minDistToBorder = min([self.getMazeDistance(myPos, b) for b in self.border])
        closestBorder = min(self.border, key=lambda x: self.getMazeDistance(myPos, x))
        foodList = self.getFood(self.lastState).asList()
        capsuleList = self.getCapsules(self.lastState)
        ghostList = self.getGhostList(gameState)
        invaderList = self.getInvaderList(self.lastState)

        # if num of food remaining is no more than 2 and get back to home side
        if len(foodList) <= 2 and len(invaderList) > 0 and not myState.isPacman:
            
            # set closest invader as target
            targetPos = min(invaderList, key=lambda x: self.getMazeDistance(myPos, x))
            
            # make agent on defence
            self.isDefence = True
            
            return targetPos

        # if there is an invader in our side
        if len(invaderList) > 0:
            minDistToInvader = min([self.getMazeDistance(myPos, a) for a in invaderList])
            mateMinDistToInvader = min([self.getMazeDistance(matePos, a) for a in invaderList])

            # if agent is scared by opponent pacman
            # and if mate is not on defence
            # and if dist to invader < dist to targe
            # and if agent is closer to invader than mate
            if not myState.scaredTimer > 0 and not TEAM_STATE_DICT[self.mateIndex]['isDefence'] and minDistToInvader < distToTarget and  minDistToInvader < mateMinDistToInvader:
                        
                # set closest invader as target
                targetPos = min(invaderList, key=lambda x: self.getMazeDistance(myPos, x))
                
                # make agent on defence
                self.isDefence = True
                
                return targetPos

        # if timeup is coming or num of remainig food <= 2
        timeLeft = gameState.data.timeleft / gameState.getNumAgents()
        if timeLeft - minDistToBorder <= 4 or len(foodList) <= 2:
            
            # set closest border position as target 
            targetPos = closestBorder

            return targetPos

        # if agent is currying more than or equal to 8 foods and dist to border < sist to target
        if myState.numCarrying >= 8 and minDistToBorder < distToTarget and len(ghostList) > 0 and self.isOppoentsScared(gameState):
            
            # set closest border position as target 
            targetPos = closestBorder

            return targetPos

        # if agent is being chased by opponent ghost and opponent ghost is not scared
        if self.isChased(gameState) and not self.isOppoentsScared(gameState):
            
            # set closest border position as target 
            targetPos = closestBorder
           
            # if there is a capsule
            if len(capsuleList) > 0:
                minDistToCapsule =  min([self.getMazeDistance(myPos, c) for c in capsuleList])
                
                # if mate state is pacman and on offence
                if mateState.isPacman and not TEAM_STATE_DICT[self.mateIndex]['isDefence']:
                    # overwite the target to closest capsule
                    targetPos = min(capsuleList, key=lambda x: self.getMazeDistance(myPos, x))

                # if dist to closest food is closer than or equal to border
                elif minDistToCapsule <= minDistToBorder:
                    # overwite the target to closest capsule
                    targetPos = min(capsuleList, key=lambda x: self.getMazeDistance(myPos, x))

            return targetPos

        # if reaches target, food or capsule
        if myPos == self.targetPos or myPos in foodList or myPos in capsuleList:
            
            # set food or capsule as target
            self.targetPos = self.getNextTargetFoodOrCapsule(gameState)

        # if mate has already reached target
        if self.targetPos not in (foodList + capsuleList + self.border):

            # set food or capsule as target
            self.targetPos = self.getNextTargetFoodOrCapsule(gameState)

        return targetPos


    def getNextTargetForDefence(self, gameState):
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        targetPos = min(self.border, key=lambda x: self.getMazeDistance(myPos, x))
        mateState = gameState.getAgentState(self.mateIndex)
        matePos = mateState.getPosition()
        distToTarget = self.getMazeDistance(myPos, self.targetPos)
        foodList = self.getFood(gameState).asList()
        invaderList = self.getInvaderList(gameState)

        # if there is no invader
        if len(invaderList) == 0:

            # set food or capsule as target
            targetPos = self.getNextTargetFoodOrCapsule(gameState)
            # make agent on offence
            self.isDefence = False

            return targetPos
        
        # if there is an invader
        if len(invaderList) > 0:
            minDistToInvader = min([self.getMazeDistance(myPos, a) for a in invaderList])
            
            # if both agents are on defence and num of remaining food is >= 2
            if TEAM_STATE_DICT[self.mateIndex]['isDefence'] and len(foodList) >= 2:
                minDistToFood =  min([self.getMazeDistance(myPos, f) for f in foodList])
                mateMinDistToInvader = min([self.getMazeDistance(matePos, a) for a in invaderList])

                # if agent is closer to food than mate and dist to closest food
                if mateMinDistToInvader < minDistToInvader and minDistToFood < minDistToInvader:
                    
                    # set food or capsule as target
                    targetPos = self.getNextTargetFoodOrCapsule(gameState)
                    # make agent on offence
                    self.isDefence = False

                    return targetPos

             # if agent is scared by opponent pacman
            if myState.scaredTimer > 0:
                # set food or capsule as target
                targetPos = self.getNextTargetFoodOrCapsule(gameState)
                 # make agent on offence
                self.isDefence = False
                
                # delete specific path because if specific path exists, agent follow this path even though agent is attacked by oppoent
                self.specificPath = []
                
                return targetPos
        
            # if agent is closer to food than mate and dist to closest food * 1.2
            if len(foodList) >= 2:
                minDistToFood =  min([self.getMazeDistance(myPos, f) for f in foodList])

                if minDistToFood < minDistToInvader:
                    
                    # set food or capsule as target
                    targetPos = self.getNextTargetFoodOrCapsule(gameState)
                    # make agent on offence
                    self.isDefence = False

                    return targetPos

            # set invader's position as target
            targetPos = closestInvader = min(invaderList, key=lambda x: self.getMazeDistance(myPos, x))
            return targetPos

        return targetPos

        
    def getNextTargetFoodOrCapsule(self, gameState):
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        targetPos =  min(self.border, key=lambda x: self.getMazeDistance(myPos, x))
        mateState = gameState.getAgentState(self.mateIndex)
        matePos = mateState.getPosition()
        mateTarget = TEAM_STATE_DICT[self.mateIndex]['targetPos']
        distToTarget = self.getMazeDistance(myPos, self.targetPos)
        foodList = self.getFood(gameState).asList()      
        capsuleList = self.getCapsules(gameState)

        # if there is a food
        if len(foodList) > self.initialFoodNum / 2:
            # get food list which is closer than mate
            foodListOption = []

            for food in foodList:
                distToFood = self.getMazeDistance(myPos, food)
                mateDistToFood = self.getMazeDistance(matePos, food)
                
                if distToFood < mateDistToFood:
                    foodListOption.append(food)

            if len(foodListOption) > 0:
                # set closest food from foodListOption1 as target
                targetPos = min(foodListOption, key=lambda x: self.getMazeDistance(myPos, x))
            else:
                # set closest food among all food as target
                targetPos = min(foodList, key=lambda x: self.getMazeDistance(myPos, x))

        
        elif len(foodList) > 2:
            maxDist = 0
            furthestFood = None
            
            for food in foodList:
                closestBorder = min(self.border, key=lambda x: self.getMazeDistance(food, x))
                dist = self.getMazeDistance(myPos, food) + self.getMazeDistance(food, closestBorder)
                if maxDist < dist:
                    maxDist = dist
                    furthestFood = food 

            foodList.remove(furthestFood)

            maxDist = 0
            secondFurthestFood = None
            
            for food in foodList:
                closestBorder = min(self.border, key=lambda x: self.getMazeDistance(food, x))
                dist = self.getMazeDistance(myPos, food) + self.getMazeDistance(food, closestBorder)
                if maxDist < dist:
                    maxDist = dist
                    secondFurthestFood = food 

            foodList.remove(secondFurthestFood)

            # get food list which is closer than mate
            foodListOption = []

            for food in foodList:
                distToFood = self.getMazeDistance(myPos, food)
                mateDistToFood = self.getMazeDistance(matePos, food)
                
                if distToFood < mateDistToFood:
                    foodListOption.append(food)
                    
            if len(foodListOption) > 0:
                # set closest food from foodListOption1 as target
                targetPos = min(foodListOption, key=lambda x: self.getMazeDistance(myPos, x))
            else:
                # set closest food among all food as target
                targetPos = min(foodList, key=lambda x: self.getMazeDistance(myPos, x))

        # if there is a capsule and opponent ghost is not scared
        # and if mate state is pacman and on offence
        if len(capsuleList) > 0 and not self.isOppoentsScared(gameState, 20):
            minDistToCapsule =  min([self.getMazeDistance(myPos, c) for c in capsuleList])
            
            # if dist to capsule is closer than or equal to dist to target
            if minDistToCapsule <= distToTarget:
                # set closest capsule as target
                targetPos = min(capsuleList, key=lambda x: self.getMazeDistance(myPos, x))
           
        return targetPos


    def getFeatures(self, gameState, action):
        """
        Calculate features.
        """

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        successor = self.getSuccessor(gameState, action)
        myNextState = successor.getAgentState(self.index)
        myNextPos = myNextState.getPosition()
        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        ghostList = self.getGhostList(successor)
        invaderList = self.getInvaderList(successor)

        features = util.Counter()
        
        # this process is to make weights['distToTarget']*features['distToTarget'] always negative
        if self.weights['distToTarget'] > 0:
            features['distToTarget'] = -self.getMazeDistance(myNextPos, self.targetPos) / self.mazeSize
        else:
            features['distToTarget'] = self.getMazeDistance(myNextPos, self.targetPos) / self.mazeSize


        if myNextState.isPacman and not self.isOppoentsScared(gameState) and len(ghostList) > 0:
            features['distToOpponent'] = min([self.getMazeDistance(myNextPos, a) for a in ghostList]) / self.mazeSize
        
        elif not myNextState.isPacman and myNextState.scaredTimer > 0 and len(invaderList) > 0:
            features['distToOpponent'] = min([self.getMazeDistance(myNextPos, a) for a in invaderList]) / self.mazeSize

        return features

    
    def chooseAction(self, gameState):
        """
        Choose next action.
        Call "getOffenceAction" or "getDefenceAction" based on current agent state, on offence or on defence. 
        """

        if not self.isDefence:
            return self.getOffenceAction(gameState)
        else:
            return self.getDefenceAction(gameState)


    def getOffenceAction(self, gameState):
        """
        Return next best action of offensive agent. 
        """

        # You can profile your evaluation time by uncommenting these lines
        start = time.time()

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        ghostList = self.getGhostList(gameState)
        invaderList = self.getInvaderList(gameState)

        # if self.lastState:
        #     if len(ghostList) > 0
        #         myLastState = self.lastState.getAgentState(self.index)
        #         minDistToGhost =  min([self.getMazeDistance(myPos, a) for a in ghostList])

        #         # if try to go get food, but there is oppoent ghost close to agent, find alternative path
        #         if not myState.isPacman and not self.isDefence and minDistToGhost <= 3 and self.specificPath == []:
                    
        #             path, target = self.getAlternativePath(gameState, isPacmanPenalize=True)
        #             self.specificPath = path
        #             self.targetPos = target

        # # if specific path exists, get next action fron specific path list
        if len(self.specificPath) > 0:
            return self.specificPath.pop(0)
        
        # if agent is getting stuck
        elif self.isStucking(gameState):
            if len(ghostList) > 0:
                minDistToGhost =  min([self.getMazeDistance(myPos, a) for a in ghostList])

                # find alternative path
                actions, target = self.getAlternativePath(gameState, maxPathLength=1)
                if len(actions) > 0:
                    self.specificPath = actions
                    self.targetPos = target
                    return self.specificPath.pop(0)
                else:
                    actions = gameState.getLegalActions(self.index)
                    return random.choice(actions)

        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')

        if len(ghostList) > 0 and not self.isOppoentsScared(gameState):
            distToGhost = min([self.getMazeDistance(myPos, a) for a in ghostList])
            
            actionToCapsule = []
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                myNextState = successor.getAgentState(self.index)
                myNextPos = myNextState.getPosition()

                if myNextPos in capsuleList:
                    actionToCapsule.append(action)

            # action to capsule is found
            if len(actionToCapsule) > 0:
                # update next action options
                actions =  actionToCapsule

            # if dist ot opponent ghost is more than or equal to 6
            elif distToGhost <= 8:
                # if there is capsule, safe action to capsule
                if len(capsuleList) > 0:
                    safeActions = self.getSafeActions(gameState, capsuleList)
                # else if agent is on border, safe action to next target food or capsule
                elif myPos in self.border:
                    safeActions = self.getSafeActions(gameState, [self.getNextTargetFoodOrCapsule(gameState)])
                # else safe action to border
                else:
                    safeActions = self.getSafeActions(gameState)

                if len(safeActions) > 0:
                    actions = safeActions
        
        if len(invaderList) > 0 and not myState.isPacman and myState.scaredTimer > 0:
            safeActions = self.getSafeActions(gameState, [self.targetPos])

            if len(safeActions) > 0:
                actions = safeActions

        values = [self.getQValue(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        bestAction = random.choice(bestActions)
        
        self.doAction(gameState, bestAction)

        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        return bestAction


    def getDefenceAction(self, gameState):
        """
        Return next best action of defensive agent. 
        """

        actions = gameState.getLegalActions(self.index)

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        ghostList = self.getGhostList(gameState)
        invaderList = self.getInvaderList(gameState)

        if len(self.specificPath) > 0:
            return self.specificPath.pop(0)
        
        # if agent is scared by oppoent pacman, find safe action and agent state is ghost
        if myState.scaredTimer > 0 and len(invaderList) > 0 and not myState.isPacman:
            # safe action to border
            safeActions = self.getSafeActions(gameState)
            if len(safeActions) > 0:
                actions = safeActions
        
        # if agent is pacman and there is a ghost not scared
        if myState.isPacman and len(ghostList) > 0 and not self.isOppoentsScared(gameState):
            # safe action to target
            safeActions = self.getSafeActions(gameState, [self.targetPos])

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
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        myLastPos = self.lastState.getAgentPosition(self.index)
        foodList = self.getFood(self.lastState).asList()
        capsuleList = self.getCapsules(self.lastState)

        distToTarget = self.getMazeDistance(myPos, self.targetPos)
        lastDistToTarget = self.getMazeDistance(myLastPos, self.targetPos)

        if myPos in foodList:
            reward += 1
        elif myPos in capsuleList:
            reward += 2
        else:
            reward += (lastDistToTarget - distToTarget) / self.mazeSize

        return reward


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
        Update last state and action
        """

        self.lastState = gameState
        self.lastAction = action


    def getQValue(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """

        features = self.getFeatures(gameState, action)
        return features * self.weights
    

    def update(self, gameState, action, nextState, reward):
        """
        Update weights.
        """
        actions = nextState.getLegalActions(self.index)
        values = [self.getQValue(nextState, a) for a in actions]
        maxValue = max(values)
        features = self.getFeatures(gameState, action)
        
        diff = (reward + self.discout * maxValue) - self.getQValue(gameState, action)

        for feature in features:
            self.weights[feature] += self.alpha * diff * features[feature]


    def getSafeActions(self, gameState, targets=None):
        """
        Return safe action to avoid the dead end (path surrounded by walls except for one direction).
        Use A* search, penalize current position because when agent goes to the dead end, it should visit current position again to come back.
        """

        if targets == None:
            targets = self.border

        safeActions = []
        myPos = gameState.getAgentPosition(self.index)
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')

        for action in actions:
            successor = self.getSuccessor(gameState, action)

            finalNode = self.aStarSearch(successor, targets, [myPos])
            if finalNode[2] < self.mazeSize:
                safeActions.append(action)
        
        return safeActions


    def aStarSearch(self, gameState, goals, penaltyPos=[]):
        """
        A* search.
        Penalize the position in penaltuPos.
        If avoidGhost is True, penalize position of oppoent's ghost.
        """
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        walls = gameState.getWalls().asList()
        ghostList = self.getGhostList(gameState)
        invaderList = self.getInvaderList(gameState)
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        startPos = gameState.getAgentPosition(self.index)

        cost = 0
        # penalize opponent's ghost position if ghost is not scared
        if myState.isPacman and not self.isOppoentsScared(gameState) and len(ghostList) > 0:
            if startPos in ghostList:
                cost += self.mazeSize
            elif (startPos[0] + 1, startPos[1]) in ghostList:
                cost += self.mazeSize
            elif (startPos[0] - 1, startPos[1]) in ghostList:
                cost += self.mazeSize
            elif (startPos[0], startPos[1] + 1) in ghostList:
                cost += self.mazeSize
            elif (startPos[0], startPos[1] - 1) in ghostList:
                cost += self.mazeSize

        # penalize oppoents's pacman position if agent is scared by opponent's pacman
        if not myState.isPacman and myState.scaredTimer > 0 and len(invaderList) > 0:
            if startPos in invaderList:
                cost += self.mazeSize
            elif (startPos[0] + 1, startPos[1]) in invaderList:
                cost += self.mazeSize
            elif (startPos[0] - 1, startPos[1]) in invaderList:
                cost += self.mazeSize
            elif (startPos[0], startPos[1] + 1) in invaderList:
                cost += self.mazeSize
            elif (startPos[0], startPos[1] - 1) in invaderList:
                cost += self.mazeSize

        if myPos in self.border or myPos in self.oppoentBorder:
            if startPos in ghostList or startPos in invaderList:
                cost += self.mazeSize
            elif (startPos[0] + 1, startPos[1]) in ghostList or (startPos[0] + 1, startPos[1]) in invaderList:
                cost += self.mazeSize
            elif (startPos[0] - 1, startPos[1]) in ghostList or (startPos[0] - 1, startPos[1]) in invaderList:
                cost += self.mazeSize
            elif (startPos[0], startPos[1] + 1) in ghostList or (startPos[0], startPos[1] + 1) in invaderList:
                cost += self.mazeSize
            elif (startPos[0], startPos[1] - 1) in ghostList or (startPos[0], startPos[1] - 1) in invaderList:
                cost += self.mazeSize

        currentNode = (startPos, [], cost)
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

                    # penalize position in penaltyPos list
                    if successor[0] in penaltyPos:
                        cost += self.mazeSize

                    # penalize opponent's ghost position if ghost is not scared
                    if myState.isPacman and not self.isOppoentsScared(gameState) and len(ghostList) > 0:
                        if successor[0] in ghostList:
                            cost += self.mazeSize
                    #     elif (successor[0][0] + 1, successor[0][1]) in ghostList:
                    #         cost += self.mazeSize
                    #     elif (successor[0][0] - 1, successor[0][1]) in ghostList:
                    #         cost += self.mazeSize
                    #     elif (successor[0][0], successor[0][1] + 1) in ghostList:
                    #         cost += self.mazeSize
                    #     elif (successor[0][0], successor[0][1] - 1) in ghostList:
                    #         cost += self.mazeSize

                    # penalize oppoents's pacman position if agent is scared by opponent's pacman
                    if not myState.isPacman and myState.scaredTimer > 0 and len(invaderList) > 0:
                        if successor[0] in invaderList:
                            cost += self.mazeSize
                    #     elif (successor[0][0] + 1, successor[0][1]) in invaderList:
                    #         cost += self.mazeSize
                    #     elif (successor[0][0] - 1, successor[0][1]) in invaderList:
                    #         cost += self.mazeSize
                    #     elif (successor[0][0], successor[0][1] + 1) in invaderList:
                    #         cost += self.mazeSize
                    #     elif (successor[0][0], successor[0][1] - 1) in invaderList:
                    #         cost += self.mazeSize

                    pQueue.push((position, path, cost))

        return currentNode


    def isOppoentsScared(self, gameState, timer=6):
        """
        Check whether oppoent ghost is scared
        """

        myPos = gameState.getAgentPosition(self.index)
        opponentList = self.getOpponents(gameState)
        
        closestOpponent = min(opponentList, key=lambda x: self.getMazeDistance(myPos, gameState.getAgentPosition(x)))
        
        return gameState.getAgentState(closestOpponent).scaredTimer >= timer


    def isChased(self, gameState, chasedCount=8, minDist=2):
        """
        Check whether agent is being chased by oppoent's ghost
        """

        history = self.observationHistory
        myState = gameState.getAgentState(self.index)
        ghostList = self.getGhostList(gameState)
        
        if len(history) == 0 or len(ghostList) == 0:
            return False

        myPos = myState.getPosition()
        distToGhost = min([self.getMazeDistance(myPos, a) for a in ghostList])
        
        if distToGhost > minDist:
            return False

        for i in range(min(chasedCount, len(history))):
            pastState = history[-i - 1]
            myPastPos = pastState.getAgentPosition(self.index)
            pastGhosts = self.getGhostList(pastState)
            
            if len(pastGhosts) == 0:
                return False
            
            pastDistToGhost = min([self.getMazeDistance(myPastPos, a) for a in pastGhosts])
 
            if pastDistToGhost != distToGhost:
                return False
            
        return True


    def getGhostList(self, gameState):
        """
        Return list of oppoent's ghost position.
        """

        enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
        ghostList = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition()]
        return ghostList


    def getInvaderList(self, gameState):
        """
        Return list of oppoent's invader position.
        """

        enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
        invaderList = [a.getPosition() for a in enemies if a.isPacman and a.getPosition()]
        return invaderList


    def getAlternativePath(self, gameState, maxPathLength=3, isPacmanPenalize=False, penaltyDist=2, exploreRange=3):
        """
        Return list of alternative actions and target.
        """

        walls = gameState.getWalls()
        myPos = gameState.getAgentPosition(self.index)
        ghostList = self.getGhostList(gameState)
        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)

        mateTarget = TEAM_STATE_DICT[self.mateIndex]['targetPos']

        # get food list which is more than or equal to 8 steps away from teammate
        foodListOption = [f for f in foodList if self.getMazeDistance(mateTarget, f) >= 10]

        if len(foodListOption) > 0:
            # update food list
            foodList = foodListOption
    
        targetList = foodList + capsuleList
       
        if len(foodList) <= 2:
            targetList = self.border
        
        penaltyPos = []
        
        for ghost in ghostList:
            for x in range(int(max(1, myPos[0] - exploreRange)), int(min(myPos[0] + exploreRange, walls.width))):
                for y in range(int(max(1, myPos[1] - exploreRange)), int(min(myPos[1] + exploreRange, walls.height))):
                    pos = (int(x), int(y))
                    if not pos in walls.asList():
                        distToGhost = self.getMazeDistance(pos, ghost)

                        # if dist to ghost is closer than penalty dist
                        if distToGhost <= penaltyDist:
                            penaltyPos.append(pos)

                        # if isPacmanPenalize is true, avoid to go to oppoent side.
                        if isPacmanPenalize:
                            borderX = self.border[0][0]
                            if self.red:
                                if x > borderX:
                                    penaltyPos.append(pos)
                            else:
                                if x < borderX:
                                    penaltyPos.append(pos)

                    # if position is already in target list or team mate target list, remove this position. Because this function is to find alternative path and target.
                    if pos in targetList or pos in TEAM_STATE_DICT[self.mateIndex]['targetPos']:
                        targetList.remove(pos)

        if len(targetList) == 0:
            return [], None

        finalNode = self.aStarSearch(gameState, targetList, penaltyPos)
        pathLength = min(maxPathLength, len(finalNode[1]))
        return finalNode[1][0:pathLength], finalNode[0]


    def isStucking(self, gameState, stuckingCount=3):
        """
        Check whether agent is getting stuck at same position, going and back.
        """

        history = self.observationHistory
        count = 0
        myPos = gameState.getAgentPosition(self.index)
        
        if len(history) > 0:
            for i in range(min(10, len(history))):
                myPastPos = history[-i - 1].getAgentPosition(self.index)
                if myPastPos == myPos:
                    count += 1
            
        return count >= stuckingCount