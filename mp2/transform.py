
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    # get alpha and beta limit
    alpha, beta = arm.getArmLimit()
    
    # get rows and cols and allocate maze array
    rows, cols = angleToIdx([alpha[1], beta[1]], [alpha[0], beta[0]], granularity)
    rows, cols = [rows + 1, cols + 1]   # increase for counting
    maze = [[0 for x in range(cols)] for y in range(rows)]
    
    angles = arm.getArmAngle()
    pos_y = (angles[0] - alpha[0]) // granularity
    pos_x = (angles[1] - beta[0])  // granularity
    
    granularity = math.floor(granularity)   # for fast calc
    
    for row in range(rows):
        for col in range(cols):
            arm.setArmAngle(((row * granularity + alpha[0]), (col * granularity + beta[0])))
            cell = SPACE_CHAR   # for default
            if not isArmWithinWindow(arm.getArmPos(), window):
                cell = WALL_CHAR
            elif doesArmTouchObjects(arm.getArmPosDist(), obstacles, False):
                cell = WALL_CHAR
            elif doesArmTouchObjects(arm.getArmPosDist(), goals, True):
                cell = OBJECTIVE_CHAR if doesArmTipTouchGoals(arm.getEnd(), goals) else WALL_CHAR
            maze[row][col] = cell

    # override START_CHAR to start position
    maze[pos_y][pos_x] = START_CHAR
    
    # allocate Maze object and return     
    maze_obj = Maze(maze, list([alpha[0], beta[0]]), granularity)

    return maze_obj
