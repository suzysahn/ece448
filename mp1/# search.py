# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)
from collections import deque
def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "extra": extra,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
     # variables and data structure declared here   
    start = maze.getStart()
    frontier = []
    explored = set()

    # add the start to frontier
    frontier.append([start])

    while frontier:
        # initialize/update path and position
        path = frontier.pop(0)
        x,y = path[-1]

        # if never explored, add it to our explored set
        if (x, y) not in explored:
            explored.add((x, y))
            # return path if we reached goal
            if maze.isObjective(x, y):
                return path
            # in each not explored neighbor, add new path length
            for pos in maze.getNeighbors(x, y):
                if pos not in explored:
                    frontier.append(path + [pos])
    return []
    
    # tiny maze path debugging
    # return [(1,5),(2,5),(3,5),(3,4),(4,4),(4,3),(4,2),(5,2),(5,1)]

   
# consulted from wikipedia page on a* algorithm
# https://en.wikipedia.org/wiki/A*_search_algorithm
def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return astar_base(maze, 0)

def astar_base(maze, n):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    if (len(maze.getObjectives()) <= n):
        return []
    end = maze.getObjectives()[n]
    # intialize frontier + explored set 
    frontier = {}
    frontier[start] = get_h(start, end)
    prev = {}
    gDist = {} # g(n) = distance already traveled, calculated by bfs
    fDist = {} # f(n) = distance already traveled + manhattan distance to goal from curr state

    gDist[start] = 0
    fDist[start] = frontier[start]

    while(len(frontier.keys()) != 0):
        #  the node in frontierSet having the lowest fScore[] value
        cur = min_fDist(frontier, fDist)
        if (cur == end):
            return getPrevPath(prev, cur)

        frontier.pop(cur)

        for neighbor in maze.getNeighbors(cur[0],cur[1]):
            gDistNew = gDist[cur] + 1

            if (neighbor not in gDist): #construct path
                gDist[neighbor] = float('inf')

            if (gDistNew < gDist[neighbor]):
                prev[neighbor] = cur
                gDist[neighbor] = gDistNew
                fDist[neighbor] = gDist[neighbor] + man_dist(neighbor, end)

                if (neighbor not in frontier):
                    frontier[neighbor] = fDist[neighbor]
    return [] 

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.
        
    @param maze: The maze to execute the search on.
        
    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    goals = maze.getObjectives()
    curPos = maze.getStart()
    path = deque()
    reached = deque()
    path.appendleft(curPos)

    unvisitedCorners = maze.getObjectives()
        while len(unvisitedCorners):
            cornerDist = []
            for corner in unvisitedCorners:
                distance = man_dist(curr, corner)
                cornerDist.append((distance, corner))
            currentDist, currentCorner = min(cornerDist)
    # ((x,y),{})
    state = {}

    # corner heuristic
    curr = maze.getStart()
    heuristic = 0
    unvisitedCorners = maze.getObjectives()
    while len(unvisitedCorners):
        cornerDist = []
        for corner in unvisitedCorners:
            distance = man_dist(curr, corner)
            cornerDist.append((distance, corner))
        currentDist, currentCorner = min(cornerDist)
        heuristic += currentDist
        curr = currentCorner
        unvisitedCorners.remove(currentCorner)
            


    firstList = astar_base(maze, 0)
    firstSize = len(firstList)

    for n in range(1, len(maze.getObjectives())):
        nextList = astar_base(maze, n)
        nextSize = len(nextList)

        if (firstSize > nextSize):
            firstList = nextList
            firstSize = nextSize

    return firstList

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []


def extra(maze):
    """
    Runs extra credit suggestion.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []

# -------------------------------------------------------------------------------------------------
#           Private Functions
# -------------------------------------------------------------------------------------------------

def man_dist(start, end):
    x = abs(end[0] - start[0])
    y = abs(end[1] - start[1])
    sum = x + y
    return sum

def min_fDist(frontier, fDist):
    # node in frontier set having the lowest fDist value
    minDist = float('inf')
    minIdx = ()

    for idx in frontier.keys():
        # problematic, pls fix suzy
        if (fDist[idx] < minDist):
            minDist = fDist[idx]
            minIdx = idx
    return minIdx


def getPrevPath(prev, curr):
    sumPath = [curr]
    while (curr in prev.keys()): 
        curr = prev[curr]
        sumPath.append(curr)
    sumPath.reverse()
    return sumPath

def get_h(start, end):
    sum = 0
    for i in range(0,2):
        sum += abs(start[i]-end[i])
    return sum
