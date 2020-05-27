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

import heapq
import queue as queue
import math
from copy import deepcopy

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
    queue = []
    visited = set()
    queue.append([maze.getStart()])
    while queue:
        cur_path = queue.pop(0)
        cur_row, cur_col = cur_path[-1]
        if (cur_row, cur_col) in visited:
            continue
        visited.add((cur_row, cur_col))
        if maze.isObjective(cur_row, cur_col):
            return cur_path
        for item in maze.getNeighbors(cur_row, cur_col):
            if item not in visited:
                queue.append(cur_path + [item])
    return []


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return extra(maze)


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.
        
    @param maze: The maze to execute the search on.
        
    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return extra(maze)


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return extra(maze)


def extra(maze):
    """
    Runs extra credit suggestion.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    goals = maze.getObjectives()
    
    # create states from mazeRaw when coord is not wall
    states = set()
    rows, cols = maze.getDimensions()
    for row in range(rows):
        for col in range(cols):
            if not maze.isWall(row, col):
                states.add((row, col))
                
    # use better heuristics for entire allay
    distances = get_distances(states, goals)
    
    # init node list with distances from start
    nodes = []         
    nodes.append((start, distances.get(start), 0))

    # init path list
    path = [] 
    path.append(start)
    
    # init visited dict
    visited = {}
    visited[start] = start
    
    while nodes:
        minNode = min(nodes, key = lambda x:x[1])
        coord, _, cost = minNode
        nodes.remove(minNode)

        if coord in goals:
            # print("coord = ", coord")
            goals.remove(coord)
            distances = get_distances(states, goals)
            path.extend(visited_to_path(visited, coord))
            visited.clear()
            visited[coord] = coord
            nodes = []

        if len(goals) == 0:
            # print("finished")
            return path

        for neighbor in maze.getNeighbors(coord[0], coord[1]):
            if neighbor not in visited:
                nodes.append((neighbor, cost + distances.get(neighbor), cost + 1))
                visited[neighbor] = coord

    return []

# =============================================================
#   Supplemental functions
# =============================================================

def manhattan(p1, p2):
	return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_distances(states, goals):
    results = {}
    for state in states:
        if state in goals:
            results[state] = 0
            continue

        smallest = -1
        for goal in goals:
            dist = manhattan(state, goal)
            smallest = dist if (dist < smallest or smallest == -1) else smallest
            if smallest == 1:
                break
        results[state] = smallest

    return results

def visited_to_path(visited, goal):
	path = []
	curr, prev = goal, visited.get(goal)
	while curr != prev and curr != None:
		path.insert(0, curr)
		curr = prev
		prev = visited.get(prev)
	return path
