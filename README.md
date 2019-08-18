## Projects of the "Artificial Intelligence" course (CS188, UC Berkeley)

This repository contains my solutions to the projects of the course 
of "Artificial Intelligence" (CS188) taught by Pieter Abbeel and Dan Klein
at the UC Berkeley. 

I'm using the material from Fall 2018 (https://inst.eecs.berkeley.edu/~cs188/fa18/).

The output of the auto-grader is saved in each project folder as `autograder.out`.

### Project 1 - Graph search - Implementation Notes

Project 1 is about applying graph search algorithms to PacMan (with no adversaries in the maze):
https://inst.eecs.berkeley.edu/~cs188/fa18/project1.html

#### Question 1-4 - Search algorithms
All the search algorithms variants were implemented using a single 
generic search function and various `Fringe` implementations, one for each
search variant: 
- for DFS, it is stack 
- for BSF, it is queue 
- for UCS, it is a priority queue 
- for A*, it is a priority queue whose keys are computed summing 
the backward cost (as in UCS) to the estimated forward cost computed
by the provided heuristic.

States are wrapped in a `SearchNode` that stores: 

- the cost to reach the node,
- the previous node, 
- the action that led to the node. 

The path from the start node to a given node, as well as the 
corresponding action plan, is easily retrieved from the node itself 
by going backward to the start node.

Alternative implementations could:
- store the entire path in the node itself
- store the previous node in an external dictionary.

#### Question 6 - Eating foods on corners: heuristic
The heuristic was obtained by relaxation, assuming there are no walls in
the maze. It is obtained by summing:
1. the Manhattan distance to the nearest unvisited corner
2. the shortest Manhattan path from this corner to the remaining corners
(if any)

The second term is pre-computed for the cases in which the unvisited corners are 3 or 4, 
even though it wouldn't be expensive to compute.

#### Question 7 - Eating all dots: heuristic
The heuristic sums:
- the minimum cost for reaching any dot
- the maximum cost for going from the "nearest" dot found in the previous
step to another dot

The costs are obtained computing the optimal path to each single dot and
are cached in a dictionary to save computation.

In the auto-grading problem, the number of expanded nodes using the above heuristic (719) was way less than the maximum 
required for the maximum score (7000).
