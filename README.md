## Improving AI functionalities in Pacman

In this project, I apply an array of AI techniques to improve the efficiency of the classic "Pacman" game. The project is broken down into three parts:

- Part 1 - Informed State-Space Search
- Part 2 - Multi-Agent Search
- Part 3 - MDPs and Reinforcement Learning 


##### Notes
- Each part is in its own folder. 



### Part 1 - Informed State-Space Search
Applying graph search algorithms to PacMan with no adversaries in the maze. Implementing depth-first, breadth-first, uniform cost, and A* search algorithms. These algorithms are used to solve navigation and traveling salesman problems in the Pacman world. 

#### Search algorithms
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


#### Eating foods on corners: heuristic
The heuristic was obtained by relaxation, assuming there are no walls in
the maze. It is obtained by summing:
1. the Manhattan distance to the nearest unvisited corner
2. the shortest Manhattan path from this corner to the remaining corners
(if any)

The second term is pre-computed for the cases in which the unvisited corners are 3 or 4, 
even though it wouldn't be expensive to compute.

#### Eating all dots: heuristic
The heuristic sums:
- the minimum cost for reaching any dot
- the maximum cost for going from the "nearest" dot found in the previous
step to another dot

The costs are obtained computing the optimal path to each single dot and
are cached in a dictionary to save computation.


### Part 2 - Multi-Agent Search
Classic Pacman is modeled as both an adversarial and a stochastic search problem. Implemented here:  multiagent minimax and expectimax algorithms, as well as designing evaluation functions.

### Part 3 - MDPs and Reinforcement Learning
Objective here is to develop a Pacman agent using reinforcement learning. Implemented model-based and model-free reinforcement learning algorithms and applied them to the AIMA textbook’s Gridworld, Pacman, and a simulated crawling robot. 
