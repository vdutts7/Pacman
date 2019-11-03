## Projects of the "Artificial Intelligence" course (CS 188, UC Berkeley)

This repository contains my solutions to the projects of the course 
of "Artificial Intelligence" (CS188) taught by Pieter Abbeel and Dan Klein
at the UC Berkeley. I'm using the material from 
[Fall 2018](https://inst.eecs.berkeley.edu/~cs188/fa18/).

- [x] Project 1 - Search
- [x] Project 2 - Multi-agent Search
- [x] Project 3 - MDPs and Reinforcement Learning 
- [x] Project 4 - Ghostbusters (HMMs, Particle filtering, Dynamic Bayes Nets)
- [ ] ~~Project 5 - Machine learning~~ (I won't do this because it is about neural networks, topic I've already studied at a deeper level)

##### Notes
- Each project has his own folder. 
For each project, the output of the auto-grader is saved as `autograder.out` 
inside the project folder.

- I added a `setup.py` file and installed the root folder as a package (in editable mode) with 

        pip install -e . 
    
  I did it for not having import issues when importing stuff from past projects or from
  my_utils.py (and because PyCharm is happier this way).

- For the sake of clarity, my additional comments in the code start with the 
  character `§`.


### Project 1 - Graph search - Implementation Notes
[Project 1](https://inst.eecs.berkeley.edu/~cs188/fa18/project1.html) is about applying 
graph search algorithms to PacMan (with no adversaries in the maze)

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

### Project 2 - Multi-Agent Search
[Project 2](https://inst.eecs.berkeley.edu/~cs188/fa18/project2.html) is about using 
MiniMax ed ExpectiMax to implement a PacMan agent.

### Project 3 - MDPs and Reinforcement Learning
[Project 3](https://inst.eecs.berkeley.edu/~cs188/fa18/project3.html) is about developing 
a PacMan agent using reinforcement learning.

As an extra exercise, I wrote an additional feature extractor for PacMan called 
`CustomExtractor` that is a slightly modified version of the provided `SimpleExtractor`;
it just encourages the agent to eat adjacent scared ghosts instead of avoiding them as 
they were not scared. Of course, this alone increases a lot the average score. 

For fitting and evaluating an agent using `CustomExtractor` on `mediumClassic`
maze, run:

    python pacman.py -p ApproximateQAgent -a extractor=CustomExtractor -x 50 -n 60 -l mediumClassic 


### Project 4 - Ghostbusters
[Project 4](https://inst.eecs.berkeley.edu/~cs188/fa18/project4.html#Q4) is about 
Hidden Markov Models and Particle Filtering.

Problem: the maze is populated with `numGhosts` _invisible_ ghosts and we want PacMan to
catch them all; we don't know where the ghosts are precisely, but we are given some noisy
distances from PacMan to them.

The assignment can be divided into 3 parts:
1. in part 1, the problem is solved using the forward algorithm for HMM (exact inference);
2. in part 2, the problem is solved using approximate inference powered by particle filtering;
3. in part 3, ghosts don't move independently from each other, so the model is described
   by a Dynamic Bayes Net; the problem is still solved by using particle filtering;
   the difference is that rather than using `numGhosts` independent `ParticleFilter`s,
   we now have a single `JointParticleFilter` whose particles are tuples of positions 
   (one for each ghost).
