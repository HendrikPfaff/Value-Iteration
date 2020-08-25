# Value Iteration in the Gridworld

Our autonomous car wants to move accross a two-dimensional grid from its starting position to its goal. The car should not collide with any obstacle/wall (yet is allowed to refuel at a gas station) while only the movement to its next adjacent cell is possible.

![Picture of the gridworld for our car](https://hendrikpfaff.de/img/valueiteration/map.png)

How can we find the best route for the car using the reinforcement learning method of [Value Iteration](https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration)? 

## Introduction
This Project aims to achieve three objectives:
* Implementing the Value Iteration algorithm for a two dimensional gridworld.
* Finding the optimal value function (_V*_) and policy (_pi*_).
* Observe and visualize the learning process.

## Modelling the MVP

Before we start programming, we need to consider the underlying constraints and rules of our gridworld. Out of these, a [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) (_MDP_) can be defined as a tuple of (S, A, p, c).
* S (= the set of states): 
    
    Every cell of the grid is a distinct state of the MDP (s11, s12, s13, ..., s28). The cells s13, s26 and s36 as well as the outer border of the map are defined as _s_wall_ and can not be driven onto. The cell s34 is defined as _s_gasstation_. Finally our goal / absorbent terminal state _s0_ is reached in s38. 

* A (= the set of available actions):

    The only available actions to choose from in any cell are _UP_, _DOWN_, _LEFT_ and _RIGHT_.

* p (= the probability function for actually executing the intended action):

    The probability of executing an action, transitioning from state _s_ to state _s'_, is defined as _0.8_ which means every one of the remaining actions (including staying in the cell) is executed with the probability of _0.05_.
    
* c (= the cost function of executing a certain action):

    The costs for transitioning from one state _s_ to the next, are dependent on how the resulting state _s'_ is defined. When reaching the terminal state (_s'_ = _s0_), the car receives negative costs of _-10_. On entering (not for staying at) the gas station (_s'_ = _s_gasstation_), negative costs of _-1_ are given. Collisions with walls are penalized with costs of _+2_, while costs of _+1_ arise for every other transition.    
    
Due to the existence of the absorbent terminal, from which no further action or costs are applicable, we can define this MDP as a _stochastic shortest path problem_ and thus relinquish from a discount factor (i.e. set the discount factor to _1.0_).

![Drawing of the MDP](https://hendrikpfaff.de/img/valueiteration/mdp_en.png)

## Implementation

### Initializing

### Value Iteration Algorithm

### Problems regarding standard parameters

## Learning Process