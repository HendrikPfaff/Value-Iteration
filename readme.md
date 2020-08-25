# Value Iteration in the Gridworld

Our autonomous car wants to move accross a two-dimensional grid from its starting position to its goal. The car should not collide with any obstacle/wall (yet is allowed to refuel at a gas station) while only the movement to its next adjacent cell is possible.

![Picture of the gridworld for our car](https://hendrikpfaff.de/img/valueiteration/map.png)

How can we find the best route for the car using the reinforcement learning method of [Value Iteration](https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration)? 

## Introduction
This Project aims to achieve three objectives:
* Implementing the Value Iteration algorithm for a two dimensional gridworld (based on [Mohammad Ashrafs work](https://towardsdatascience.com/reinforcement-learning-demystified-solving-mdps-with-dynamic-programming-b52c8093c919)) in python.
* Finding the optimal value function (_V*_) and policy (_pi*_).
* Observe and visualize the learning process.

## Modelling the MVP

Before we start programming, we need to consider the underlying constraints and rules of our gridworld. Out of these, a [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) (_MDP_) can be defined as a tuple of (S, A, p, c).
* S (= the set of states): 
    
    Every cell of the grid is a distinct state of the MDP (s11, s12, s13, ..., s28). The cells s13, s26 and s36 as well as the outer border of the map are defined as _s_wall_ and can not be driven onto. The cell s34 is defined as _s_gasstation_. Finally our goal / absorbent terminal state _s0_ is reached in s38. 

* A (= the set of available actions):

    The only available actions to choose from in any cell are _UP_, _DOWN_, _LEFT_ and _RIGHT_.

* p (= the probability function for actually executing the intended action):

    The probability of executing an action, transitioning from state _s_ to state _s'_, is defined as _0.8_ (_1.0 would be determinism_) which means every one of the remaining actions (including staying in the cell) is executed with the probability of _0.05_.
    
* c (= the cost function of executing a certain action):

    The costs for transitioning from one state _s_ to the next, are dependent on how the resulting state _s'_ is defined. When reaching the terminal state (_s'_ = _s0_), the car receives negative costs of _-10_. On entering (not for staying at) the gas station (_s'_ = _s_gasstation_), negative costs of _-1_ are given. Collisions with walls are penalized with costs of _+2_, while costs of _+1_ arise for every other transition.    
    
Due to the existence of the absorbent terminal state, from which no further action or costs are applicable, we can define this MDP as a _stochastic shortest path problem_ and thus relinquish from a discount factor (i.e. set the discount factor to _1.0_).

![Drawing of the MDP](https://hendrikpfaff.de/img/valueiteration/mdp_en.png)

## Implementation

The implementation consists of three parts. The `main.py`, the `environment.py` and the `map`-file.

* The `map`-file contains an ASCII-representation of gridworld and thus the information of which states are directly connected. Many different maps can be created and used. All states are defined by distinct symbols that are later parsed to create the environment. 
  * `X`: _s_wall_, which surrounds the whole grid and poses some obstacles to navigate around.
  * `S`: Starting state/position of the car. 
  * `T`: The gas station (german: '_Tankstelle_'), which gives a small reward.
  * `G`: Goal / terminal state.
  * `.`: Every other state.

* The `environment.py` contains the `Environment`-class, that controls all environmental parameters and behaviors. In its constructor, the `map`-file is loaded and parsed into a usable gridworld. In the class constants, all relevant costs, probabilities and actions are defines.

* In the `main.py`, the actual value iteration takes place. It is based on [Mohammad Elsersys Value Iteration code](https://gist.github.com/Neo-47/b8f6af451211d43ceaf950cfea1ded96), modified for our scenario. Every state includes the value of its next optimal state (see [Bellmann Principle](https://en.wikipedia.org/wiki/Bellman_equation)). They are updated in every iteration until the differences in values are smaller than a defined epsilon. 
    ```python
    while true:
      V_old = V_new
      
      ... 
                                                                       
      iteration += 1
      delta[iteration] = 0
                       
      V_new = np.zeros(env.num_states)
  
      # Iterate through all states.
      for state in range(env.num_states):
          action_values = np.zeros(env.NUM_ACTIONS)
  
          # Iterate through all actions of state.     
          for action in range(env.NUM_ACTIONS):
              # Apply Bellman equation to calculate v.
              action_values[action] = calculate_v_values(V_old, action, state)
        
          # Pick the best action in this state (minimal costs).
          best_action_value = min(action_values)        
  
          # Get biggest difference between best action value and our old value function.
          delta[iteration] = max(delta[iteration], abs(best_action_value - V_old[state]))
  
          # Apply Bellman optimality principle.
          V_new[state] = best_action_value
  
          # Update the policy.
          best_action = np.argmin(action_values)
          policy[state] = np.eye(env.NUM_ACTIONS)[best_action]
  
      # Check for convergence.
      if delta[iteration] < epsilon:
          break
    ```

    To save the values of all iterations (for statistical reason), the method `write_to_csv()` can be used to create data files. 
  
## Learning Process

Now that we understand the implemented files, we can start running our value iteration algorithm to find our optimal policy and values. The process begins with an initialization and proceeds updating in a while loop until no significant changes in the policy are made.   

### Initializing

We need to parse the environment and state-definitions from the `map`-file with the `load_map()` method, before initializing the policy and values for every state. 

Our two-dimensional gridworld can be represented in the console using the `print_map()` method of the `environment.py`. It shows all states and their defined role. 
![Initialized Map](https://hendrikpfaff.de/img/valueiteration/console_map.png)

The policy, value function and delta need to be initialized with 0, before starting the value iteration.
![Initialized Policy](https://hendrikpfaff.de/img/valueiteration/policy_init.png)
![Initialized Value function](https://hendrikpfaff.de/img/valueiteration/value_func_init.png)

### Delta convergence

Starting at the terminal state, every state value gets calculated until a convergence is reached.
![Iteration 1](https://hendrikpfaff.de/img/valueiteration/lernverlauf24_01.png)

![Iteration 6](https://hendrikpfaff.de/img/valueiteration/lernverlauf24_06.png)

![Iteration 12](https://hendrikpfaff.de/img/valueiteration/lernverlauf24_finished.png)




### Problems regarding standard parameters
In the search for _pi*_, certain problems can arise with the standard parameter settings. The car can get stuck in its starting position/state, if it is too far away away from the goal. The car "interprets" this distance as too risky/costly compared to simply driving into the nearest wall.

Another issue is the reward gained by driving into the gas station. While the car can't indefinitely stay in the station and pump gas / gain the reward, it will tries to circle around and enter it again and again.   
![Marked problems in policy and value func.](https://hendrikpfaff.de/img/valueiteration/result_std_marked.png)

Three possible modifications to the environment can be made to solve these issues:
* **Modification of costs in _s0_**: Further reduction of the costs (_<= -24_ in this case) when reaching the terminal state, results in a bigger incentive for steering towards it.
 
* **Modification of probability function**: Increasing the probability of actually executing the intended action (up to near determinism, _p ~ 0.98_), makes it safer for the car to reach its goal. 

* **Modification of collision costs**: By increasing the wall collision costs (_>= 3_), the car will be more likely to get out of corners.    

## References
This project was given and evaluated by [Prof. Dr. Thomas Gabel](https://www.tgabel.de/) at [Frankfurt University of Applied Sciences](https://www.frankfurt-university.de/)