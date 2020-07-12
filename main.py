#!/usr/bin/python3
import colorama
import environment
import numpy as np


def value_iteration(env, epsilon=0.00001, discount_factor=1.0):
    def calculate_v_values(V, action, state):
        [(probability, next_state, cost, done)] = env.P[state][action]
        return probability * (cost + discount_factor * V[next_state])

    def printV(V, env):
        n = 0
        for i in range(5):
            for j in range(10):
                if env.is_wall(n):
                    color = colorama.Fore.RED
                elif env.is_gas_station(n):
                    color = colorama.Fore.GREEN
                elif env.is_terminal(n):
                    color = colorama.Fore.CYAN
                else:
                    color = colorama.Fore.RESET

                if V[n] == -0:
                    value = 0
                else:
                    value = V[n]

                print(color + "%+.3f" % value, end="")
                print(colorama.Fore.RESET + " | ", end="")
                n += 1
            print(colorama.Fore.RESET)

    V = np.zeros(env.num_states)
    policy = np.zeros([env.num_states, env.NUM_ACTIONS])
    iteration = 0

    while True:
        iteration += 1
        delta = 0

        # Iterate through all states.
        for state in range(env.num_states):
            action_values = np.zeros(env.NUM_ACTIONS)

            # Iterate through all actions of state.
            for action in range(env.NUM_ACTIONS):
                # Apply Bellman equation to calculate v.
                action_values[action] = calculate_v_values(V, action, state)

            # Pick the best action in this state (minimal costs).
            best_action_value = min(action_values)

            # Get biggest difference between best action value and our old value function.
            delta = max(delta, abs(best_action_value - V[state]))

            # Apply Bellman optimality principle.
            V[state] = best_action_value

            # Update the policy.
            best_action = np.argmax(action_values)
            policy[state] = np.eye(env.NUM_ACTIONS)[best_action]

        print("\nIteration:", iteration)
        print("\nGrid policy (0=up, 1=right, 2=down, 3=left):\n", np.reshape(np.argmax(policy, axis=1), env.mapShape))
        #print("\nGrid policy (0=up, 1=right, 2=down, 3=left):")
        print("\nGrid Value Function:")
        printV(V, env)
        print("\nDelta:", delta)
        print("\n====================================================================================")

        # Check for convergence.
        if delta < epsilon:
            break

    return policy, V


if __name__ == "__main__":
    env = environment.Environment("./map")
    policy, v = value_iteration(env)
