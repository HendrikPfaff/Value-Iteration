#!/usr/bin/python3
import environment
import agent
import vi

if __name__ == "__main__":
    car = agent.Agent()
    env = environment.Environment()
    vi = vi.ValueIteration(env, car)
    vi.run()

