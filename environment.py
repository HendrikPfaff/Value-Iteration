class Environment:

    xCost = 2  # Costs for touching the wall.
    gCost = -1  # Costs / Reward for driving through the gas station.
    eCost = -10  # Costs / Reward for arriving at the goal.
    transCost = 1  # General transition costs.

    def __init__(self):
        print("Load Environment")
        self.loadMap()

    def loadMap(self, path="./map"):
        print("Load Map:", path)
