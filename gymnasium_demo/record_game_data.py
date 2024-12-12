import json
import os
import numpy as np

class GameRecorder:
    def __init__(self, save_path, size):
        self.save_path = save_path
        self.size = size
        self.data = []

    def reconstruct_grid(self, agent, target):
        grid = np.zeros((self.size, self.size), dtype=int) # white (background)
        grid[target[0], target[1]]=1  # red (target)
        grid[agent[0], agent[1]]=2  # blue (agent)

        # if target and agent overlap (game is won), set the position to 3
        if agent[0]==target[0] and agent[1]==target[1]:
            grid[target[0], target[1]]=3 # blue/red

        return grid

    def log_state_and_input(self, agent, target, action):
        grid = self.reconstruct_grid(agent, target)
        step_data = {
            "grid": grid.tolist(),
            "action": action
        }
        self.data.append(step_data)

    def save_recording(self, filename):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        filepath = os.path.join(self.save_path, filename)
        with open(filepath, "w") as f:
            json.dump(self.data, f)
        self.data = []
