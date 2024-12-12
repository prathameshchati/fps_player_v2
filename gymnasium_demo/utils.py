import os
import numpy as np
import random
from tqdm import tqdm

# define actions
ACTIONS= {
            3: np.array([1, 0]), # right
            1: np.array([0, -1]), # up
            2: np.array([-1, 0]), # left
            0: np.array([0, 1]), # down
        }

# provide initial agent and target positions from env.reset() observations -> should be more efficient than calculate_minimum_path

# potentially void terminal state as well

# shuffle: changes order of actions but still maintains optimal -> downstream we can add noise to the path to generate less optimal solutions.
def compute_min_steps_to_win(observation, shuffle=False, print_values=False):
    agent=np.array(observation["agent"])
    target=np.array(observation["target"])
    diff=target-agent

    # define up/down and left/right
    if diff[0]>0:
        x_action=3 # right
    else: 
        x_action=2 # left

    if diff[1]>0:
        y_action=0 # down
    else:
        y_action=1 # up

    xy_actions=[x_action, y_action]
    actions=[a for a, n in zip(xy_actions, abs(diff)) for _ in range(n)] # abs val of diff to get proper multiples

    if print_values:
        print("Agent:", agent, "Target:", target)
        print("Difference:", diff)
        print("XY Actions:", xy_actions)

    if shuffle:
        random.shuffle(actions)

    return actions

# each position represents the state AFTER the action has been taken to provide a proper mapping

# includes initial position as well
def generate_agent_target_positions(observation, actions):
    positions={}
    agent=np.array(observation["agent"])
    target=np.array(observation["target"])

    # initialize position at time step 0 (keys: time step; values: {agent, target, actions (-1 for init)}); target pos doesn't change, used for reconstructing grid
    ts=0
    positions[ts]={"agent":np.array(agent), "target":np.array(target), "action":-1}

    for a in actions:
        ts+=1
        agent+=ACTIONS[a]
        positions[ts]={"agent":np.array(agent), "target":np.array(target), "action":a}

    return positions


def reconstruct_grid(agent, target, observation=None, size=5):
    if observation:
        agent=np.array(observation["agent"])
        target=np.array(observation["target"])

    grid = np.zeros((size, size), dtype=int) # white (background)
    grid[target[1], target[0]]=1  # red (target)
    grid[agent[1], agent[0]]=2  # blue (agent)

    # if target and agent overlap (game is won), set the position to 3
    if agent[0]==target[0] and agent[1]==target[1]:
        grid[target[1], target[0]]=3 # blue/red

    # transpose to get proper 2d ordering - nvm
    return grid.flatten()

def generate_grid_action_dict(positions, size=5):
    grid_data=[]
    for ts, pos in positions.items():
        agent=pos["agent"]
        target=pos["target"]
        grid=reconstruct_grid(agent, target, observation=None, size=size)
        grid_data.append({"grid":list(grid), "action":pos["action"]})
    return grid_data


def generate_training_data(env, N, shuffle=False, print_values=False, size=5):
    data_agg={}
    for i in tqdm(range(N)):
        observation, info = env.reset()
        actions=compute_min_steps_to_win(observation, shuffle=shuffle, print_values=print_values)
        positions=generate_agent_target_positions(observation, actions)
        grid_data=generate_grid_action_dict(positions, size=size)
        data_agg[i]=grid_data
    return data_agg


