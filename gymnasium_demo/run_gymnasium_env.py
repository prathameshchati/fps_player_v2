import gymnasium as gym
import gymnasium_env
from gymnasium_env.envs.grid_world import Actions
from gymnasium.wrappers import FlattenObservation
import keyboard
import pygame
from record_game_data import *

"""
Top left of grid is (0,0) and bottom right of grid is (4,4). From our observation, we can get the position of the agent and the target, and then encoder everything else as white space. For the sake of our imitation modeling problem, do not grab the exact positions of our target and agent, rather, encode all the pixels directly. 

Decide how to handle terminal state. If whitespace is encoded as 0, target as 1, and player as 2, when the player overlaps the target, we need to handle this accordingly. 

Save reward data as well. 


Write a wrapper to flatten the observations into a 1d vector as opposed to making the recorder do the preprocessing (will save time, hopefully...)
"""

size=5
env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human", size=size) # size=10, default 5
recorder = GameRecorder(save_path="game_data", size=size) # initialize game recorder
observation, info = env.reset()

running = True
print("Controls: W (up), S (down), A (left), D (right), Q (quit)")
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:  
                action = Actions.DOWN.value
            elif event.key == pygame.K_s:  
                action = Actions.UP.value
            elif event.key == pygame.K_a:  
                action = Actions.LEFT.value
            elif event.key == pygame.K_d:  
                action = Actions.RIGHT.value
            elif event.key == pygame.K_q:
                running = False
                break

            observation, reward, terminated, truncated, info = env.step(action)
            agent = observation["agent"]
            target = observation["target"]

            # log data 
            recorder.log_state_and_input(agent, target, action)


            print(f"Observation: {observation}")
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")

            if terminated or truncated:
                recorder.save_recording("game_recording.json")
                observation, info = env.reset()

    env.render()

env.close()
