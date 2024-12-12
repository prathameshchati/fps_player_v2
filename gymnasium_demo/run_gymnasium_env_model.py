import gymnasium as gym
import gymnasium_env
from gymnasium_env.envs.grid_world import Actions
from gymnasium.wrappers import FlattenObservation
import keyboard
import pygame
from record_game_data import *
from utils import *

size = 5
fps = 30 # for rendering


env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human", size=size)
recorder = GameRecorder(save_path="game_data", size=size)

# change fps as needed
clock = pygame.time.Clock()

observation, info = env.reset()
agent = observation["agent"]
target = observation["target"]

recorder.log_state_and_input(agent, target, -1)

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            running = False
            break

        # use only the next predicted action
        actions = compute_min_steps_to_win(observation)
        action = actions[0]

        observation, reward, terminated, truncated, info = env.step(action)
        agent = observation["agent"]
        target = observation["target"]

        recorder.log_state_and_input(agent, target, action)

        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")

        if terminated or truncated:
            recorder.save_recording("game_recording.json")
            observation, info = env.reset()
            agent = observation["agent"]
            target = observation["target"]
            recorder.log_state_and_input(agent, target, -1)

    env.render()
    clock.tick(fps) # for rendering

env.close()