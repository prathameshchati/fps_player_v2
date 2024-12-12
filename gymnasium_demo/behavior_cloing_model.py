import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import *

"""
Note, we are excluding the initial state (action: -1) from our dataset.
"""
class GridDataset(Dataset):
    def __init__(self, data):
        self.samples = []
        for game_data in data.values():
            for entry in game_data:
                if entry["action"] >= 0:  # include only valid actions
                    self.samples.append((entry["grid"], entry["action"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        grid, action = self.samples[idx]
        return torch.tensor(grid, dtype=torch.float32), torch.tensor(action, dtype=torch.long)

class BehaviorCloningModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x):
        return self.fc(x)

def train_bc_model(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for grids, actions in dataloader:
            optimizer.zero_grad()
            outputs = model(grids)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

def evaluate_bc_model(model, env, max_steps=100, print_steps=False):
    observations={}
    observation, info = env.reset()
    agent = observation["agent"]
    target = observation["target"]
    running = True
    steps = 0

    observations[steps]=observation

    if print_steps:
        print("Initial state:", observation)

    true_actions = compute_min_steps_to_win(observation) # ground truth minimum actions

    pred_actions=[]
    while running and steps < max_steps:
        grid = reconstruct_grid(agent, target).flatten()
        grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model(grid_tensor).argmax(dim=1).item()

        pred_actions.append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        agent = observation["agent"]
        target = observation["target"]
        steps += 1

        observations[steps]=observation

        if print_steps:
            print(steps, observation, action)

        if terminated or truncated:
            print(f"Game ended in {steps} steps.")
            break

    # env.close()
    return true_actions, pred_actions, observations

