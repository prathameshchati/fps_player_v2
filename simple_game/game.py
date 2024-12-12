import pygame
import numpy as np
import os
import json
from datetime import datetime
import time
import signal
import sys


"""
- Initialize a small game window for memory efficient capture.

"""

class DotGame:
    def __init__(self, width=200, height=200):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Dot Collection Game")
        
        self.dot_radius = 10
        self.movement_speed = 5
        
        self.blue_dot_pos = np.array([width/2, height/2])
        self.red_dot_pos = self.get_random_position()
        
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.frame_interval = 1.0 / self.fps
        self.last_frame_time = time.time()
        
        self.recording = []
        self.frame_count = 0
        # self.record_interval = 1.0 / 30  # 30 fps
        # self.last_record_time = time.time()
        
        self.base_dir = "game_data"
        self.ensure_directories()
        
        # small surface for recording
        # self.record_width = width // 4
        # self.record_height = height // 4
        # self.record_surface = pygame.Surface((self.record_width, self.record_height))
        
    def get_random_position(self):
        return np.array([
            np.random.randint(self.dot_radius, self.width - self.dot_radius),
            np.random.randint(self.dot_radius, self.height - self.dot_radius)
        ])
    
    def ensure_directories(self):
        directories = [
            self.base_dir,
            os.path.join(self.base_dir, "recordings")
        ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def reset_game(self):
        self.blue_dot_pos = np.array([self.width/2, self.height/2])
        self.red_dot_pos = self.get_random_position()
    
    def save_recording(self):
        if not self.recording:
            return
                
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.base_dir, "recordings", f"game_recording_{timestamp}.json")
        
        processed_recording = []
        for frame in self.recording:
            if 'game_end' in frame:
                processed_recording.append({'game_end': frame['game_end']})
            else:
                processed_frame = {
                    'frame': frame['frame'],
                    'timestamp': frame['timestamp'],
                    'screen_state': frame['screen_state'],
                    'inputs': frame['inputs']
                }
                processed_recording.append(processed_frame)
        
        with open(filename, 'w') as f:
            json.dump(processed_recording, f)
        
        self.recording = []    

    def capture_frame(self):
        screen_state = np.zeros((self.height, self.width), dtype=np.uint8)
        
        blue_x, blue_y = self.blue_dot_pos.astype(int)
        for x in range(max(0, blue_x - self.dot_radius), min(self.width, blue_x + self.dot_radius + 1)):
            for y in range(max(0, blue_y - self.dot_radius), min(self.height, blue_y + self.dot_radius + 1)):
                if (x - blue_x)**2 + (y - blue_y)**2 <= self.dot_radius**2:
                    screen_state[y, x] = 1
        
        red_x, red_y = self.red_dot_pos.astype(int)
        for x in range(max(0, red_x - self.dot_radius), min(self.width, red_x + self.dot_radius + 1)):
            for y in range(max(0, red_y - self.dot_radius), min(self.height, red_y + self.dot_radius + 1)):
                if (x - red_x)**2 + (y - red_y)**2 <= self.dot_radius**2:
                    screen_state[y, x] = 2
        
        keys = pygame.key.get_pressed()
        inputs = {
            'up': bool(keys[pygame.K_UP]),
            'down': bool(keys[pygame.K_DOWN]),
            'left': bool(keys[pygame.K_LEFT]),
            'right': bool(keys[pygame.K_RIGHT])
        }
        
        frame_data = {
            'frame': self.frame_count,
            'timestamp': time.time(),
            'screen_state': screen_state.tolist(),
            'inputs': inputs
        }
        self.recording.append(frame_data)
        self.frame_count += 1

    def run(self):
        
        def signal_handler(sig, frame):
            print("\nSaving game session and exiting...")
            self.save_recording()
            pygame.quit()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        running = True
        game_number = 1
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_UP]:
                self.blue_dot_pos[1] -= self.movement_speed
            if keys[pygame.K_DOWN]:
                self.blue_dot_pos[1] += self.movement_speed
            if keys[pygame.K_LEFT]:
                self.blue_dot_pos[0] -= self.movement_speed
            if keys[pygame.K_RIGHT]:
                self.blue_dot_pos[0] += self.movement_speed
                
            self.blue_dot_pos = np.clip(
                self.blue_dot_pos,
                [self.dot_radius, self.dot_radius],
                [self.width - self.dot_radius, self.height - self.dot_radius]
            )
            
            distance = np.linalg.norm(self.blue_dot_pos - self.red_dot_pos)
            if distance < 2 * self.dot_radius:
                self.recording.append({
                    'game_end': game_number
                })
                game_number += 1
                self.reset_game()
            
            self.screen.fill((255, 255, 255))
            pygame.draw.circle(self.screen, (0, 0, 255), self.blue_dot_pos.astype(int), self.dot_radius)
            pygame.draw.circle(self.screen, (255, 0, 0), self.red_dot_pos.astype(int), self.dot_radius)
            
            self.capture_frame()
                
            pygame.display.flip()
            self.clock.tick(self.fps)
            
        if self.recording:
            self.recording.append({
                'game_end': game_number
            })
            self.save_recording()
        pygame.quit()


if __name__ == "__main__":
    game = DotGame()
    game.run()