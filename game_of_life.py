import gymnasium as gym
import numpy as np 
import pygame
import sys
import random
import os
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, DQN

# Constants
MAX_GENERATION=250
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 64  # Matches sprite size
ROWS, COLS = (HEIGHT // GRID_SIZE), ((WIDTH - 200) // GRID_SIZE)  # Leave space for UI
FPS = 10

# Colors
GREEN = (34, 139, 34)
BROWN = (139, 69, 19)
WHITE = (255, 255, 255)

class FarmLifeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, rows=ROWS, cols=COLS, max_generation=MAX_GENERATION):
        super(FarmLifeEnv, self).__init__()
        
        # Define action and observation spaces
        # Actions: 0-3 for movement (up, down, left, right), 4 for planting, 5 for removing
        self.action_space = spaces.Discrete(6)
        
        # Observation space: Grid state + agent position
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(rows, cols), dtype=np.int8),
            "agent_position": spaces.Box(low=np.array([0, 0]), high=np.array([cols-1, rows-1]), dtype=np.int8)
        })
        
        self.rows = rows
        self.cols = cols
        self.grid = None
        self.crops_map = {}
        self.agent = None
        self.generation = 0
        self.reset()
        self.max_generation = max_generation
        self.training_mode = False  # Add a flag to indicate if the environment is in training mode
        
        # Setup rendering
        self.screen = None
        self.clock = None
        self.initialized_rendering = False
        self.paused = False
        self.speed = 10
        
        # UI Buttons
        self.buttons = {
            "Pause": pygame.Rect(WIDTH - 170, 50, 150, 40),
            "Restart": pygame.Rect(WIDTH - 170, 100, 150, 40),
            "Exit": pygame.Rect(WIDTH - 170, 150, 150, 40),
            "Speed Up": pygame.Rect(WIDTH - 170, 200, 150, 40),
            "Slow Down": pygame.Rect(WIDTH - 170, 250, 150, 40),
        }
        
    def _get_obs(self):
        return {
            "grid": np.array(self.grid),
            "agent_position": np.array([self.agent.x, self.agent.y])
        }
    
    def _get_info(self):
        # Count number of living cells (1s in the grid)
        living_cells = sum(row.count(1) for row in self.grid)
        return {"living_cells": living_cells, "generation": self.generation}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Initialize grid and agent
        self.grid, self.crops_map = self._random_grid()
        self.agent = Agent(
            random.randint(0, self.cols - 1), 
            random.randint(0, self.rows - 1), 
            self.cols, 
            self.rows
        )
        self.generation = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # Execute one time step within the environment
        
        # Only process actions if not paused
        if not self.paused:
            # Process action
            if action < 4:  # Movement actions
                self.agent.move(action)
            else:  # Plant or remove action
                if action == 4:  # Plant
                    self.grid[self.agent.y][self.agent.x] = 1
                    self.crops_map[(self.agent.x, self.agent.y)] = None  # We'll assign the crop type in render
                elif action == 5:  # Remove
                    self.grid[self.agent.y][self.agent.x] = 0
                    if (self.agent.x, self.agent.y) in self.crops_map:
                        del self.crops_map[(self.agent.x, self.agent.y)]
            
            # Update grid according to Conway's Game of Life rules
            self.grid, self.crops_map = self._next_generation(self.grid, self.crops_map)
            self.generation += 1
        
        # Calculate reward - even when paused, this gives feedback
        reward = self._calculate_reward()
        
        terminated = (self.generation >= self.max_generation) or (self._get_info()["living_cells"] == 0) if self.training_mode else False
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _calculate_reward(self):
        # Reward based on number of living cells
        living_cells = sum(row.count(1) for row in self.grid)
        return living_cells / (self.rows * self.cols)  # Normalize to [0, 1]
    
    def _random_grid(self):
        grid = [[random.choice([0, 1]) for _ in range(self.cols)] for _ in range(self.rows)]
        crops_map = {}
        
        for y in range(self.rows):
            for x in range(self.cols):
                if grid[y][x] == 1:
                    crops_map[(x, y)] = None  # We'll assign the crop type in render
        
        return grid, crops_map
    
    def _next_generation(self, grid, crops_map):
        new_grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        new_crops_map = {}

        for y in range(self.rows):
            for x in range(self.cols):
                neighbors = sum(
                    1 for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                    if (dx or dy) and 0 <= y + dy < self.rows and 0 <= x + dx < self.cols and grid[y + dy][x + dx] == 1
                )

                if grid[y][x] == 1:  # Cell is alive
                    new_grid[y][x] = 1 if 2 <= neighbors <= 3 else 0
                    if new_grid[y][x]:  # If still alive, keep its crop type
                        new_crops_map[(x, y)] = crops_map.get((x, y), None)
                else:  # Cell is dead
                    if neighbors == 3:
                        new_grid[y][x] = 1
                        new_crops_map[(x, y)] = None  # We'll assign a crop type when rendering

        return new_grid, new_crops_map
    
    def handle_events(self):
        """Handle pygame events for UI controls"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return True  # Signal to stop the simulation
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for text, rect in self.buttons.items():
                    if rect.collidepoint(event.pos):
                        if text == "Pause":
                            self.paused = not self.paused
                        elif text == "Restart":
                            self.grid, self.crops_map = self._random_grid()
                            self.agent = Agent(
                                random.randint(0, self.cols - 1),
                                random.randint(0, self.rows - 1),
                                self.cols,
                                self.rows
                            )
                            self.generation = 0
                        elif text == "Exit":
                            self.close()
                            return True  # Signal to stop the simulation
                        elif text == "Speed Up":
                            self.speed = min(30, self.speed + 2)
                        elif text == "Slow Down":
                            self.speed = max(2, self.speed - 2)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused  # Toggle pause with spacebar
        
        return False  # Continue the simulation
    
    def render(self, mode='human', title="Farm Life - Conway's Game of Life (Trained Agent)"):
        if not self.initialized_rendering:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption(title)
            self.clock = pygame.time.Clock()
            self.wheat_img = pygame.image.load("./assets/wheat_resized.png")
            self.flower_img = pygame.image.load("./assets/flower_resized.png")
            self.agent_img = pygame.image.load("./assets/farmer_resized.png")
            self.crop_types = [self.wheat_img, self.flower_img]
            self.initialized_rendering = True
        
        self.screen.fill(GREEN)
        
        # Draw the field
        for y in range(self.rows):
            for x in range(self.cols):
                if self.grid[y][x] == 1:
                    # Assign crop type if not assigned
                    if (x, y) not in self.crops_map or self.crops_map[(x, y)] is None:
                        self.crops_map[(x, y)] = random.choice(self.crop_types)
                    
                    self.screen.blit(self.crops_map[(x, y)], (x * GRID_SIZE, y * GRID_SIZE))
        
        # Draw the agent
        self.screen.blit(self.agent_img, (self.agent.x * GRID_SIZE, self.agent.y * GRID_SIZE))
        
        # Draw UI
        self._draw_ui()
        
        pygame.display.flip()
        self.clock.tick(self.speed)  # Use the adjustable speed
        
        # Handle events
        return self.handle_events()
    
    def _draw_ui(self):
        # Draw UI panel
        pygame.draw.rect(self.screen, BROWN, (WIDTH - 180, 30, 170, 320))
        
        # Draw buttons
        for text, rect in self.buttons.items():
            pygame.draw.rect(self.screen, WHITE, rect, border_radius=5)
            font = pygame.font.Font(None, 26)
            text_surf = font.render(text, True, BROWN)
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)
        
        # Display generation counter
        font = pygame.font.Font(None, 26)
        gen_text = font.render(f"Generation: {self.generation}", True, WHITE)
        gen_rect = gen_text.get_rect(center=(WIDTH - 95, 300))
        self.screen.blit(gen_text, gen_rect)
        
        # Display living cells count
        info = self._get_info()
        cells_text = font.render(f"Living Cells: {info['living_cells']}", True, WHITE)
        cells_rect = cells_text.get_rect(center=(WIDTH - 95, 330))
        self.screen.blit(cells_text, cells_rect)
        
        # Display speed
        speed_text = font.render(f"Speed: {self.speed}", True, WHITE)
        speed_rect = speed_text.get_rect(center=(WIDTH - 95, 360))
        self.screen.blit(speed_text, speed_rect)
        
        # Display paused status
        status = "PAUSED" if self.paused else "RUNNING"
        status_text = font.render(status, True, WHITE)
        status_rect = status_text.get_rect(center=(WIDTH - 95, 390))
        self.screen.blit(status_text, status_rect)
    
    def close(self):
        if self.initialized_rendering:
            pygame.quit()
            self.initialized_rendering = False

class Agent:
    """A class to represent an agent in the grid."""
    def __init__(self, x, y, COLS, ROWS):
        self.x = x
        self.y = y
        self.COLS = COLS
        self.ROWS = ROWS
    
    def move(self, action):
        """Move based on the given action."""
        if action == 0 and self.y > 0:  # Move Up
            self.y -= 1
        elif action == 1 and self.y < self.ROWS - 1:  # Move Down
            self.y += 1
        elif action == 2 and self.x > 0:  # Move Left
            self.x -= 1
        elif action == 3 and self.x < self.COLS - 1:  # Move Right
            self.x += 1

def run_trained_agent(env, algorithm='PPO', model_path="./models/ppo_farm_life"):
    """Run the trained agent indefinitely until user exits"""
    # Load the trained model
    # model = PPO.load(model_path)
    model = globals()[algorithm].load(model_path)
    
    obs, info = env.reset()
    env.training_mode = False  # Set training mode to False when running the trained agent
    
    should_exit = False
    while not should_exit:
        # Only predict actions when not paused
        if not env.paused:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        else:
            # Just render UI and process events while paused, without changing state
            # We don't call step() during pause
            pass
        
        # Render and check if we should exit
        should_exit = env.render(title= f"Farm Life - Conway's Game of Life ({algorithm} Agent)")
        
        # If game is terminated (shouldn't happen with our setup)
        if not env.paused and (terminated or truncated):
            obs, info = env.reset()

# Run a random untrained agent 
def run_random_agent(env):
    """Run a random untrained agent indefinitely until user exits"""
    obs, info = env.reset()
    env.training_mode = False  # Set training mode to False when running the random agent
    
    should_exit = False
    while not should_exit:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render and check if we should exit
        should_exit = env.render(title='Farm Life - Conway\'s Game of Life (Random Agent)')
        
        # If game is terminated (shouldn't happen with our setup)
        if terminated or truncated:
            obs, info = env.reset()

# Function to train the agent
def train_agent(env, algorithm='PPO', total_timesteps=100000, save_path="./models/farm_life"):
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Select the algorithm
    if algorithm == 'PPO':
        model = PPO("MultiInputPolicy", env, verbose=1)
    elif algorithm == 'A2C':
        model = A2C("MultiInputPolicy", env, verbose=1)
    elif algorithm == 'DQN':
        model = DQN("MultiInputPolicy", env, verbose=1)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    env.training_mode = True  # Set training mode to True when training the agent
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save the trained model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model

def evaluate_agent(env, model, num_episodes=100):
    total_rewards = []
    env.training_mode = True
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), np.std(total_rewards)

def evaluate_random_agent(env, num_episodes=100):
    total_rewards = []
    env.training_mode = True 
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), np.std(total_rewards)

# Main function
def main():
    # Create the environment
    env = FarmLifeEnv()
    
    algorithms = ['PPO', 'A2C', 'DQN']
    results = {}
    
    for algo in algorithms:
        model_path = f"./models/{algo.lower()}_farm_life"
        if not os.path.exists(model_path + ".zip"):
            print(f"Training new model with {algo}...")
            model = train_agent(env, algorithm=algo, total_timesteps=10000, save_path=model_path)
        else:
            model = globals()[algo].load(model_path)
        
        print(f"Evaluating {algo} agent...")
        mean_reward, std_reward = evaluate_agent(env, model)
        results[algo] = (mean_reward, std_reward)
    
    print("Evaluating random agent...")
    mean_reward, std_reward = evaluate_random_agent(env)
    results['Random'] = (mean_reward, std_reward)
    
    print("Results:")
    for algo, (mean_reward, std_reward) in results.items():
        print(f"{algo}: Mean Reward = {mean_reward}, Std Reward = {std_reward}")

if __name__ == "__main__":
    main()