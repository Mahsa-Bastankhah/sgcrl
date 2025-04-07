import matplotlib.pyplot as plt
import numpy as np

# Assuming WALLS is already defined in your script
# You can copy the WALLS dictionary above and paste it here if needed
from typing import Optional

import gym
import numpy as np
import scipy

# Only Spiral11x11 is supported, but the other walls can easily be added 
# by adding the start and ending coordinates to the fixed_goal_dict in lp_contrastive.py
WALLS = { 
    'Small':  
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
    'Cross':  
        np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]),
    'Impossible':
        np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 1, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 1],
                  [0, 1, 0, 0, 0, 1, 0, 1, 0]]),
    'FourRooms':  
        np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]]),
    'U': 
        np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 0, 0]]),
    'Spiral11x11': 
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]),
    'Wall11x11':  
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    'Maze11x11':  
        np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                  [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
}

import os
import matplotlib.pyplot as plt
import numpy as np

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def plot_all_mazes(walls_dict, save_folder='maze_visualization'):
    os.makedirs(save_folder, exist_ok=True)

    for name, walls in walls_dict.items():
        H, W = walls.shape

        wall_img = np.ones_like(walls, dtype=float)
        wall_img[walls == 1] = 0.5  # walls are gray

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(wall_img, origin='lower', cmap='gray', extent=[0, W, 0, H])

        # Add a black border around the maze
        border = Rectangle((0, 0), W, H, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(border)

        ax.axis('off')
        ax.set_xlim([0, W])
        ax.set_ylim([0, H])
        ax.set_aspect('equal')

        plt.tight_layout()
        save_path = os.path.join(save_folder, f'{name}.png')
        plt.savefig(save_path, dpi=300)
        print(f"Saved maze with border: {name} -> {save_path}")
        plt.close()


plot_all_mazes(WALLS)
