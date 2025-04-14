import matplotlib.pyplot as plt
import numpy as np

# Define tromino colors (ensuring three distinct colors)
COLORS = ['red', 'blue', 'green']

def tile_board(board, size, missing_x, missing_y, start_x, start_y, color_idx):
    """ Recursively tiles a 2^n x 2^n board with a missing square using L-trominoes. """
    if size == 2:
        # Base case: place a single tromino
        tromino_cells = []
        for dx in range(2):
            for dy in range(2):
                if (start_x + dx, start_y + dy) != (missing_x, missing_y):
                    tromino_cells.append((start_x + dx, start_y + dy))
        
        # Color the tromino
        for x, y in tromino_cells:
            board[x, y] = color_idx
        return
    
    # Identify quadrant of missing cell
    half = size // 2
    
    missing_quadrant = (missing_x >= start_x + half, missing_y >= start_y + half)
    
    # Place central tromino at the dividing point
    tromino_coords = [
        (start_x + half - 1, start_y + half - 1),
        (start_x + half, start_y + half - 1),
        (start_x + half - 1, start_y + half)
    ]
    
    if missing_quadrant == (True, True):
        tromino_coords.append((start_x + half, start_y + half))
    elif missing_quadrant == (True, False):
        tromino_coords.append((start_x + half, start_y + half - 1))
    elif missing_quadrant == (False, True):
        tromino_coords.append((start_x + half - 1, start_y + half))
    else:
        tromino_coords.append((start_x + half - 1, start_y + half - 1))
    
    # Assign a color (ensuring no adjacent trominoes share the same color)
    tromino_color = (color_idx + 1) % 3
    for x, y in tromino_coords:
        board[x, y] = tromino_color
    
    # Recur for the four quadrants
    sub_problems = [
        (start_x, start_y, start_x + half - 1, start_y + half - 1),
        (start_x, start_y + half, start_x + half - 1, start_y + half),
        (start_x + half, start_y, start_x + half, start_y + half - 1),
        (start_x + half, start_y + half, start_x + half, start_y + half)
    ]
    
    for i, (sx, sy, mx, my) in enumerate(sub_problems):
        if (mx, my) not in tromino_coords:
            mx, my = missing_x, missing_y
        tile_board(board, half, mx, my, sx, sy, (tromino_color + 1) % 3)

def visualize_board(board):
    """ Display the board using matplotlib. """
    n = board.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    for x in range(n):
        for y in range(n):
            color = COLORS[board[x, y]]
            rect = plt.Rectangle((y, n - x - 1), 1, 1, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

# Define board size (2^n x 2^n)
n = 4  # You can change this value
size = 2 ** n
board = np.full((size, size), -1)  # Initialize board with -1 (empty)

# Define missing tile position
missing_x, missing_y = np.random.randint(0, size, size=2)
board[missing_x, missing_y] = 3  # Mark missing cell with a special value

tile_board(board, size, missing_x, missing_y, 0, 0, 0)
visualize_board(board)
