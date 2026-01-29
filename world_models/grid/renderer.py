"""
Game Renderer - Convert board states to grayscale images

Provides simple rendering for 2048 and Othello boards that can be
used for training pixel-based world models.
"""

import numpy as np
from typing import Optional


class GameRenderer:
    """
    Render game boards as grayscale images.
    
    Uses simple block rendering with intensity values proportional
    to tile values (for 2048) or fixed values (for Othello).
    """
    
    def __init__(self, img_size: int = 64):
        """
        Args:
            img_size: Output image size (square)
        """
        self.img_size = img_size
    
    def render_2048(self, board: np.ndarray) -> np.ndarray:
        """
        Render a 2048 board as a grayscale image.
        
        Args:
            board: [4, 4] array with tile values (0, 2, 4, 8, ...)
            
        Returns:
            [img_size, img_size] grayscale image in [0, 1]
        """
        size = board.shape[0]
        cell_size = self.img_size // size
        
        # Create image
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        # Map tile values to intensities
        # 0 -> dark (0.1), 2 -> 0.2, 4 -> 0.3, ..., 2048 -> 0.9
        # Using log2 scaling: intensity = 0.1 + 0.07 * log2(tile) for tile > 0
        
        for r in range(size):
            for c in range(size):
                val = board[r, c]
                
                if val == 0:
                    intensity = 0.1  # Empty cell
                else:
                    # log2(2) = 1, log2(4) = 2, ..., log2(2048) = 11
                    log_val = np.log2(val)
                    intensity = min(0.1 + 0.07 * log_val, 0.95)
                
                # Fill the cell
                r0 = r * cell_size
                c0 = c * cell_size
                
                # Add a small border
                border = max(1, cell_size // 8)
                img[r0+border:r0+cell_size-border, c0+border:c0+cell_size-border] = intensity
        
        return img
    
    def render_othello(self, board: np.ndarray, current_player: int = 1) -> np.ndarray:
        """
        Render an Othello board as a grayscale image.
        
        Args:
            board: [8, 8] array with values {-1, 0, 1}
            current_player: 1 (black) or -1 (white)
            
        Returns:
            [img_size, img_size] grayscale image in [0, 1]
        """
        size = board.shape[0]
        cell_size = self.img_size // size
        
        # Create image with green-ish background
        img = np.ones((self.img_size, self.img_size), dtype=np.float32) * 0.3
        
        # Draw grid lines
        for i in range(size + 1):
            pos = i * cell_size
            if pos < self.img_size:
                img[pos:pos+1, :] = 0.2
                img[:, pos:pos+1] = 0.2
        
        # Draw pieces
        for r in range(size):
            for c in range(size):
                val = board[r, c]
                
                if val == 0:
                    continue
                
                # Center of cell
                cy = r * cell_size + cell_size // 2
                cx = c * cell_size + cell_size // 2
                radius = cell_size // 3
                
                # Draw circle (approximate with filled square for simplicity)
                # Black pieces = dark, White pieces = bright
                intensity = 0.1 if val == 1 else 0.9
                
                # Simple circle approximation
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dy*dy + dx*dx <= radius*radius:
                            py, px = cy + dy, cx + dx
                            if 0 <= py < self.img_size and 0 <= px < self.img_size:
                                img[py, px] = intensity
        
        return img
    
    def render(self, board: np.ndarray, game_type: str = '2048', **kwargs) -> np.ndarray:
        """
        Generic render function.
        
        Args:
            board: Game board array
            game_type: '2048' or 'othello'
            **kwargs: Additional arguments passed to specific renderer
            
        Returns:
            Grayscale image in [0, 1]
        """
        if game_type.lower() == '2048':
            return self.render_2048(board)
        elif game_type.lower() == 'othello':
            return self.render_othello(board, **kwargs)
        else:
            raise ValueError(f"Unknown game type: {game_type}")


def create_renderer(img_size: int = 64) -> GameRenderer:
    """Create a game renderer."""
    return GameRenderer(img_size)


if __name__ == '__main__':
    # Quick visual test
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    renderer = GameRenderer(img_size=64)
    
    # Test 2048
    board_2048 = np.array([
        [2, 4, 8, 16],
        [32, 64, 128, 256],
        [512, 1024, 2048, 0],
        [0, 0, 2, 4]
    ], dtype=np.int32)
    
    img_2048 = renderer.render_2048(board_2048)
    
    # Test Othello
    board_othello = np.zeros((8, 8), dtype=np.int32)
    board_othello[3, 3] = -1
    board_othello[3, 4] = 1
    board_othello[4, 3] = 1
    board_othello[4, 4] = -1
    
    img_othello = renderer.render_othello(board_othello)
    
    # Save test images
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_2048, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('2048')
    axes[0].axis('off')
    axes[1].imshow(img_othello, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Othello')
    axes[1].axis('off')
    plt.savefig('/tmp/renderer_test.png')
    print("Saved test images to /tmp/renderer_test.png")
