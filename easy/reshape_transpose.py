# Given a flat array x and dimensions B, C, H, W, reshape x to (B, C, H, W) and then transpose it to (B, H, W, C).
# Time complexity: O(n) where n is the number of elements in x, due to the reshape and transpose operations.
# Space complexity: O(n) for the output array, as both reshape and transpose create new arrays in memory. 
#The algorithm used is reshape followed by transpose.






import numpy as np

def reshape_and_transpose(x: np.ndarray, B: int, C: int, H: int, W: int) -> np.ndarray:
    """
    Reshapes flat x to (B, C, H, W) then transposes to (B, H, W, C).
    
    Parameters:
    - x: input flat array
    - B, C, H, W: target dimensions
    
    Returns:
    - reshaped and transposed array
    """
    # Step 1: reshape
    x_reshaped = x.reshape(B, C, H, W)
    
    # Step 2: transpose
    x_transposed = x_reshaped.transpose(0, 2, 3, 1)  # B,H,W,C
    
    return x_transposed