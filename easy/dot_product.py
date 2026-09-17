import numpy as np
from typing import Dict

def vector_products(a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes dot and cross products for batches of 3D vectors.
    
    Args:
        a: Shape (N, 3)
        b: Shape (N, 3)
        
    Returns:
        Dict with "dot" (N,) and "cross" (N, 3)
    """
    
    dot = np.sum(a * b, axis=1) # dot product along the last axis 

    # cross product for 3d vectors 
    cross_x = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    cross_y = a[:,2] * b[:, 0] - a[:,0] * b[:, 2]
    cross_z = a[:,0] * b[:, 1] - a[:, 1] * b[:, 0]

    # Stack the (N,) components as columns to get shape (N, 3).
    cross = np.stack([cross_x, cross_y, cross_z], axis=1)

    return {"dot": dot,
            "cross": cross}
                                   

## example usage
if __name__ == "__main__":
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8, 9], [10, 11, 12]])
    
    result = vector_products(a, b)
    print("Dot products:", result["dot"])
    print("Cross products:", result["cross"])
