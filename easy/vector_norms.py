# Given a batch of vectors, compute the L1 and L2 norms for each vector.
# Time complexity: O(N*D) where N is the number of vectors and D is the dimensionality of each vector, due to the need to iterate through all elements to compute the norms.
# Space complexity: O(N) for the output arrays containing the norms, as we need to store the L1 and L2 norms for each of the N vectors. The intermediate space complexity is O(1) for the computations, as we can compute the norms in-place without needing additional space proportional to the input size.     
# The algorithm used is straightforward: for L1 norm, we sum the absolute values of the elements in each vector; for L2 norm, we compute the square root of the sum of squares of the elements in each vector.

import numpy as np
from typing import Dict

def compute_norms(x: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes L1 and L2 norms for a batch of vectors.
    
    Args:
        x: Input matrix of shape (N, D)
        
    Returns:
        Dictionary with keys "l1" and "l2", each containing an array of shape (N,)
    """
   
    l1 = np.sum(np.abs(x), axis=1)

    l2 = np.sqrt(np.sum(x**2, axis=1))

    return {"l1": l1, "l2": l2}

    pass

# Example usage
if __name__ == "__main__":  
    x = np.array([[1, -2, 3], [-4, 5, -6]])
    norms = compute_norms(x)
    print("L1 Norms:", norms["l1"])
    print("L2 Norms:", norms["l2"])
