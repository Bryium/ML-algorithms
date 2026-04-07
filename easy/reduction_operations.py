# This file contains functions to perform reduction operations on tensors.
# Time complexity: O(n) where n is the number of elements along the specified axis, due to the need to iterate through all elements to compute the reductions.
# Space complexity: O(m) where m is the size of the output array after reduction, as we need to store the results of the reductions. The intermediate space complexity is O(1) for the computations, as we can compute the reductions in-place without needing additional space proportional to the input size.
# The algorithm used is straightforward: we utilize NumPy's built-in functions to compute the sum, mean, max, and argmax along the specified axis of the input tensor.  

import numpy as np
from typing import Dict, Union

def tensor_reductions(x: np.ndarray, axis: int) -> Dict[str, Union[np.ndarray, float]]:
    """
    Computes sum, mean, max, argmax along a given axis.
    """
    # Compute reductions
    sum_result = np.sum(x, axis=axis)
    mean_result = np.mean(x, axis=axis)
    max_result = np.max(x, axis=axis)
    argmax_result = np.argmax(x, axis=axis)

    return {
        "sum": sum_result,
        "mean": mean_result,
        "max": max_result,
        "argmax": argmax_result
    }

# Example usage
if __name__ == "__main__":
    x = np.array([[1, 2, 3], [4, 5, 6]])
    reductions = tensor_reductions(x, axis=0)
    print("Reductions along axis 0:")
    print("Sum:", reductions["sum"])
    print("Mean:", reductions["mean"])
    print("Max:", reductions["max"])
    print("Argmax:", reductions["argmax"])