import numpy as np
from typing import Dict

def elementwise_ops(a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes element-wise add, mul, and safe div.
    
    Args:
        a: First tensor
        b: Second tensor (same shape)
        
    Returns:
        Dictionary with keys "add", "mul", "div"
    """
    epsilon = 1e-8

    add = a + b
    mul = a * b
    div = a / (b + epsilon)

    return {
        "add": add,
        "mul": mul,
        "div": div
    }
   
    
    pass
if __name__ == "__main__":
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    results = elementwise_ops(a, b)
    print("Element-wise Addition:")
    print(results["add"])

    print("\nElement-wise Multiplication:")
    print(results["mul"])

    print("\nElement-wise Division:")
    print(results["div"])

