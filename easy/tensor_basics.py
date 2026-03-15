# This code demonstrates how to use broadcasting in NumPy to perform operations on arrays of different shapes.
# Time complexity: O(n*m) where n is the number of rows in X and m is the number of columns in X.
# Space complexity: O(n*m) for the output array, but the intermediate operations are O(1) due to broadcasting.



import numpy as np

def broadcast_ops_scale_rows(X: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (X + b) * w.reshape(-1, 1)  # w length must equal X.shape[0]

if __name__ == "__main__":
    X = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([1, 2])
    w = np.array([0.5, 1.0, 1.5])  # length 3 to scale rows

    result = broadcast_ops_scale_rows(X, b, w)
    print(result)

