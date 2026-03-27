# Matrix Multiplication
# In this problem, you will implement two versions of matrix multiplication: a naive version using three nested loops, and a vectorized version using NumPy's built-in operations.
# Time complexity: O(M*N*K) for the naive version, where M, N, and K are the dimensions of the input matrices. The vectorized version has a time complexity of O(M*N*K) as well, but it is optimized and typically much faster due to underlying C/Fortran implementations.
# Space complexity: O(M*N) for the output matrix C in both versions. The intermediate space complexity is O(1) for the naive version, while the vectorized version may use additional space for temporary arrays during computation, but it is generally optimized to minimize this overhead.

import numpy as np

def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes matrix product C = AB using 3 nested loops.
    """
    # Get dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"
    
    # Initialize result matrix with zeros
    C = np.zeros((M, N))
    
    # Your code here
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

def matmul_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes matrix product C = AB using vectorized operations.
    """
    # Your code here
    C = A @ B
    return C
    pass

if __name__ == "__main__":
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    C_naive = matmul_naive(A, B)
    C_vectorized = matmul_vectorized(A, B)

    print("Naive Matrix Multiplication Result:")
    print(C_naive)

    print("\nVectorized Matrix Multiplication Result:")
    print(C_vectorized)
