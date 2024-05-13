import numpy as np
from main import gram_schmidt_orthogonalization, scalar_product, square_norma_vector


def vector_projection(vector: np.array, plane: np.array):
    base = gram_schmidt_orthogonalization(plane).T

    n, m = base.shape

    result = np.array([0 for _ in range(vector.__len__())], dtype="float")


    for i in range(n):
        result += base[i] * scalar_product(base[i], vector) / square_norma_vector(base[i])


    return result



if __name__ == "__main__":
    plane = np.array([[1, 2],
                      [1, 0],
                      [1, -2]])
    vector = np.array([1, -2, 1])
    a = vector_projection(vector, plane)

    print(a)