import numpy as np


def square_norma_vector(vector):
    return vector.dot(vector)


def scalar_product(v1, v2):
    return v1.dot(v2)


def gram_schmidt_orthogonalization(_base: np.array) -> np.array:
    rows, cols = _base.shape

    assert rows >= cols

    orthogonalization_matrix = np.array([[0 for _ in range(rows)] for _ in range(cols)], dtype="float")

    base = _base.copy().T

    e = base[0].copy()

    orthogonalization_matrix[0] = e

    for i in range(1, cols):
        next_vector = base[i].copy().astype(np.float32)

        for j in range(i):
            next_vector -= 1 / square_norma_vector(orthogonalization_matrix[j]) * scalar_product(
                orthogonalization_matrix[j], base[i]) * orthogonalization_matrix[j]

        orthogonalization_matrix[i] = next_vector

    return orthogonalization_matrix.T


if __name__ == "__main__":
    matrix = gram_schmidt_orthogonalization(np.array([[1, 1],
                                                      [2, 3],
                                                      [1, 3]]))
