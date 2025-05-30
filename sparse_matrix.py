# sparse_matrix.py

import numpy as np

class SparseMatrix:
    def __init__(self, rows, cols, data, shape):
        self.rows = rows        # Row indices
        self.cols = cols        # Column indices
        self.data = data        # Non-zero values
        self.shape = shape      # Matrix shape (height, width)

    @classmethod
    def from_image(cls, image):
        """Convert dense image (2D array) to sparse representation."""
        rows, cols = np.nonzero(image)
        data = image[rows, cols]
        return cls(rows, cols, data, image.shape)

    def to_dense(self):
        """Convert sparse matrix back to dense 2D NumPy array."""
        dense = np.zeros(self.shape)
        dense[self.rows, self.cols] = self.data
        return dense

    def apply_threshold(self, threshold=0.1):
        """Zero out values below a threshold (for denoising)."""
        mask = self.data >= threshold
        return SparseMatrix(self.rows[mask], self.cols[mask], self.data[mask], self.shape)

    def multiply(self, other):
        """
        Sparse matrix multiplication with another sparse matrix.
        Only works if self is (m x n) and other is (n x p).
        """
        if self.shape[1] != other.shape[0]:
            raise ValueError("Matrix shapes not aligned for multiplication")

        result_dict = {}
        for i in range(len(self.data)):
            row_a, col_a, val_a = self.rows[i], self.cols[i], self.data[i]
            for j in range(len(other.data)):
                row_b, col_b, val_b = other.rows[j], other.cols[j], other.data[j]
                if col_a == row_b:
                    key = (row_a, col_b)
                    result_dict[key] = result_dict.get(key, 0) + val_a * val_b

        if not result_dict:
            return SparseMatrix(np.array([]), np.array([]), np.array([]), (self.shape[0], other.shape[1]))

        result_rows, result_cols, result_data = zip(*[(r, c, v) for (r, c), v in result_dict.items()])
        return SparseMatrix(np.array(result_rows), np.array(result_cols), np.array(result_data), (self.shape[0], other.shape[1]))
