import cupy as cp

def top_n_idx_sparse(matrix, n):
        """Return indices of top n values in each row of a sparse matrix
        Retrieved from:
            https://stackoverflow.com/questions/49207275/finding-the-top-n-values-in-a-row-of-a-scipy-sparse-matrix
        Args:
            matrix: The sparse matrix from which to get the
            top n indices per row
            n: The number of highest values to extract from each row
        Returns:
            indices: The top n indices per row
        """
        top_n_idx = []
        mat_inptr_np_ar = matrix.indptr.get()
        le_np = mat_inptr_np_ar[:-1]
        ri_np = mat_inptr_np_ar[1:]

        for le, ri in zip(le_np, ri_np):
            le = le.item()
            ri = ri.item()
            n_row_pick = min(n, ri - le)
            top_n_idx.append(
                matrix.indices[
                    le + cp.argpartition(
                        matrix.data[le:ri], -n_row_pick
                    )[-n_row_pick:]
                ]
            )
        return cp.array(top_n_idx)

def top_n_values_sparse(matrix, indices):
    """Return the top n values for each row in a sparse matrix
    Args:
        matrix: The sparse matrix from which to get the top n
        indices per row
        indices: The top n indices per row
    Returns:
        top_values: The top n scores per row
    """
    top_values = []
    for row, values in enumerate(indices):
        scores = cp.array(
            [matrix[row, value] if value is not None
                else 0 for value in values]
        )
        top_values.append(scores)
    return cp.array(top_values)