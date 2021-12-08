from cuml.feature_extraction._tfidf import TfidfTransformer
import cupyx.scipy.sparse as cp_sparse
import cupy as cp
from cuml.common.sparsefuncs import csr_row_normalize_l1


class ClassTFIDF(TfidfTransformer):
    """
    A Class-based TF-IDF procedure using scikit-learns
    TfidfTransformer as a base. C-TF-IDF can best be
    explained as a TF-IDF formula adopted for multiple classes
    by joining all documents per class. Thus, each class
    is converted to a single document instead of set of documents.
    Then, the frequency of words **t** are extracted for
    each class **i** and divided by the total number of
    words **w**. Next, the total, unjoined, number of documents
    across all classes **m** is divided by the total sum of
    word **i** across all classes.
    """

    def __init__(self, *args, **kwargs):
        super(ClassTFIDF, self).__init__(*args, **kwargs)

    def fit(self,
            X: cp_sparse.csr_matrix,
            n_samples: int,
            multiplier: cp.ndarray = None):
        """Learn the idf vector (global term weights).
        Arguments:
            X: A matrix of term/token counts.
            n_samples: Number of total documents
        """
        if not cp_sparse.issparse(X):
            X = cp_sparse.csr_matrix(X)
        dtype = cp.float64

        if self.use_idf:
            _, n_features = X.shape
            df = cp.squeeze(cp.asarray(X.sum(axis=0)))
            avg_nr_samples = int(X.sum(axis=1).mean())
            idf = cp.log(avg_nr_samples / df)
            if multiplier is not None:
                idf = idf * multiplier
            self._idf_diag = cp_sparse.diags(
                idf,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=dtype,
            )

        return self

    def transform(self, X: cp_sparse.csr_matrix, copy=True):
        """Transform a count-based matrix to c-TF-IDF
        Arguments:
            X (sparse matrix): A matrix of term/token counts.
        Returns:
            X (sparse matrix): A c-TF-IDF matrix
        """
        if self.use_idf:
            X = csr_row_normalize_l1(X, inplace=False)
            X = X * self._idf_diag

        return X
