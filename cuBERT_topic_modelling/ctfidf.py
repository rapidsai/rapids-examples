from cuml.feature_extraction._tfidf import TfidfTransformer
import cupyx.scipy.sparse
import cupy as cp
from cuml.common.sparsefuncs import csr_row_normalize_l1


# Reference: https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_ctfidf.py
class ClassTFIDF(TfidfTransformer):
    """
    A Class-based TF-IDF procedure using cuml's
    TfidfTransformer as a base. BERTopic highlights a
    very useful approach, a variant of TF-IDF known as c-TF-IDF
    or class based TF-IDF. Applying  TF-IDF on a set of documents,
    we get the relative importance of words between documents but if
    we group all documents with the same cluster ID, we get the scores
    for words within a cluster (topic). The words with the highest scores
    will represent the theme of that cluster.
    """

    def __init__(self, *args, **kwargs):
        super(ClassTFIDF, self).__init__(*args, **kwargs)

    def fit(self,
            X: cupyx.scipy.sparse.csr_matrix,
            n_samples: int,
            multiplier: cp.ndarray = None):
        """Learn the idf vector (global term weights).
        Arguments:
            X: A matrix of term/token counts.
            n_samples: Number of total documents
        """
        if not cupyx.scipy.sparse.issparse(X):
            X = cupyx.scipy.sparse.csr_matrix(X)
        dtype = cp.float64

        if self.use_idf:
            _, n_features = X.shape
            df = cp.squeeze(cp.asarray(X.sum(axis=0)))
            avg_nr_samples = int(X.sum(axis=1).mean())
            idf = cp.log(avg_nr_samples / df)
            if multiplier is not None:
                idf = idf * multiplier
            self._idf_diag = cupyx.scipy.sparse.diags(
                idf,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=dtype,
            )

        return self

    def transform(self, X: cupyx.scipy.sparse.csr_matrix, copy=True):
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
