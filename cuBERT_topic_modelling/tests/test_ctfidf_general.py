import sys
from os.path import dirname, realpath
import cudf
import numpy as np
from cupyx.scipy.sparse.csr import csr_matrix
from sklearn.datasets import fetch_20newsgroups

filepath = realpath(__file__)
dir_of_file = dirname(filepath)
parent_dir_of_file = dirname(dir_of_file)
parents_parent_dir_of_file = dirname(parent_dir_of_file)
sys.path.append(parents_parent_dir_of_file + "/cuBERT-topic-modelling/")
from ctfidf import ClassTFIDF
from vectorizer import CountVecWrapper


newsgroup_docs = fetch_20newsgroups(
    subset="all", remove=("headers", "footers", "quotes")
)["data"][:1000]


def test_ctfidf():
    """Test c-TF-IDF
    Test whether the c-TF-IDF matrix is correctly calculated.
    This includes the general shape of the matrix as well as the
    possible values that could occupy the matrix.
    """
    nr_topics = 10
    docs_df = cudf.DataFrame(newsgroup_docs, columns=["Document"])
    docs_df["Topic"] = np.random.randint(-1, nr_topics, len(newsgroup_docs))

    count = CountVecWrapper(ngram_range=(1, 1))
    X = count.fit_transform(docs_df)
    words = count.get_feature_names()
    multiplier = None

    transformer = ClassTFIDF().fit(
        X, n_samples=len(newsgroup_docs), multiplier=multiplier
    )

    c_tf_idf = transformer.transform(X)

    words = words.to_arrow().to_pylist()
    assert len(words) > 1000
    assert all([isinstance(x, str) for x in words])

    assert isinstance(X, csr_matrix)
    assert isinstance(c_tf_idf, csr_matrix)

    assert X.shape[0] == nr_topics + 1
    assert X.shape[1] == len(words)

    assert c_tf_idf.shape[0] == nr_topics + 1
    assert c_tf_idf.shape[1] == len(words)

    assert np.min(c_tf_idf) > -1
    assert np.max(c_tf_idf) < 1

    assert np.min(X) == 0
