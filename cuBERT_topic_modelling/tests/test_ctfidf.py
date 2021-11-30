from bertopic import BERTopic
import pandas as pd
import numpy as np
import cudf
from sklearn.datasets import fetch_20newsgroups
import pytest
from cupyx.scipy.sparse.csr import csr_matrix
from cuBERTopic import gpu_BERTopic
from ctfidf import ClassTFIDF
from vectorizer.vectorizer import CountVecWrapper

@pytest.fixture
def input_data_trivial():
    data_trivial = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]

    docs_df_trivial = pd.DataFrame(data_trivial, columns=["Document"])
    docs_df_trivial["Topic"] = [1, 2, 0, 1]
    docs_df_trivial = docs_df_trivial.sort_values("Topic")
    return docs_df_trivial

@pytest.fixture
def input_data_big():
    data_big = fetch_20newsgroups(subset="all")["data"]
    docs_df_big = pd.DataFrame(data_big, columns=["Document"])
    docs_df_big["Topic"] = np.random.randint(0, 100, len(docs_df_big))
    docs_df_big = docs_df_big.sort_values("Topic")
    return docs_df_big

@pytest.fixture
def input_newsgroup_dataset():
    newsgroup_docs = fetch_20newsgroups(subset="all",
                                        remove=("headers", "footers", "quotes")
                                        )["data"][:1000]
    return newsgroup_docs

def extract_c_tf_idf_scores(documents: pd.DataFrame):
    cpu_bertopic = BERTopic()
    documents_per_topic = documents.groupby(["Topic"], as_index=False).agg(
        {"Document": " ".join}
    )
    cpu_bertopic.c_tf_idf, words = cpu_bertopic._c_tf_idf(
        documents_per_topic, m=len(documents)
    )
    return cpu_bertopic.c_tf_idf, words


@pytest.mark.parametrize("docs_df",
                         [pytest.lazy_fixture("input_data_trivial"),
                          pytest.lazy_fixture("input_data_big")])
def test_ctfidf_values(docs_df):
    """Test c-TF-IDF values
    Here we test the values against the _c_tf_idf method from BERTopic
    to make sure we get the same correctness.
    """
    tfidf_score, w = extract_c_tf_idf_scores(docs_df)
    docs_df_gpu = cudf.from_pandas(docs_df)
    gpu_topic = gpu_BERTopic()
    X = gpu_topic.new_c_tf_idf(docs_df_gpu, len(docs_df_gpu))
    np.testing.assert_almost_equal(X[0].toarray().get(), tfidf_score.toarray())


def test_ctfidf_general(input_newsgroup_dataset):
    """Test c-TF-IDF general
    Test whether the c-TF-IDF matrix is correctly calculated.
    This includes the general shape of the matrix as well as the
    possible values that could occupy the matrix.
    """
    nr_topics = 10
    docs_df = cudf.DataFrame(input_newsgroup_dataset, columns=["Document"])
    docs_df["Topic"] = np.random.randint(-1, nr_topics, len(input_newsgroup_dataset))

    count = CountVecWrapper(ngram_range=(1, 1))
    X = count.fit_transform(docs_df)
    words = count.get_feature_names()
    multiplier = None

    transformer = ClassTFIDF().fit(
        X, n_samples=len(input_newsgroup_dataset), multiplier=multiplier
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
    