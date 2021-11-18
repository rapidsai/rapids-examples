from bertopic import BERTopic
import pandas as pd
import numpy as np
import sys
from os.path import dirname, realpath
import cudf
from sklearn.datasets import fetch_20newsgroups
import pytest

filepath = realpath(__file__)
dir_of_file = dirname(filepath)
parent_dir_of_file = dirname(dir_of_file)
parents_parent_dir_of_file = dirname(parent_dir_of_file)
sys.path.append(parents_parent_dir_of_file + "/cuBERT-topic-modelling/")
from cuBERTopic import gpu_bertopic


data_trivial = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

docs_df_trivial = pd.DataFrame(data_trivial, columns=["Document"])
docs_df_trivial["Topic"] = [1, 2, 0, 1]
docs_df_trivial = docs_df_trivial.sort_values("Topic")

data_big = fetch_20newsgroups(subset="all")["data"]
docs_df_big = pd.DataFrame(data_big, columns=["Document"])
docs_df_big["Topic"] = np.random.randint(0, 100, len(docs_df_big))
docs_df_big = docs_df_big.sort_values("Topic")


def extract_c_tf_idf_scores(documents: pd.DataFrame):
    cpu_bertopic = BERTopic()
    documents_per_topic = documents.groupby(["Topic"], as_index=False).agg(
        {"Document": " ".join}
    )
    cpu_bertopic.c_tf_idf, words = cpu_bertopic._c_tf_idf(
        documents_per_topic, m=len(documents)
    )
    return cpu_bertopic.c_tf_idf, words


@pytest.mark.parametrize("docs_df", [(docs_df_trivial), (docs_df_big)])
def test_trivia_case(docs_df):
    tfidf_score, w = extract_c_tf_idf_scores(docs_df)
    docs_df_gpu = cudf.from_pandas(docs_df)
    gpu_topic = gpu_bertopic()
    X = gpu_topic.new_c_tf_idf(docs_df_gpu, len(docs_df_gpu))
    np.testing.assert_almost_equal(X[0].toarray().get(), tfidf_score.toarray())
