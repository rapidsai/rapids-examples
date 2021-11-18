import sys
from os.path import dirname, realpath
import numpy as np
from sklearn.datasets import fetch_20newsgroups

filepath = realpath(__file__)
dir_of_file = dirname(filepath)
parent_dir_of_file = dirname(dir_of_file)
parents_parent_dir_of_file = dirname(parent_dir_of_file)
sys.path.append(parents_parent_dir_of_file + "/cuBERT-topic-modelling/")
from cuBERTopic import gpu_bertopic

newsgroup_docs = fetch_20newsgroups(
    subset="all", remove=("headers", "footers", "quotes")
)["data"][:1000]


def test_extract_embeddings():
    """Test SentenceTransformer
    Check whether the embeddings are correctly generated
    for both a single string or a list of strings. This means that
    the correct shape should be outputted. The embeddings by itself
    should not exceed certain values as a sanity check.
    """
    gpu_topic = gpu_bertopic()
    single_embedding = gpu_topic.create_embeddings(["a document"])
    multiple_embeddings = gpu_topic.create_embeddings(["a document", "another document"])
    assert single_embedding.shape[0] == 1
    assert single_embedding.shape[1] == 384
    assert np.min(single_embedding) > -5
    assert np.max(single_embedding) < 5

    assert multiple_embeddings.shape[0] == 2
    assert multiple_embeddings.shape[1] == 384
    assert np.min(multiple_embeddings) > -5
    assert np.max(multiple_embeddings) < 5
