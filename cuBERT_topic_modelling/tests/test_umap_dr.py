import numpy as np
from sklearn.datasets import fetch_20newsgroups
import pytest
from cuBERTopic import gpu_BERTopic

newsgroup_docs = fetch_20newsgroups(
    subset="all", remove=("headers", "footers", "quotes")
)["data"][:1000]

@pytest.mark.parametrize("embeddings,shape", [(np.random.rand(100, 68), 100),
                                              (np.random.rand(1000, 5), 1000)])
def test_umap_reduce_dimensionality(embeddings, shape):
    """ Test UMAP
    Testing whether the dimensionality across different shapes is
    reduced to the correct shape. For now, testing the shape is sufficient
    as the main goal here is to reduce the dimensionality, the quality is
    tested in the full pipeline.
    """
    gpu_topic = gpu_BERTopic()
    umap_embeddings = gpu_topic.reduce_dimensionality(embeddings)
    assert umap_embeddings.shape == (shape, 5)
    