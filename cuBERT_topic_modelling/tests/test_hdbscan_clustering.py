import cudf
from sklearn.datasets import fetch_20newsgroups, make_blobs
import pytest
from cuBERTopic import gpu_BERTopic


newsgroup_docs = fetch_20newsgroups(
    subset="all", remove=("headers", "footers", "quotes")
)["data"][:1000]


@pytest.mark.parametrize(
    "samples,features,centers",
    [
        (200, 500, 1),
        (500, 200, 1),
        (200, 500, 2),
        (500, 200, 2),
        (200, 500, 4),
        (500, 200, 4),
    ],
)
def test_hdbscan_cluster_embeddings(samples, features, centers):
    """Test HDBSCAN
    Testing whether the clusters are correctly created and if the old
    and new dataframes are the exact same aside from the Topic column.
    """
    embeddings, _ = make_blobs(
        n_samples=samples,
        centers=centers,
        n_features=features,
        random_state=42
    )
    documents = [str(i + 1) for i in range(embeddings.shape[0])]
    old_df = cudf.DataFrame(
        {"Document": documents, "ID": range(len(documents)), "Topic": None}
    )
    gpu_topic = gpu_BERTopic()
    new_df, _ = gpu_topic.clustering_hdbscan(embeddings, old_df)

    assert len(new_df.Topic.unique()) == centers
    assert "Topic" in new_df.columns
