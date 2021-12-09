from bertopic import BERTopic
from bertopic._utils import (
    check_documents_type,
    check_embeddings_shape,
)
from bertopic.backend._utils import select_backend
import pandas as pd
import numpy as np
import cudf
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from cuBERTopic import gpu_BERTopic
import pytest

@pytest.fixture
def input_data_docs():
    docs = fetch_20newsgroups(subset="all")["data"]
    return docs


class berttopic_wrapper(BERTopic):
    def fit_transform(self, documents, embeddings=None, y=None):

        check_documents_type(documents)
        check_embeddings_shape(embeddings, documents)

        documents = pd.DataFrame(
            {"Document": documents, "ID": range(len(documents)), "Topic": None}
        )

        # Extract embeddings
        if embeddings is None:
            self.embedding_model = select_backend(
                self.embedding_model, language=self.language
            )
            embeddings = self._extract_embeddings(
                documents.Document, method="document", verbose=self.verbose
            )
        else:
            if self.embedding_model is not None:
                self.embedding_model = select_backend(
                    self.embedding_model, language=self.language
                )

        # Reduce dimensionality with UMAP
        if self.seed_topic_list is not None and \
                self.embedding_model is not None:
            y, embeddings = self._guided_topic_modeling(embeddings)
        umap_embeddings = self._reduce_dimensionality(embeddings, y)

        with open("berttopic_umapembeddings.npy", "wb") as f:
            np.save(f, umap_embeddings)

        # Cluster UMAP embeddings with HDBSCAN
        documents, probabilities = self._cluster_embeddings(
            umap_embeddings, documents
        )

        with open("berttopic_clusterobj.npy", "wb") as f:
            np.save(f, self.hdbscan_model)

        documents.to_parquet("berttopic_docs")

        with open("berttopic_probs.npy", "wb") as f:
            np.save(f, probabilities)

        # Sort and Map Topic IDs by their frequency
        if not self.nr_topics:
            documents = self._sort_mappings_by_frequency(documents)

        # Extract topics by calculating c-TF-IDF
        self._extract_topics(documents)

        # Reduce topics
        if self.nr_topics:
            documents = self._reduce_topics(documents)

        self._map_representative_docs(original_topics=True)
        probabilities = self._map_probabilities(probabilities,
                                                original_topics=True)
        predictions = documents.Topic.to_list()

        return predictions, probabilities


class gpubertopic_wrapper(gpu_BERTopic):
    def fit_transform(self, data):
        """Fit the models on a collection of documents, generate topics,
        and return the docs with topics
        Arguments:
            documents: A list of documents to fit on

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The probability of the assigned topic per document.
        """

        umap_embeddings = np.load("berttopic_umapembeddings.npy")

        probabilities = np.load("berttopic_probs.npy")

        documents = pd.read_parquet("berttopic_docs")
        documents = cudf.from_pandas(documents)
        self.update_topic_size(documents)

        del umap_embeddings

        tf_idf, count, docs_per_topics_topics = self.create_topics(
            documents
        )
        top_n_words, name_repr = self.extract_top_n_words_per_topic(
            tf_idf, count, docs_per_topics_topics, n=30
        )

        self.topic_sizes_df["Name"] = self.topic_sizes_df["Topic"].map(
            name_repr
        )
        self.top_n_words = top_n_words
        predictions = documents.Topic.to_arrow().to_pylist()

        return (predictions, probabilities)


def test_fit_transform(input_data_docs):
    model_sbert = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model_sbert.encode(
        input_data_docs,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )
    topic_model = berttopic_wrapper()
    _, probs_cpu = topic_model.fit_transform(input_data_docs, embeddings)

    gpu_topic = gpubertopic_wrapper()
    _, probs_gpu = gpu_topic.fit_transform(input_data_docs)
    
    a = topic_model.get_topic_info().reset_index(drop=True)
    b = gpu_topic.get_topic_info().reset_index(drop=True)

    b_gpu = b.Name.str.split("_", expand=True)[[1, 2, 3, 4]]
    b_gpu["Count"] = b["Count"]
    b_gpu = (
        b_gpu.sort_values(by=["Count", 1, 2, 3, 4], ascending=False)
        .reset_index(drop=True)
        .to_pandas()
    )
    a_cpu = a.Name.str.split("_", expand=True)[[1, 2, 3, 4]]
    a_cpu["Count"] = a["Count"]
    a_cpu = a_cpu.sort_values(
        by=["Count", 1, 2, 3, 4], ascending=False).reset_index(
        drop=True
    )
    assert probs_gpu.all() == probs_cpu.all()
    assert sum(a_cpu["Count"] == b_gpu["Count"]) == len(a_cpu) == len(b_gpu)
    pd.testing.assert_series_equal(
        a_cpu[1][:100], b_gpu[1][:100],
        check_dtype=False
    )
    pd.testing.assert_series_equal(
        a_cpu[2][:100], b_gpu[2][:100],
        check_dtype=False
    )
    pd.testing.assert_series_equal(
        a_cpu[3][:100], b_gpu[3][:100],
        check_dtype=False
    )
    pd.testing.assert_series_equal(
        a_cpu[4][:100], b_gpu[4][:100],
        check_dtype=False
    )
    pd.testing.assert_frame_equal(
        a_cpu[:100], b_gpu[:100],
        check_dtype=False
    )
    