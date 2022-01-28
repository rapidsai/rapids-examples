import os
import warnings

if "NVCC" not in os.environ:
    os.environ["NVCC"] = "/usr/local/cuda-11.5/bin/nvcc"
    warnings.warn(
        "NVCC Path not found, set to  : /usr/local/cuda-11.5/bin/nvcc . \nPlease set NVCC as appropitate to your environment"
    )

import torch
import cuml
import cudf
from cuml.neighbors import NearestNeighbors
from cuml.manifold import UMAP
import cupy as cp
from ctfidf import ClassTFIDF
from mmr import mmr
from utils.sparse_matrix_utils import top_n_sparse
from vectorizer.vectorizer import CountVecWrapper
import math
from embedding_extraction import create_embeddings
from transformers import AutoModel

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class gpu_BERTopic:
    def __init__(self, embedding_model=None, vocab_file=None):
        self.top_n_words = None
        self.topic_sizes_df = None
        self.original_topic_mapping = None
        self.new_topic_mapping = None
        self.final_topic_mapping = None
        if embedding_model is None:
            embedding_model = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            ).to(device)

        if vocab_file is None:
            vocab_file = "vocab/voc_hash.txt"

        self.vocab_file = vocab_file
        self.embedding_model = embedding_model

    # Dimensionality reduction
    def reduce_dimensionality(self, embeddings):
        """Reduce dimensionality of embeddings using UMAP and train a UMAP model

        Args:
            embeddings (cupy.ndarray): The extracted embeddings using the
            sentence transformer module.

        Returns:
            umap_embeddings: The reduced embeddings
        """
        m_cos = NearestNeighbors(n_neighbors=15, metric="cosine")
        m_cos.fit(embeddings)
        knn_graph_cos = m_cos.kneighbors_graph(embeddings, mode="distance")
        u1 = UMAP(n_neighbors=15, n_components=5, min_dist=0.0)
        umap_embeddings = u1.fit_transform(embeddings, knn_graph=knn_graph_cos)

        return umap_embeddings

    # Clustering step
    def cluster_embeddings(self, umap_embeddings, documents):
        """Cluster UMAP embeddings with HDBSCAN

        Args:
            umap_embeddings: The reduced sentence embeddings with UMAP
            documents: DataFrame from the original data

        Returns:
            documents: Modified dataframe with topic labels
            probabilities: response from cluster.probabilities_ which
            represents the likelihood of the doc belonging to a cluster.
        """
        cluster = cuml.cluster.HDBSCAN(
            min_cluster_size=10, metric="euclidean", cluster_selection_method="eom"
        ).fit(umap_embeddings)

        documents["Topic"] = cluster.labels_
        probabilities = cluster.probabilities_
        self.update_topic_size(documents)

        return documents, probabilities

    def new_c_tf_idf(self, document_df, m, ngram_range=(1, 1)):
        """Calculate a class-based TF-IDF where m is the number of total documents.

        Arguments:
            document_df (cudf.DataFrame): dataFrame containing our strings
            and docids
            m (int): The total number of documents (unjoined)
            ngram_range (tuple (int, int), default=(1, 1)): The lower and
            upper boundary
            of the range of n-values for different word n-grams or char
            n-grams to be extracted.

        Returns:
            tf_idf: The resulting matrix giving a value (importance score)
            for each word per topic
            count: object of class CountVecWrapper
        """
        count = CountVecWrapper(ngram_range=ngram_range)
        X = count.fit_transform(document_df)
        multiplier = None

        transformer = ClassTFIDF().fit(X, n_samples=m, multiplier=multiplier)

        c_tf_idf = transformer.transform(X)

        return c_tf_idf, count

    def create_topics(self, docs_df):
        """Extract topics from the clusters using a class-based TF-IDF
        Arguments:
            docs_df: DataFrame containing documents and other information
        Returns:
            tf_idf: The resulting matrix giving a value (importance score) for
            each word per topic
            vectorizer: object of class CountVecWrapper, which inherits from
            CountVectorizer
            topic_labels: A list of unique topic labels
        """
        topic_labels = docs_df["Topic"].unique()

        tf_idf, vectorizer = self.new_c_tf_idf(docs_df, len(docs_df))
        return tf_idf, vectorizer, topic_labels

    # Topic representation
    def extract_top_n_words_per_topic(self, tf_idf, vectorizer, labels, n=30):
        """Based on tf_idf scores per topic, extract the top n words per topic

        Arguments:
            tf_idf: A c-TF-IDF matrix from which to calculate the top words
            vectorizer: object of class CountVecWrapper, which is derived from CountVectorizer
            labels: A list of unique topic labels
            n: number of words per topic (Default: 30)
        Returns:
            top_n_words: The top n words per topic
            topic_names: Dictionary containing key as Topic ID and value
            as top 4 words from that topic cluster
        """

        words = vectorizer.get_feature_names().to_arrow().to_pylist()
        labels = sorted(labels.to_arrow().to_pylist())
        indices, scores = top_n_sparse(tf_idf, n)
        sorted_indices = cp.argsort(scores, 1)

        indices = cp.take_along_axis(indices, sorted_indices, axis=1).get()
        scores = cp.take_along_axis(scores, sorted_indices, axis=1).get()

        # Get top 30 words per topic based on c-TF-IDF score
        topics = {
            label: [
                (words[word_index], score)
                if word_index and score > 0
                else ("", 0.00001)
                for word_index, score in zip(indices[index][::-1], scores[index][::-1])
            ]
            for index, label in enumerate(labels)
        }

        topic_names = {
            key: f"{key}_" + "_".join([word[0] for word in values[:4]])
            for key, values in topics.items()
        }
        return topics, topic_names

    def extract_topic_sizes(self, df):
        """Calculate the topic sizes
        Arguments:
            documents: dataframe with documents and their corresponding IDs
            and added Topics

        Returns:
            topic_sizes: DataFrame containing topic cluster sizes
        """
        topic_sizes = (
            df.groupby(["Topic"])
            .Document.count()
            .reset_index()
            .rename({"Topic": "Topic", "Document": "Count"}, axis="columns")
            .sort_values("Count", ascending=False)
        )
        return topic_sizes

    def fit_transform(self, data):
        """Fit the models on a collection of documents, generate topics, and return
        the docs with topics
        Arguments:
            data: A list of documents to fit on

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The probability of the assigned topic per document.
        """

        documents = cudf.DataFrame(
            {"Document": data, "ID": cp.arange(len(data)), "Topic": None}
        )

        # Extract embeddings
        embeddings = create_embeddings(
            documents.Document, self.embedding_model, self.vocab_file
        )

        # Reduce dimensionality with UMAP
        umap_embeddings = self.reduce_dimensionality(embeddings)
        del embeddings

        # Cluster UMAP embeddings with HDBSCAN
        documents, probabilities = self.cluster_embeddings(umap_embeddings, documents)

        del umap_embeddings

        # Topic representation
        tf_idf, count, labels = self.create_topics(documents)
        top_n_words, name_repr = self.extract_top_n_words_per_topic(
            tf_idf, count, labels, n=30
        )

        self.topic_sizes_df["Name"] = self.topic_sizes_df["Topic"].map(name_repr)
        self.top_n_words = top_n_words
        predictions = documents.Topic

        return (predictions, probabilities)

    def get_topic(self, topic, num_words=10):
        """Return top n words for a specific topic and their c-TF-IDF scores
        Arguments:
            topic: A specific topic for which you want its representation
            num_words: Number of words we want in the representation
        Returns:
            The top n words for a specific word and its respective
            c-TF-IDF scores
        """

        return self.top_n_words[int(self.final_topic_mapping[0][topic + 1])][:num_words]

    def get_topic_info(self):
        """Get information about each topic including its id, frequency, and name

        Returns:
            info: The information relating to all topics
        """

        # Note: getting topics in sorted order without using
        # TopicMapper, as in BERTopic
        topic_sizes_df_columns = self.topic_sizes_df.Name.str.split("_", expand=True)[
            [0, 1, 2, 3, 4]
        ]
        self.original_topic_mapping = topic_sizes_df_columns[0]

        self.new_topic_mapping = self.topic_sizes_df["Topic"].sort_values()

        self.original_topic_mapping = self.original_topic_mapping.astype("int64")
        new_mapping_values = self.new_topic_mapping.values
        new_mapping_series = self.new_topic_mapping.reset_index(drop=True)
        original_mapping_series = self.original_topic_mapping.reset_index(drop=True)
        self.final_topic_mapping = cudf.concat(
            [new_mapping_series, original_mapping_series], axis=1
        )

        topic_sizes_df_columns[0] = new_mapping_values
        topic_sizes_df_columns["Name"] = (
            topic_sizes_df_columns[0].astype("str")
            + "_"
            + topic_sizes_df_columns[1]
            + "_"
            + topic_sizes_df_columns[2]
            + "_"
            + topic_sizes_df_columns[3]
            + "_"
            + topic_sizes_df_columns[4]
        )
        self.topic_sizes_df["Name"] = topic_sizes_df_columns["Name"]
        self.topic_sizes_df["Topic"] = topic_sizes_df_columns[0]
        return self.topic_sizes_df

    def update_topic_size(self, documents):
        """Calculate the topic sizes
        Arguments:
            documents: Updated dataframe with documents and
            their corresponding IDs and newly added Topics
        """

        topic_sizes = (
            documents.groupby(["Topic"])
            .Document.count()
            .reset_index()
            .rename({"Topic": "Topic", "Document": "Count"}, axis="columns")
            .sort_values("Count", ascending=False)
        )
        self.topic_sizes_df = topic_sizes
