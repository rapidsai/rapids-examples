from sentence_transformers import SentenceTransformer
import cuml
import cudf
from cuml.neighbors import NearestNeighbors
from cuml.manifold import UMAP
from cuml.metrics import pairwise_distances
import cupy as cp
from torch.utils.dlpack import to_dlpack
from ctfidf import ClassTFIDF
from mmr import mmr
from utils.sparse_matrix_utils import top_n_idx_sparse, top_n_values_sparse
from vectorizer.vectorizer import CountVecWrapper

class gpu_BERTopic:
    def __init__(self):
        self.top_n_words_df = None
        self.topic_sizes_df = None
        self.original_topic_mapping = None
        self.new_topic_mapping = None
        self.final_topic_mapping = None

    def create_embeddings(self, data):
        """Creates the sentence embeddings using SentenceTransformer

        Args:
            data (List[str]): a Python List of strings

        Returns:
            embeddings (cupy.ndarray): corresponding sentence
            embeddings for the strings passed
        """
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(
            data,
            show_progress_bar=False,
            device="cuda:0",
            batch_size=64,
            convert_to_numpy=False,
            convert_to_tensor=True,
        )

        # https://docs.cupy.dev/en/stable/user_guide/interoperability.html
        # we can get tensor object from SentenceTransformer, however further
        # method used from cuML do not work well with it,
        # hence, we obtain the entire tensor stack from the transformer, and
        # then use the interoperability between CuPy/PyTorch to our use.

        dx = to_dlpack(embeddings)
        embeddings = cp.fromDlpack(dx)
        return embeddings

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
    def clustering_hdbscan(self, umap_embeddings, documents):
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
            min_cluster_size=10,
            metric="euclidean",
            cluster_selection_method="eom"
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
            count: object of class CountVecWrapper, which inherits from
            CountVectorizer
            docs_per_topics_topics: A list of unique topic labels
            docs_df: cudf DataFrame created from data
        """
        docs_per_topics_topics = docs_df["Topic"].unique()

        tf_idf, count = self.new_c_tf_idf(docs_df, len(docs_df))
        return tf_idf, count, docs_per_topics_topics, docs_df

    # def top_n_idx_sparse(self, matrix, n):
    #     """Return indices of top n values in each row of a sparse matrix
    #     Retrieved from:
    #         https://stackoverflow.com/questions/49207275/finding-the-top-n-values-in-a-row-of-a-scipy-sparse-matrix
    #     Args:
    #         matrix: The sparse matrix from which to get the
    #         top n indices per row
    #         n: The number of highest values to extract from each row
    #     Returns:
    #         indices: The top n indices per row
    #     """
    #     top_n_idx = []
    #     mat_inptr_np_ar = matrix.indptr.get()
    #     le_np = mat_inptr_np_ar[:-1]
    #     ri_np = mat_inptr_np_ar[1:]

    #     for le, ri in zip(le_np, ri_np):
    #         le = le.item()
    #         ri = ri.item()
    #         n_row_pick = min(n, ri - le)
    #         top_n_idx.append(
    #             matrix.indices[
    #                 le + cp.argpartition(
    #                     matrix.data[le:ri], -n_row_pick
    #                 )[-n_row_pick:]
    #             ]
    #         )
    #     return cp.array(top_n_idx)

    # def top_n_values_sparse(self, matrix, indices):
    #     """Return the top n values for each row in a sparse matrix
    #     Args:
    #         matrix: The sparse matrix from which to get the top n
    #         indices per row
    #         indices: The top n indices per row
    #     Returns:
    #         top_values: The top n scores per row
    #     """
    #     top_values = []
    #     for row, values in enumerate(indices):
    #         scores = cp.array(
    #             [matrix[row, value] if value is not None
    #              else 0 for value in values]
    #         )
    #         top_values.append(scores)
    #     return cp.array(top_values)

    # Topic representation
    def extract_top_n_words_per_topic(
        self, tf_idf, count, docs_per_topics_topics, mmr_flag=False, n=30
    ):
        """Based on tf_idf scores per topic, extract the top n words per topic

        Arguments:
            tf_idf: A c-TF-IDF matrix from which to calculate the top words
            count: object of class CountVecWrapper, which is derived
            from CountVectorizer
            docs_per_topics_topics: A list of unique topic labels
            mmr_flag: Boolean value indicating whether or not we want
            to run MMR
            n: number of words per topic (Default: 30)
        Returns:
            top_n_words: The top n words per topic
            topic_names: Dictionary containing key as Topic ID and value
            as top 4 words from that topic cluster
        """

        words = count.get_feature_names().to_arrow().to_pylist()

        labels = sorted(docs_per_topics_topics.to_arrow().to_pylist())

        indices = top_n_idx_sparse(tf_idf, n)
        scores = top_n_values_sparse(tf_idf, indices)
        sorted_indices = cp.argsort(scores, 1)
        indices = cp.take_along_axis(indices, sorted_indices, axis=1)
        scores = cp.take_along_axis(scores, sorted_indices, axis=1)
        indices = indices.get()

        # Get top 30 words per topic based on c-TF-IDF score
        top_n_words = {
            label: [
                (words[word_index], score)
                if word_index and score > 0
                else ("", 0.00001)
                for word_index, score in zip(
                    indices[index][::-1], scores[index][::-1]
                )
            ]
            for index, label in enumerate(labels)
        }
        if mmr_flag:
            for topic, topic_words in top_n_words.items():
                words = [word[0] for word in topic_words]
                word_embeddings = self.create_embeddings(words)
                topic_embedding = self.create_embeddings(
                    " ".join(words)
                ).reshape(1, -1)
                topic_words = mmr(
                    topic_embedding,
                    word_embeddings,
                    words,
                    top_n=n,
                    diversity=0
                )
                top_n_words[topic] = [
                    (word, value)
                    for word, value in top_n_words[topic]
                    if word in topic_words
                ]

        top_n_words = {
            label: values[:n] for label, values in top_n_words.items()
        }

        topic_names = {
            key: f"{key}_" + "_".join([word[0] for word in values[:4]])
            for key, values in top_n_words.items()
        }
        return top_n_words, topic_names

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

    # Topic reduction
    def reduce_topics(self, num_topics, tf_idf, docs_df):
        """Reduce topics to num_topics
        Arguments:
            tf_idf: c_tf_idf matrix obtained
            docs_df: Dataframe with documents and their corresponding IDs
            and Topics
        Returns:
            docs_df: Updated dataframe with documents and the reduced
            number of Topics
            top_n_words: top n words based on the updated dataFrame
        """
        for i in range(num_topics):
            # Calculate cosine similarity
            similarities = pairwise_distances(tf_idf, metric="cosine")
            cp.fill_diagonal(similarities, 0)

            # Extract label to merge into and from where
            topic_sizes = (
                docs_df.groupby(["Topic"])
                .count()
                .sort_values("Document", ascending=False)
                .reset_index()
            )
            topic_to_merge = topic_sizes.iloc[-1]["Topic"]
            topic_to_merge_into = cp.argmax(
                similarities[topic_to_merge + 1]) - 1

            # Adjust topics
            topic_to_merge_into_series = cudf.Series(topic_to_merge_into)
            docs_df.loc[
                docs_df["Topic"] == topic_to_merge, "Topic"
            ] = topic_to_merge_into_series
            old_topics = docs_df.Topic.unique().sort_values()
            old_topics = old_topics.to_arrow().to_pylist()
            map_topics = {
                old_topic: index - 1 for index,
                old_topic in enumerate(old_topics)
            }
            docs_df["Topic"] = docs_df.Topic.map(map_topics)
            docs_per_topics_topics = docs_df["Topic"].unique()

            # Calculate new topic words
            tf_idf, count = self.new_c_tf_idf(docs_df, len(docs_df))
            top_n_words, name_repr = self.extract_top_n_words_per_topic(
                tf_idf, count, docs_per_topics_topics, n=30
            )

            return docs_df, top_n_words

    def fit_transform(self, data):
        """Fit the models on a collection of documents, generate topics, and return
        the docs with topics
        Arguments:
            documents: A list of documents to fit on

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The probability of the assigned topic per document.
        """

        documents = cudf.DataFrame(
            {"Document": data, "ID": range(len(data)), "Topic": None}
        )

        # Extract embeddings
        embeddings = self.create_embeddings(
            documents.Document.to_arrow().to_pylist()
        )

        # Reduce dimensionality with UMAP
        umap_embeddings = self.reduce_dimensionality(embeddings)
        del embeddings

        # Cluster UMAP embeddings with HDBSCAN
        documents, probabilities = self.clustering_hdbscan(
            umap_embeddings,
            documents
        )

        del umap_embeddings

        documents = self.sort_mappings_by_frequency(documents)

        tf_idf, count, docs_per_topics_topics, docs_df = self.create_topics(
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

    def get_topic(self, topic, num_words=10):
        """Return top n words for a specific topic and their c-TF-IDF scores
        Arguments:
            topic: A specific topic for which you want its representation
            num_words: Number of words we want in the representation
        Returns:
            The top n words for a specific word and its respective
            c-TF-IDF scores
        """
        return self.top_n_words[int(self.final_topic_mapping[topic])][:num_words]

    def get_topic_info(self):
        """Get information about each topic including its id, frequency, and name

        Returns:
            info: The information relating to all topics
        """

        # Note: getting topics in sorted order without using
        # TopicMapper, as in BERTopic
        topic_sizes_df_columns = self.topic_sizes_df.Name.str.split(
            "_", expand=True
        )[[0, 1, 2, 3, 4]]
        self.original_topic_mapping = topic_sizes_df_columns[0]
        self.new_topic_mapping = sorted(
            self.topic_sizes_df["Topic"].to_pandas()
        )

        self.original_topic_mapping = self.original_topic_mapping.to_arrow().to_pylist()
        self.final_topic_mapping = dict(zip(self.new_topic_mapping, 
                                            self.original_topic_mapping))
        topic_sizes_df_columns[0] = self.new_topic_mapping
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

    def sort_mappings_by_frequency(self, documents):
        """Reorder mappings by their frequency.
        For example, if topic 88 was mapped to topic
        5 and topic 5 turns out to be the largest topic,
        then topic 5 will be topic 0. The second largest,
        will be topic 1, etc.
        If there are no mappings since no reduction of topics
        took place, then the topics will simply be ordered
        by their frequency and will get the topic ids based
        on that order.
        This means that -1 will remain the outlier class, and
        that the rest of the topics will be in descending order
        of ids and frequency.
        Arguments:
            documents: Dataframe with documents and their
            corresponding IDs and Topics
        Returns:
            documents: Updated dataframe with documents and the mapped
                       and re-ordered topic ids
        """

        self.update_topic_size(documents)
        # Map topics based on frequency
        df = (
            cudf.DataFrame(
                self.topic_sizes_df.to_pandas().to_dict(),
                columns=["Old_Topic", "Size"]
            )
            .sort_values("Size", ascending=False)
            .to_pandas()
        )
        df = df[df.Old_Topic != -1]
        sorted_topics = {**{-1: -1}, **dict(zip(df.Old_Topic, range(len(df))))}

        # Map documents
        documents.Topic = (
            documents.Topic.map(
                sorted_topics
            ).fillna(documents.Topic).astype(int)
        )

        self.update_topic_size(documents)
        return documents
    