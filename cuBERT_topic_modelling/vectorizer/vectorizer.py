from cuml.feature_extraction.text import CountVectorizer
import cudf
from cuml.common.sparsefuncs import create_csr_matrix_from_count_df
import cupy as cp


class CountVecWrapper(CountVectorizer):
    def preprocess_text_gpu(self, doc):
        """
        Chain together an optional series of text preprocessing steps to
        apply to a document.
        Parameters
        ----------
        doc: cudf.Series[str]
            The string to preprocess

        Returns
        -------
        doc: cudf.Series[str]
            preprocessed string
        """
        doc = doc.str.lower()
        doc = doc.str.replace("\n", " ", regex=False)
        doc = doc.str.replace("\t", " ", regex=False)
        doc = doc.str.filter_characters(
            {"a": "z", "0": "9", " ": " ", "A": "Z"}, True, ""
        )
        doc[doc == ""] = "emptydoc"

        # TODO: check if its required
        # sklearn by default removes tokens of
        # length 1, if its remove alphanumerics
        # if remove_single_token_len:
        doc = doc.str.filter_tokens(2)

        return doc

    def fit_transform(self, docs_df):

        self._warn_for_unused_params()
        self._validate_params()
        topic_series = docs_df["Topic"]
        topic_df = topic_series.to_frame(name="Topic_ID")
        topic_df["doc_id"] = cp.arange(len(topic_df))

        docs = self.preprocess_text_gpu(docs_df["Document"])
        n_doc = len(topic_df["Topic_ID"].unique())

        tokenized_df = self._create_tokenized_df(docs)
        self.vocabulary_ = tokenized_df["token"].unique()

        merged_count_df = (
            cudf.merge(tokenized_df, topic_df, how="left")
            .sort_values("Topic_ID")
            .rename({"Topic_ID": "doc_id"}, axis=1)
        )

        count_df = self._count_vocab(merged_count_df)

        # TODO: handle empty docids case later
        empty_doc_ids = cp.empty(shape=0, dtype=cp.int32)
        X = create_csr_matrix_from_count_df(
            count_df,
            empty_doc_ids,
            n_doc,
            len(self.vocabulary_),
            dtype=self.dtype
        )
        if self.binary:
            X.data.fill(1)

        return X