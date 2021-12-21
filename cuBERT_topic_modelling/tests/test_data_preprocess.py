import pandas as pd
import cudf
import re
from vectorizer.vectorizer import CountVecWrapper
import pytest

@pytest.fixture
def input_data_docs_df():
    data_trivial = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
        ""
    ]

    docs_df = pd.DataFrame(data_trivial, columns=["Document"])
    return docs_df


def preprocess_text_bertopic(documents):
    documents = documents.to_arrow().to_pylist()
    cleaned_documents = [doc.lower() for doc in documents]
    cleaned_documents = [doc.replace("\n", " ") for doc in cleaned_documents]
    cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]
    cleaned_documents = [
        re.sub(r"[^A-Za-z0-9 ]+", "", doc) for doc in cleaned_documents
    ]
    cleaned_documents = [
        doc if doc != "" else "emptydoc" for doc in cleaned_documents
    ]
    return cudf.Series(cleaned_documents, name="Document")


def test_trivia_case(input_data_docs_df):
    docs_df_gpu = cudf.from_pandas(input_data_docs_df)
    clean_docs_bertopic = preprocess_text_bertopic(docs_df_gpu["Document"])
    cv = CountVecWrapper()
    clean_docs_gpu = cv.preprocess_text_gpu(docs_df_gpu["Document"])

    cudf.testing.testing.assert_series_equal(clean_docs_gpu,
                                             clean_docs_bertopic)
