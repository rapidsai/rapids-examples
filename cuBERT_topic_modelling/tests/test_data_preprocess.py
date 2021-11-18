import pandas as pd
import sys
from os.path import dirname, realpath
import cudf
import re

filepath = realpath(__file__)
dir_of_file = dirname(filepath)
parent_dir_of_file = dirname(dir_of_file)
parents_parent_dir_of_file = dirname(parent_dir_of_file)
sys.path.append(parents_parent_dir_of_file + "/cuBERT-topic-modelling/")
from vectorizer import CountVecWrapper


data = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

docs_df = pd.DataFrame(data, columns=["Document"])


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


def test_trivia_case():
    docs_df_gpu = cudf.from_pandas(docs_df)
    clean_docs_bertopic = preprocess_text_bertopic(docs_df_gpu["Document"])
    cv = CountVecWrapper()
    clean_docs_gpu = cv.preprocess_text_gpu(docs_df_gpu["Document"])

    cudf.testing.testing.assert_series_equal(clean_docs_gpu,
                                             clean_docs_bertopic)