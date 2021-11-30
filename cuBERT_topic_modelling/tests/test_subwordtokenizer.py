from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import cudf
import pytest
from embedding_extraction import mean_pooling, create_embeddings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Sentences we want sentence embeddings for
@pytest.fixture
def input_sentences_fixture():
    sentences = fetch_20newsgroups(subset="all")["data"]
    return sentences


def run_embedding_creation_transformers(sentences):
    # Load AutoModel from huggingface model repository
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Tokenize sentences
    encoded_input = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(
        model_output,
        encoded_input["attention_mask"]
    )

    return sentence_embeddings

@pytest.mark.xfail(
    reason="sentence number 11927 is being encoded different in \
        AutoTokenizer and cuDF's SubwordTokenizer",
    strict=True)
def test_custom_tokenizer(input_sentences_fixture):
    sentence_embeddings_gpu = create_embeddings(
        cudf.Series(input_sentences_fixture)
    )
    sentence_embeddings = run_embedding_creation_transformers(
        input_sentences_fixture
    )
    np.testing.assert_array_almost_equal(
        sentence_embeddings.to("cpu").numpy(),
        sentence_embeddings_gpu.to("cpu").numpy()
    )
