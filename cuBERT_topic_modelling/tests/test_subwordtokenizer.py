from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from cuBERTopic import gpu_BERTopic
from sklearn.datasets import fetch_20newsgroups
import cudf
import pytest

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Sentences we want sentence embeddings for

# TODO: sentence number 11927 is being encoded different in AutoTokenizer
# and cuDF's SubwordTokenizer, look into that - this is causing high margin of error.

sentences = fetch_20newsgroups(subset="all")["data"][:10000]


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

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

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

@pytest.mark.parametrize("sentences", [(sentences)])
def test_custom_tokenizer(sentences):
    gpu_topic = gpu_BERTopic()
    sentence_embeddings_gpu = gpu_topic.create_embeddings(
        cudf.Series(sentences)
    )
    sentence_embeddings = run_embedding_creation_transformers(sentences)
    np.testing.assert_array_almost_equal(
        sentence_embeddings.to("cpu").numpy(),
        sentence_embeddings_gpu.to("cpu").numpy()
    )
