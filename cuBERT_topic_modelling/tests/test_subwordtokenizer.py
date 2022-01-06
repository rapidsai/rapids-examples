from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import cupy as cp
from sklearn.datasets import fetch_20newsgroups
import cudf
import pytest
from embedding_extraction import mean_pooling, create_embeddings, tokenize_strings
from sentence_transformers import SentenceTransformer

from cudf.core.subword_tokenizer import SubwordTokenizer
from torch.utils.dlpack import to_dlpack, from_dlpack


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_tokenization():
    data = fetch_20newsgroups(subset="all")["data"]
    ## sentence number 11927 is being encoded different in \
    ## AutoTokenizer and cuDF's SubwordTokenizer"
    data = data[:11927] + data[11928:]

    cudf_tokenizer = SubwordTokenizer(
        hash_file="vocab/voc_hash.txt", do_lower_case=True
    )

    cudf_td = tokenize_strings(cudf.Series(data), tokenizer=cudf_tokenizer)

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    st_td = st_model.tokenize(data)

    np.testing.assert_array_almost_equal(
        cudf_td["attention_mask"].to("cpu").numpy(), st_td["attention_mask"].numpy()
    )

    np.testing.assert_equal(
        cudf_td["attention_mask"].to("cpu").numpy(), st_td["attention_mask"].numpy()
    )


## Todo: Remove Below
def fix_padding(tnsr):
    """Function to fix padding on a torch.Tensor object

    Args:
        tnsr ([torch.Tensor]): Tensor representing input_ids,
        attention_mask

    Returns:
        [torch.Tensor]: trimmed stack of Tensor objects
    """

    # Remove all the padding from the end
    trimmed_collections = list()
    max_arr_length = -1
    dx = to_dlpack(tnsr)
    embeddings_collecton = cp.fromDlpack(dx)
    for embeddings in embeddings_collecton:
        trimmed = cp.trim_zeros(embeddings, trim="b")
        max_arr_length = max(max_arr_length, len(trimmed))
        trimmed_collections.append(trimmed)

    first_arr_stack = cp.pad(
        trimmed_collections[0],
        (0, max_arr_length - len(trimmed_collections[0])),
        "constant",
    )

    # Add the required padding back
    for a in range(1, len(trimmed_collections)):
        padded = cp.pad(
            trimmed_collections[a],
            (0, max_arr_length - len(trimmed_collections[a])),
            "constant",
        )
        first_arr_stack = cp.vstack([first_arr_stack, padded])
        # Convert it back to a PyTorch tensor.
    tx2 = from_dlpack(first_arr_stack.toDlpack())

    # Taking care of case where we have only one sentence
    # Then, we need to reshape to get the right dimensions
    # since in the other cases cp.vstack handles that.

    if len(tx2.shape) == 1:
        dim = tx2.shape[0]
        tx2 = torch.reshape(tx2, (1, dim))

    return tx2


# Sentences we want sentence embeddings for
@pytest.fixture
def input_sentences_fixture():
    sentences = fetch_20newsgroups(subset="all")["data"]
    return sentences


def run_embedding_creation_transformers(sentences):
    # Load AutoModel from huggingface model repository
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Tokenize sentences
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    return sentence_embeddings


@pytest.mark.xfail(
    reason="sentence number 11927 is being encoded different in \
        AutoTokenizer and cuDF's SubwordTokenizer",
    strict=True,
)
def test_custom_tokenizer(input_sentences_fixture):
    sentence_embeddings_gpu = create_embeddings(cudf.Series(input_sentences_fixture))
    sentence_embeddings = run_embedding_creation_transformers(input_sentences_fixture)
    np.testing.assert_array_almost_equal(
        sentence_embeddings.to("cpu").numpy(), sentence_embeddings_gpu.to("cpu").numpy()
    )


@pytest.mark.xfail(
    reason="sentence number 11927 is being encoded different in \
        AutoTokenizer and cuDF's SubwordTokenizer",
    strict=True,
)
def test_encoded_input(input_sentences_fixture):
    cudf_tokenizer = SubwordTokenizer("vocab/voc_hash.txt", do_lower_case=True)
    input_sentences_fixture_cudf = cudf.Series(input_sentences_fixture)
    # Tokenize sentences
    encoded_input_cudf = cudf_tokenizer(
        input_sentences_fixture_cudf,
        max_length=128,
        max_num_rows=len(input_sentences_fixture_cudf),
        padding="max_length",
        return_tensors="pt",
        truncation=True,
    )

    encoded_input_cudf["input_ids"] = fix_padding(encoded_input_cudf["input_ids"])
    encoded_input_cudf["attention_mask"] = fix_padding(
        encoded_input_cudf["attention_mask"]
    )

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Tokenize sentences
    encoded_input = tokenizer(
        input_sentences_fixture,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    np.testing.assert_array_almost_equal(
        encoded_input_cudf["attention_mask"].to("cpu").numpy(),
        encoded_input["attention_mask"].numpy(),
    )
    np.testing.assert_array_almost_equal(
        encoded_input_cudf["input_ids"].to("cpu").numpy(),
        encoded_input["input_ids"].numpy(),
    )
