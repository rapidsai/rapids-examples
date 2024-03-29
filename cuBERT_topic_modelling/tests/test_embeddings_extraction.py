import cudf
import torch
from embedding_extraction import create_embeddings

def test_extract_embeddings():
    """Test SentenceTransformer
    Check whether the embeddings are correctly generated
    for both a single string or a list of strings. This means that
    the correct shape should be outputted. The embeddings by itself
    should not exceed certain values as a sanity check.
    """
    single_embedding = create_embeddings(
        cudf.Series(["a document"])
    )
    multiple_embeddings = create_embeddings(
        cudf.Series(["a document",
                     "another document"])
    )

    assert single_embedding.shape[0] == 1
    assert single_embedding.shape[1] == 384
    assert torch.min(single_embedding) > -5
    assert torch.max(single_embedding) < 5

    assert multiple_embeddings.shape[0] == 2
    assert multiple_embeddings.shape[1] == 384
    assert torch.min(multiple_embeddings) > -5
    assert torch.max(multiple_embeddings) < 5
