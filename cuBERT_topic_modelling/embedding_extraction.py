from cudf.core.subword_tokenizer import SubwordTokenizer, _cast_to_appropriate_type
import torch
from torch.utils.data import TensorDataset, DataLoader

import time

# Vocabulary is included in the root directory of this repo
# however, below is the command to modify / update it -->
# from cudf.utils.hash_vocab_utils import hash_vocab
# hash_vocab('vocab.txt', 'voc_hash.txt')


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    """Function to implement mean pooling on top of the AutoModel
    See: https://www.sbert.net/examples/applications/computing-embeddings/README.html#sentence-embeddings-with-transformers

    Args:
        model_output \
            (transformers.BaseModelOutputWithPoolingAndCrossAttentions): BERT model
        attention_mask (torch.Tensor): torch.Tensor representing attention
        mask values

    Returns:
        [torch.Tensor]: correct averaging of attention mask
    """
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def tokenize_strings(sentences, tokenizer):
    max_length = 128

    # Tokenize cudf Series
    token_o = tokenizer(
        sentences,
        max_length=max_length,
        max_num_rows=len(sentences),
        padding="max_length",
        return_tensors="cp",
        truncation=True,
        add_special_tokens=True,
    )

    clip_len = max_length - int((token_o["input_ids"][:, ::-1] != 0).argmax(1).min())
    token_o["input_ids"] = _cast_to_appropriate_type(
        token_o["input_ids"][:, :clip_len], "pt"
    )
    token_o["attention_mask"] = _cast_to_appropriate_type(
        token_o["attention_mask"][:, :clip_len], "pt"
    )

    del token_o["metadata"]
    return token_o


def create_embeddings(sentences, embedding_model, vocab_file="vocab/voc_hash.txt"):
    """Creates the sentence embeddings using SentenceTransformer

    Args:
        sentences (cudf.Series[str]): a cuDF Series of Input strings

    Returns:
        embeddings (cupy.ndarray): corresponding sentence
        embeddings for the strings passed
    """

    cudf_tokenizer = SubwordTokenizer(vocab_file, do_lower_case=True)
    batch_size = 64
    pooling_output_ls = []
    model_st = time.time()
    with torch.no_grad():
        for s_ind in range(0, len(sentences), batch_size):
            e_ind = min(s_ind + batch_size, len(sentences))
            b_s = sentences[s_ind:e_ind]

            tokenized_d = tokenize_strings(b_s, cudf_tokenizer)
            b_input_ids = tokenized_d["input_ids"]
            b_attention_mask = tokenized_d["attention_mask"]

            model_obj = embedding_model(
                **{"input_ids": b_input_ids, "attention_mask": b_attention_mask}
            )
            pooling_output_ls.append(mean_pooling(model_obj, b_attention_mask))

    pooling_output = torch.cat(pooling_output_ls)
    model_et = time.time()
    print(f"DL time = {model_et-model_st}")

    return pooling_output
