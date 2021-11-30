from cudf.core.subword_tokenizer import SubwordTokenizer
import torch
import cupy as cp
from torch.utils.dlpack import to_dlpack, from_dlpack
from transformers import AutoModel
from torch.utils.data import TensorDataset, DataLoader
import transformers
from cudf.utils.hash_vocab_utils import hash_vocab
hash_vocab('vocab.txt', 'voc_hash.txt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO: find a way to not iterate through the torch.Tensor
# using built-in CuPy/cuDF methods
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
        trimmed = cp.trim_zeros(embeddings, trim='b')
        max_arr_length = max(max_arr_length, len(trimmed))
        trimmed_collections.append(trimmed)

    first_arr_stack = cp.pad(
        trimmed_collections[0],
        (0, max_arr_length-len(trimmed_collections[0])),
        'constant')

    # Add the required padding back
    for a in range(1, len(trimmed_collections)):
        padded = cp.pad(
            trimmed_collections[a],
            (0, max_arr_length-len(trimmed_collections[a])),
            'constant')
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
    input_mask_expanded = attention_mask.\
        unsqueeze(-1).\
        expand(token_embeddings.size()).\
        float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def create_embeddings(sentences):
    """Creates the sentence embeddings using SentenceTransformer

    Args:
        sentences (cudf.Series[str]): a cuDF Series of Input strings

    Returns:
        embeddings (cupy.ndarray): corresponding sentence
        embeddings for the strings passed
    """

    cudf_tokenizer = SubwordTokenizer(
        'voc_hash.txt',
        do_lower_case=True
    )

    # Tokenize sentences
    encoded_input_cudf = cudf_tokenizer(
        sentences,
        max_length=128,
        max_num_rows=len(sentences),
        padding='max_length',
        return_tensors='pt',
        truncation=True
    )

    # Load AutoModel from huggingface model repository
    model_gpu = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    ).to(device)

    # Delete the key and associated values we do not need
    del encoded_input_cudf['metadata']

    encoded_input_cudf['input_ids'] = fix_padding(
        encoded_input_cudf['input_ids']
    )
    encoded_input_cudf['attention_mask'] = fix_padding(
        encoded_input_cudf['attention_mask']
    )

    batch_size = 64

    dataset = TensorDataset(
        encoded_input_cudf['input_ids'],
        encoded_input_cudf['attention_mask']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size
    )

    model_o_ls = []
    model_pooler_output = []
    with torch.no_grad():
        for data in dataloader:
            mapping = {}
            mapping['input_ids'] = data[0]
            mapping['attention_mask'] = data[1]
            model_obj = model_gpu(**mapping)
            model_o_ls.append(model_obj.last_hidden_state)
            model_pooler_output.append(model_obj.pooler_output)

    model_stacked_lhs = torch.cat(model_o_ls)
    model_stacked_po = torch.cat(model_pooler_output)

    bert_mod = transformers.\
        modeling_outputs.\
        BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=model_stacked_lhs,
            pooler_output=model_stacked_po
        )

    # Perform pooling. In this case, mean pooling
    sentence_embeddings_gpu = mean_pooling(
        bert_mod,
        encoded_input_cudf['attention_mask']
    )

    return sentence_embeddings_gpu
