import cupy as cp
from cuml.metrics import pairwise_distances


# Reference: https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_mmr.py
def mmr(
    doc_embedding,
    word_embeddings,
    words,
    top_n=5,
    diversity=0.8,
):

    """
    Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.
    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.
    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyhprases to return
        diversity: How diverse the select keywords/keyphrases are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.
    Returns:
         List[str]: The selected keywords/keyphrases
    """

    # Extract similarity within words, and between words and the document
    word_doc_similarity = 1 - pairwise_distances(
        word_embeddings, doc_embedding, metric="cosine"
    )
    word_similarity = 1 - pairwise_distances(word_embeddings, metric="cosine")

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = cp.argmax(word_doc_similarity)
    target = cp.take(keywords_idx, 0)
    candidates_idx = [i for i in range(len(words)) if i != target]
    for i in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        if i == 0:
            first_row = cp.reshape(
                word_similarity[candidates_idx][:, keywords_idx],
                (word_similarity[candidates_idx][:, keywords_idx].shape[0], 1),
            )
            target_similarities = cp.max(first_row, axis=1)
        else:
            target_similarities = cp.max(
                word_similarity[candidates_idx][:, keywords_idx], axis=1
            )
        # Calculate MMR
        mmr = (
            1 - diversity
        ) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)

        mmr_idx = cp.take(cp.array(candidates_idx), cp.argmax(mmr))

        # Update keywords & candidates
        keywords_idx = cp.append(keywords_idx, mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx.get()]
