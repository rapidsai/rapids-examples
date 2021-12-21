from bertopic._mmr import mmr as mmr_cpu
from mmr import mmr as mmr_gpu
import numpy as np
import cupy as cp
import pytest

@pytest.mark.parametrize("words,diversity",
                         [(['stars', 'star', 'starry', 'astronaut', 'astronauts'], 0),
                          (['stars', 'spaceship', 'nasa', 'skies', 'sky'], 1)])
def test_mmr_model(words, diversity):
    """ Test MMR
    Testing both low and high diversity when selecing candidates.
    In the parameters, you can see that low diversity leads to very
    similar words/vectors to be selected, whereas a high diversity
    leads to a selection of candidates that, albeit similar to the input
    document, are less similar to each other.
    """
    candidates_gpu = mmr_gpu(doc_embedding=cp.array([5, 5, 5, 5]).reshape(1, -1),
                     word_embeddings=cp.array([[1.0, 1.0, 2.0, 2.0],
                                               [1.0, 2.0, 4.0, 7.0],
                                               [4.0, 4.0, 4.0, 4.0],
                                               [4.0, 4.0, 4.0, 4.0],
                                               [4.0, 4.0, 4.0, 4.0],
                                               [1.0, 1.0, 9.0, 3.0],
                                               [5.0, 3.0, 5.0, 8.0],
                                               [6.0, 6.0, 6.0, 6.0],
                                               [6.0, 6.0, 6.0, 6.0],
                                               [5.0, 8.0, 7.0, 2.0]]),
                     words=['space', 'nasa', 'stars', 'star', 'starry', 'spaceship',
                            'sky', 'astronaut', 'astronauts', 'skies'],
                     diversity=diversity)
    
    candidates_cpu = mmr_cpu(doc_embedding=np.array([5, 5, 5, 5]).reshape(1, -1),
                     word_embeddings=np.array([[1.0, 1.0, 2.0, 2.0],
                                               [1.0, 2.0, 4.0, 7.0],
                                               [4.0, 4.0, 4.0, 4.0],
                                               [4.0, 4.0, 4.0, 4.0],
                                               [4.0, 4.0, 4.0, 4.0],
                                               [1.0, 1.0, 9.0, 3.0],
                                               [5.0, 3.0, 5.0, 8.0],
                                               [6.0, 6.0, 6.0, 6.0],
                                               [6.0, 6.0, 6.0, 6.0],
                                               [5.0, 8.0, 7.0, 2.0]]),
                     words=['space', 'nasa', 'stars', 'star', 'starry', 'spaceship',
                            'sky', 'astronaut', 'astronauts', 'skies'],
                     diversity=diversity)
    
    assert candidates_gpu == words == candidates_cpu
