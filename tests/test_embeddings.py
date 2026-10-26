from collections import OrderedDict
from unittest.mock import Mock

import numpy as np

from src.embeddings import Embedder


def test_cache_and_uncached_query_contract():
    encoder = Embedder.__new__(Embedder)
    encoder.model_name = "test-model"
    encoder.model = Mock()
    encoder.model.encode.side_effect = lambda texts, **kwargs: np.array([[1., 0.] for _ in texts])
    encoder.cache = OrderedDict()
    encoder.encode(["passage"])
    encoder.encode(["passage"])
    assert encoder.model.encode.call_count == 1
    encoder.encode(["query"], use_cache=False)
    encoder.encode(["query"], use_cache=False)
    assert encoder.model.encode.call_count == 3
    assert len(encoder.cache) == 1
