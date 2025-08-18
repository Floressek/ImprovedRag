def test_placeholder_reranker_present():
    from src.ragx.retrieval.reranker import rerank
    assert callable(rerank)
