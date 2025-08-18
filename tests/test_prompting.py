def test_placeholder_prompt_builder():
    from src.ragx.generation.prompts.builder import build_answer_with_citations
    out = build_answer_with_citations("q", ["c1", "c2"])
    assert "[question]" in out
