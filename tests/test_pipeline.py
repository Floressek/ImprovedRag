def test_placeholder_baseline_pipeline():
    from src.ragx.pipeline.baseline import run
    out = run("What?", [])
    assert isinstance(out, str)
