from src.rag_system.ingestion.pipeline import run_ingestion

def test_run_ingestion_placeholder():
    store = run_ingestion("/path/to/wiki.xml")
    assert hasattr(store, "add")
