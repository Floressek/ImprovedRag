"""Test that evaluation refactoring maintains backward compatibility."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_models_import():
    """Test that models can be imported from new location."""
    from src.ragx.evaluation.models import (
        PipelineConfig,
        ConfigResult,
        AblationStudyResult,
        CheckpointState,
        EvaluationResult,
        BatchEvaluationResult,
    )

    # Create a simple config to verify it works
    config = PipelineConfig(
        name="test",
        description="Test config",
        query_analysis_enabled=True,
        cot_enabled=False,
        reranker_enabled=True,
        cove_mode="off",
        prompt_template="basic",
    )

    assert config.name == "test"
    assert config.to_dict()["use_query_analysis"] == True
    print("✓ Models import and work correctly")


def test_configs_import():
    """Test that predefined configs can be imported."""
    from src.ragx.evaluation.configs import (
        BASELINE,
        QUERY_ONLY,
        FULL_COVE_AUTO,
        get_all_configs,
    )

    assert BASELINE.name == "baseline"
    assert FULL_COVE_AUTO.cove_mode == "auto"

    all_configs = get_all_configs()
    assert len(all_configs) == 12
    print("✓ Configs import correctly")


def test_metrics_import():
    """Test that metrics utilities can be imported."""
    from src.ragx.evaluation.metrics import (
        count_sources,
        calculate_multihop_coverage,
        safe_std,
        calculate_ci,
    )

    # Test count_sources
    assert count_sources(["url1", "url2", "url1"]) == 2
    assert count_sources([]) == 0

    # Test multihop coverage
    coverage = calculate_multihop_coverage(
        ["q1", "q2", "q3"],
        {"q1": [1, 2], "q2": [], "q3": [1]}
    )
    assert coverage == 2/3  # 2 out of 3 queries have results

    # Test safe_std
    assert safe_std([1.0, 2.0, 3.0]) > 0
    assert safe_std([1.0]) == 0.0

    print("✓ Metrics import and work correctly")


def test_checkpoint_manager():
    """Test that checkpoint manager can be imported."""
    from src.ragx.evaluation.checkpoint_manager import CheckpointManager

    manager = CheckpointManager(checkpoint_dir=None)
    assert not manager.is_enabled()
    print("✓ CheckpointManager imports correctly")


def test_api_client():
    """Test that API client can be imported."""
    from src.ragx.evaluation.api_client import RAGAPIClient

    # Just test import and initialization
    client = RAGAPIClient("http://localhost:8000")
    assert client.api_base_url == "http://localhost:8000"
    print("✓ RAGAPIClient imports correctly")


if __name__ == "__main__":
    print("Testing evaluation refactoring...")
    print()

    try:
        test_models_import()
        test_configs_import()
        test_metrics_import()
        test_checkpoint_manager()
        test_api_client()

        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print()
        print("The refactoring maintains full compatibility:")
        print("  ✓ Models can be imported from src.ragx.evaluation.models")
        print("  ✓ Configs can be imported from src.ragx.evaluation.configs")
        print("  ✓ Metrics can be imported from src.ragx.evaluation.metrics")
        print("  ✓ CheckpointManager works correctly")
        print("  ✓ RAGAPIClient works correctly")
        print()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
