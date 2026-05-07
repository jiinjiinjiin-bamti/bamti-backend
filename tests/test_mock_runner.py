from app.inference.mock_runner import MockRunner


async def test_mock_runner_returns_attentive_result() -> None:
    result = await MockRunner().infer(b"jpeg")

    assert result.is_distracted is False
    assert result.label == "attentive"
    assert result.confidence == 0.99
