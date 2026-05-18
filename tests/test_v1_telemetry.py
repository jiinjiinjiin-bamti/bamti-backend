from fastapi.testclient import TestClient

from app.main import app


def test_telemetry_runs_saves_json_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.api.v1.telemetry.settings.telemetry_runs_dir", tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/v1/telemetry/runs",
        json={
            "label": "one-minute-performance",
            "durationSec": 60,
            "frontend": {
                "transmissionFps": {"avg": 9.44},
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["saved"] is True
    assert payload["fileName"].endswith("_one-minute-performance.json")
    assert (tmp_path / payload["fileName"]).exists()


def test_telemetry_runs_lists_saved_files(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.api.v1.telemetry.settings.telemetry_runs_dir", tmp_path)
    (tmp_path / "2026-05-09T00-00-00KST_one-minute-performance.json").write_text("{}", encoding="utf-8")
    client = TestClient(app)

    response = client.get("/api/v1/telemetry/runs")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["fileName"] == "2026-05-09T00-00-00KST_one-minute-performance.json"
