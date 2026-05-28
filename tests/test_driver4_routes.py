from fastapi.testclient import TestClient

from app.inference.schemas import DetectionClass, ModelManifest
from app.main import app


def _driver4_manifest() -> ModelManifest:
    return ModelManifest(
        model_version="final_model_0528",
        classes=(
            DetectionClass(
                variable_name="body_touching",
                class_id="body_touching",
                display_name="신체 만짐",
                description="Driver4 model class: 신체만짐",
                threshold=0.65,
            ),
            DetectionClass(
                variable_name="distraction",
                class_id="distraction",
                display_name="주의 분산",
                description="Driver4 model class: 주의분산",
                threshold=0.65,
            ),
            DetectionClass(
                variable_name="phone_operation",
                class_id="phone_operation",
                display_name="핸드폰 조작",
                description="Driver4 model class: 핸드폰 조작",
                threshold=0.65,
            ),
            DetectionClass(
                variable_name="steering_operation",
                class_id="steering_operation",
                display_name="핸들 조작",
                description="Driver4 model class: 핸들 조작",
                threshold=0.65,
            ),
        ),
    )


def test_driver4_versioned_detection_classes_routes_use_driver4_runner(monkeypatch) -> None:
    requested_runner_names: list[str] = []

    def fake_get_model_manifest(name: str):
        requested_runner_names.append(name)
        return _driver4_manifest()

    monkeypatch.setattr("app.api.driver.inference.get_model_manifest", fake_get_model_manifest)
    client = TestClient(app)

    for version in ("v4", "v6"):
        response = client.get(f"/api/driver/{version}/detection-classes")
        assert response.status_code == 200
        assert [item["variableName"] for item in response.json()["classes"]] == [
            "body_touching",
            "distraction",
            "phone_operation",
            "steering_operation",
        ]

    assert requested_runner_names == ["driver4-torch", "driver4-torch"]


def test_driver4_mobile_session_routes_support_v4_and_v6_prefixes() -> None:
    client = TestClient(app)

    for version in ("v4", "v6"):
        api_version = f"driver4-{version}"
        response = client.post(
            f"/api/driver/{version}/mobile/sessions",
            json={"cameraUrlBase": f"https://example.test/camera?sessionId={{sessionId}}&apiVersion={api_version}"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["sessionId"].startswith("mobile-")
        assert payload["cameraUrl"].endswith(f"sessionId={payload['sessionId']}&apiVersion={api_version}")
