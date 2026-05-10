from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_cors_allows_deployed_frontend_origin() -> None:
    client = TestClient(app)

    response = client.options(
        "/api/health",
        headers={
            "Access-Control-Request-Method": "GET",
            "Origin": "https://bamti.stableh.com",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://bamti.stableh.com"
