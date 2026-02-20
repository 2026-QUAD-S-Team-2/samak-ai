from fastapi.testclient import TestClient

from app.main import app


def test_wage_warning_salary_missing_returns_200() -> None:
    with TestClient(app) as client:
        resp = client.post("/v1/wage-warning", json={"countryCode": "US"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["code"] == "200"
    assert data["data"]["warningMessage"] is not None
    assert "법정 최저 시급" in data["data"]["warningMessage"]


def test_wage_warning_salary_missing_unknown_country_returns_null() -> None:
    with TestClient(app) as client:
        resp = client.post("/v1/wage-warning", json={"countryCode": "ZZ"})
    assert resp.status_code == 200
    assert resp.json()["data"]["warningMessage"] is None


def test_wage_warning_parse_fail_returns_200() -> None:
    with TestClient(app) as client:
        resp = client.post("/v1/wage-warning", json={"countryCode": "KR", "salaryText": "KRW 12000"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["code"] == "200"
    assert "시급(hourly)" in data["data"]["warningMessage"]


def test_wage_warning_currency_mismatch_returns_200() -> None:
    with TestClient(app) as client:
        resp = client.post("/v1/wage-warning", json={"countryCode": "KR", "salaryText": "USD 25/h"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["code"] == "200"
    assert "기본 통화" in data["data"]["warningMessage"]


def test_wage_warning_min_wage_warning(monkeypatch) -> None:
    import app.services.wage_service as wage_service

    async def _min(_cc: str) -> float:
        return 15000.0

    monkeypatch.setattr(wage_service, "get_min_wage", _min)

    with TestClient(app) as client:
        resp = client.post("/v1/wage-warning", json={"countryCode": "KR", "salaryText": "KRW 12000/h"})
    assert resp.status_code == 200
    msg = resp.json()["data"]["warningMessage"]
    assert "최저임금" in msg


def test_wage_warning_high_salary_warning(monkeypatch) -> None:
    import app.services.wage_service as wage_service

    async def _min(_cc: str) -> float:
        return 10.0

    monkeypatch.setattr(wage_service, "get_min_wage", _min)

    with TestClient(app) as client:
        resp = client.post("/v1/wage-warning", json={"countryCode": "KR", "salaryText": "KRW 40/h"})
    assert resp.status_code == 200
    msg = resp.json()["data"]["warningMessage"]
    assert msg is not None
    assert "4배" in msg


def test_wage_warning_no_warning_returns_null(monkeypatch) -> None:
    import app.services.wage_service as wage_service

    async def _min(_cc: str) -> float:
        return 5000.0

    monkeypatch.setattr(wage_service, "get_min_wage", _min)

    with TestClient(app) as client:
        resp = client.post("/v1/wage-warning", json={"countryCode": "KR", "salaryText": "KRW 6000/h"})
    assert resp.status_code == 200
    assert resp.json()["data"]["warningMessage"] is None
