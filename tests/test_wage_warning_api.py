from fastapi.testclient import TestClient

from app.main import app


def test_wage_warning_salary_missing_returns_200() -> None:
    with TestClient(app) as client:
        resp = client.post("/v1/wage-warning", json={"countryCode": "UA"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["code"] == "200"
    assert "제안 임금 정보가 제공되지" in data["data"]["warningMessage"]


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

    async def _avg(_cc: str) -> float:
        return 20000.0

    monkeypatch.setattr(wage_service, "get_min_wage", _min)
    monkeypatch.setattr(wage_service, "get_avg_wage", _avg)

    with TestClient(app) as client:
        resp = client.post("/v1/wage-warning", json={"countryCode": "KR", "salaryText": "KRW 12000/h"})
    assert resp.status_code == 200
    msg = resp.json()["data"]["warningMessage"]
    assert "최저임금" in msg


def test_wage_warning_high_salary_warning(monkeypatch) -> None:
    import app.services.wage_service as wage_service

    async def _min(_cc: str) -> float:
        return 10.0

    async def _avg(_cc: str) -> float:
        return 10.0

    monkeypatch.setattr(wage_service, "get_min_wage", _min)
    monkeypatch.setattr(wage_service, "get_avg_wage", _avg)

    with TestClient(app) as client:
        resp = client.post("/v1/wage-warning", json={"countryCode": "KR", "salaryText": "KRW 30/h"})
    assert resp.status_code == 200
    msg = resp.json()["data"]["warningMessage"]
    assert "평균" in msg

