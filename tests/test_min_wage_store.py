from app.services.min_wage_store import get_min_wage_local


def test_get_min_wage_local_known_country() -> None:
    out = get_min_wage_local("KR")
    assert out is not None
    hourly, currency, as_of = out
    assert hourly > 0
    assert currency == "KRW"
    assert as_of


def test_get_min_wage_local_unknown_country_returns_none() -> None:
    assert get_min_wage_local("ZZ") is None

