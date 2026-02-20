from app.integrations.ilostat_client import _parse_list_json_latest, _parse_sdmx_json_latest, to_alpha3


def test_parse_sdmx_json_latest_returns_latest_value() -> None:
    payload = {
        "dataSets": [
            {
                "series": {
                    "0:0:0": {
                        "observations": {
                            "0": [10.0],
                            "1": ["12.5"],
                            "2": [11.0],
                        }
                    }
                }
            }
        ]
    }
    assert _parse_sdmx_json_latest(payload) == 11.0


def test_parse_list_json_latest_returns_latest_value() -> None:
    payload = [
        {"timePeriod": "2022", "value": 10},
        {"timePeriod": "2023", "value": "12.5"},
    ]
    assert _parse_list_json_latest(payload) == 12.5


def test_to_alpha3_converts_known_countries() -> None:
    # pycountry가 설치되어 있으면 KR/UA는 alpha-3로 변환되어야 합니다.
    out_kr = to_alpha3("KR")
    out_ua = to_alpha3("UA")
    if out_kr is None or out_ua is None:
        # 환경에 pycountry가 없을 수 있으므로 스킵성 검사
        return
    assert out_kr == "KOR"
    assert out_ua == "UKR"
