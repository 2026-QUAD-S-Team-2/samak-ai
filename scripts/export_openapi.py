from __future__ import annotations

"""
FastAPI OpenAPI 스펙을 JSON 파일로 내보내는 스크립트.

팀원이 서버를 직접 실행하지 않아도, `openapi.json`만 공유하면
Swagger UI(온라인 에디터/뷰어)에서 API 명세를 확인할 수 있습니다.
"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="openapi.json", help="Output path (default: openapi.json)")
    args = parser.parse_args()

    spec = app.openapi()
    out_path = Path(args.out)
    out_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ OpenAPI 스펙 저장 완료: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

