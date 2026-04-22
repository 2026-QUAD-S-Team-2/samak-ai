from __future__ import annotations

"""
데이터 전처리 모듈 (MVP 평가 가능 버전).

목표:
- "라벨이 섞인(supervised) 데이터"로만 의미 있는 test 셋을 만든다.
- all-positive / all-negative 데이터셋은 train 증강(augment) 용도로만 사용한다.

출력(기본: training/data/processed/):
- combined_train.csv: RecruitmentScam train split + (옵션) augment 데이터 합친 전체 train 풀
- train.csv: 모델 학습에 실제로 사용할 파일(현재는 combined_train.csv와 동일 스키마)
- test.csv: 평가용 파일(RecruitmentScam test split ONLY)  ← 핵심
- (선택) valid.csv: RecruitmentScam에서 stratified valid split
"""

import argparse
import hashlib
from pathlib import Path
import re
import sys
import json
from typing import Any, Iterable, Sequence

# `python training/preprocess.py`로 실행 시에도 동작하도록 경로 보정
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# 원본 공통 컬럼(가능한 범위 내에서만 사용)
TARGET_COLUMNS: tuple[str, ...] = (
    "title",
    "description",
    "requirements",
    "company_profile",
    "location",
    "salary_range",
    "employment_type",
    "industry",
    "benefits",
    "fraudulent",
)

# MVP 1차에서는 제거하기로 한 컬럼들
DROP_COLUMNS: tuple[str, ...] = (
    "department",
    "function",
    "telecommuting",
    "has_company_logo",
    "has_questions",
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
_PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{4}",
)
_WS_RE = re.compile(r"\s+")
_LONG_NUM_RE = re.compile(r"\d{6,}")  # 너무 긴 숫자는 토큰으로 치환
_LEAK_RE = re.compile(
    r"\b(fraudulent|scam|fake\s+job|fake\s+posting|not\s+a\s+scam|not\s+scam)\b",
    re.IGNORECASE,
)


def build_document_text(row: dict[str, str]) -> str:
    """
    각 row를 하나의 document(text)로 통합합니다.

    (학습/평가/서빙에서 같은 형태를 유지)
    [TITLE] ...
    [LOCATION] ...
    [EMPLOYMENT_TYPE] ...
    [INDUSTRY] ...
    [SALARY] ...
    [COMPANY_PROFILE] ...
    [DESCRIPTION] ...
    [REQUIREMENTS] ...
    [BENEFITS] ...
    """
    title = row.get("title", "")
    description = row.get("description", "")
    requirements = row.get("requirements", "")
    benefits = row.get("benefits", "")

    location = row.get("location", "")
    employment_type = row.get("employment_type", "")
    industry = row.get("industry", "")
    salary_range = row.get("salary_range", "")
    company_profile = row.get("company_profile", "")

    return (
        f"[TITLE] {title}\n"
        f"[LOCATION] {location}\n"
        f"[EMPLOYMENT_TYPE] {employment_type}\n"
        f"[INDUSTRY] {industry}\n"
        f"[SALARY] {salary_range}\n"
        f"[COMPANY_PROFILE] {company_profile}\n"
        f"[DESCRIPTION] {description}\n"
        f"[REQUIREMENTS] {requirements}\n"
        f"[BENEFITS] {benefits}\n"
    )


def clean_text(text: str) -> str:
    """
    필수 클린업:
    - HTML 태그 제거
    - URL/이메일 토큰 치환
    - 너무 긴 숫자 정리
    - 소문자 변환
    - 다중 공백 제거
    """
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" <URL> ", text)
    text = _EMAIL_RE.sub(" <EMAIL> ", text)
    text = _PHONE_RE.sub(" <PHONE> ", text)
    text = _LONG_NUM_RE.sub(" <LONGNUM> ", text)
    text = text.lower()
    text = _WS_RE.sub(" ", text).strip()
    return text


def mask_label_leakage(text: str) -> str:
    # Kaggle류 데이터에 가끔 라벨 누수 키워드가 직접 포함되어 성능 착시가 발생할 수 있어,
    # 아주 명백한 단어만 최소한으로 마스킹합니다.
    return _LEAK_RE.sub(" <LEAK> ", text)


def truncate_text(text: str, *, max_chars: int) -> str:
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def parse_salary_range(s: str) -> tuple[float | None, float | None]:
    """
    salary_range 파싱 로직(간단 버전)
    - "50000-70000", "50,000 - 70,000", "10-15" 형태를 우선 지원
    - 실패하면 (None, None)
    """
    if s is None:
        return None, None
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return None, None

    normalized = s.replace(",", " ").replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[^0-9kK\.\-\s]", " ", normalized)
    normalized = _WS_RE.sub(" ", normalized).strip()

    matches = re.findall(r"\d+(?:\.\d+)?(?:[kK])?", normalized)
    if not matches:
        return None, None

    def to_float(tok: str) -> float:
        mult = 1.0
        if tok.lower().endswith("k"):
            mult = 1000.0
            tok = tok[:-1]
        return float(tok) * mult

    nums = [to_float(m) for m in matches[:2]]
    if len(nums) == 1:
        return nums[0], nums[0]
    low, high = nums[0], nums[1]
    if low > high:
        low, high = high, low
    return low, high


def salary_features(salary_range: str) -> dict[str, float | int | None]:
    low, high = parse_salary_range(salary_range)
    has_salary = 1 if (low is not None or high is not None) else 0
    mid = None
    if low is not None and high is not None:
        mid = (low + high) / 2.0
    return {"salary_low": low, "salary_high": high, "salary_mid": mid, "has_salary": has_salary}


def location_features(location: str) -> dict[str, int]:
    loc = "" if location is None else str(location).strip()
    return {"has_location": 1 if loc != "" else 0, "location_len": len(loc)}


def append_feature_tokens(text: str, *, feats: dict[str, object]) -> str:
    # TF-IDF + LR에서도 활용 가능하도록 숫자 피처를 토큰 형태로 문서에 주입합니다.
    tokens: list[str] = []
    if feats.get("has_salary") is not None:
        tokens.append(f"has_salary={int(feats['has_salary'])}")
    if feats.get("salary_low") is not None:
        tokens.append(f"salary_low={int(round(float(feats['salary_low'])))}")
    if feats.get("salary_high") is not None:
        tokens.append(f"salary_high={int(round(float(feats['salary_high'])))}")
    if feats.get("salary_mid") is not None:
        tokens.append(f"salary_mid={int(round(float(feats['salary_mid'])))}")
    if feats.get("has_location") is not None:
        tokens.append(f"has_location={int(feats['has_location'])}")
    return (text + "\n" + " ".join(tokens)).strip()


def default_input_paths() -> list[Path]:
    raw_dir = Path("training/data/raw")
    candidates = [
        raw_dir / "FakeJobPostings.csv",
        raw_dir / "LinkedInPostings.csv",
        raw_dir / "RecruitmentScam.csv",
    ]
    return [p for p in candidates if p.exists()]


def load_csv(path: Path, *, nrows: int | None = None):
    # pandas는 inference 서비스에는 불필요하므로, training 커맨드에서만 import 합니다.
    import pandas as pd  # type: ignore

    return pd.read_csv(path, encoding_errors="replace", nrows=nrows)


def normalize_columns(cols: Iterable[str]) -> list[str]:
    normed: list[str] = []
    for c in cols:
        c2 = str(c).strip().lower().replace(" ", "_").replace("-", "_")
        normed.append(c2)
    return normed


def coerce_fraudulent(series) -> "object":
    try:
        return series.fillna(0).astype(int)
    except Exception:  # noqa: BLE001
        return series.apply(lambda x: 1 if str(x).strip().lower() in {"1", "true", "t", "yes", "y"} else 0)


def ensure_columns(df) -> "object":
    for c in TARGET_COLUMNS:
        if c not in df.columns:
            df[c] = 0 if c == "fraudulent" else ""
    return df


def is_linkedin(df) -> bool:
    cols = set(df.columns)
    return "skills_desc" in cols and "job_posting_url" in cols


def map_linkedin_to_target(df) -> "object":
    """
    LinkedIn 정상 데이터 처리 규칙

    - fraud 라벨 없음 → fraudulent = 0 (정상)
    - 사용할 컬럼:
      title, description, skills_desc, location, min_salary, max_salary, currency,
      employment_type(formatted_work_type or work_type)
    - description + skills_desc를 합쳐 description 생성
    """
    import pandas as pd  # type: ignore

    df["title"] = df["title"] if "title" in df.columns else ""
    df["location"] = df["location"] if "location" in df.columns else ""

    desc = df["description"] if "description" in df.columns else pd.Series([""] * len(df))
    skills = df["skills_desc"] if "skills_desc" in df.columns else pd.Series([""] * len(df))
    desc = desc.fillna("").astype(str)
    skills = skills.fillna("").astype(str)
    df["description"] = (desc + "\n" + skills).str.strip()

    if "formatted_work_type" in df.columns and "work_type" in df.columns:
        a = df["formatted_work_type"].fillna("").astype(str)
        b = df["work_type"].fillna("").astype(str)
        df["employment_type"] = a.where(a.str.strip() != "", b)
    elif "formatted_work_type" in df.columns:
        df["employment_type"] = df["formatted_work_type"].fillna("").astype(str)
    elif "work_type" in df.columns:
        df["employment_type"] = df["work_type"].fillna("").astype(str)
    else:
        df["employment_type"] = ""

    # min/max salary가 아예 없을 수도 있으므로, round()가 가능한 float NaN 시리즈로 안전하게 초기화
    minv = (
        pd.to_numeric(df["min_salary"], errors="coerce")
        if "min_salary" in df.columns
        else pd.Series([float("nan")] * len(df), index=df.index)
    )
    maxv = (
        pd.to_numeric(df["max_salary"], errors="coerce")
        if "max_salary" in df.columns
        else pd.Series([float("nan")] * len(df), index=df.index)
    )
    cur = df["currency"].fillna("").astype(str) if "currency" in df.columns else pd.Series([""] * len(df))

    min_int = minv.round(0).astype("Int64")
    max_int = maxv.round(0).astype("Int64")
    min_str = min_int.astype(str).where(min_int.notna(), "")
    max_str = max_int.astype(str).where(max_int.notna(), "")

    range_str = min_str.where(min_str != "", max_str)
    both = (min_str != "") & (max_str != "")
    range_str = range_str.where(~both, min_str + "-" + max_str)

    has_cur = cur.str.strip() != ""
    df["salary_range"] = range_str.where(~has_cur, (range_str + " " + cur).str.strip())
    df["salary_range"] = df["salary_range"].fillna("").astype(str)

    df["fraudulent"] = 0

    df["requirements"] = ""
    df["company_profile"] = ""
    df["industry"] = ""
    df["benefits"] = ""

    return df


def select_and_clean(df) -> "object":
    drop = [c for c in DROP_COLUMNS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    df = ensure_columns(df)
    df = df.loc[:, list(TARGET_COLUMNS)]

    text_cols = [c for c in TARGET_COLUMNS if c != "fraudulent"]
    df[text_cols] = df[text_cols].fillna("")
    df["fraudulent"] = coerce_fraudulent(df["fraudulent"])
    return df


def _load_recruitment_scam(path: Path, *, nrows: int | None) -> "object":
    df = load_csv(path, nrows=nrows)
    df.columns = normalize_columns(df.columns)
    df = select_and_clean(df)
    df["source"] = "RecruitmentScam"
    return df


def _load_fakejob_pos_only(path: Path, *, nrows: int | None) -> "object":
    df = load_csv(path, nrows=nrows)
    df.columns = normalize_columns(df.columns)
    df = select_and_clean(df)
    # all-positive 여부 검사
    vc = df["fraudulent"].value_counts(dropna=False).to_dict()
    pos = int(vc.get(1, 0))
    neg = int(vc.get(0, 0))
    if pos > 0 and neg == 0:
        df["source"] = "FakeJobPostings_PosOnly"
    else:
        # (확장성) 라벨이 섞인 경우엔 supervised source로 취급 가능
        df["source"] = "FakeJobPostings"
    return df


def _load_linkedin_neg_only(path: Path, *, nrows: int | None) -> "object":
    df = load_csv(path, nrows=nrows)
    df.columns = normalize_columns(df.columns)
    if is_linkedin(df):
        df = map_linkedin_to_target(df)
    df = select_and_clean(df)
    df["fraudulent"] = 0
    df["source"] = "LinkedIn_NegOnly"
    return df


def _finalize_rows(
    df,
    *,
    max_text_chars: int,
    min_text_chars: int,
    mask_leakage: bool,
    dedupe: bool,
) -> "object":
    # 문서 텍스트 생성 + 클린업 + 길이 제한 + (토큰 주입용) 피처 계산
    def _row_to_text(r) -> str:
        row_dict = {k: ("" if r[k] is None else str(r[k])) for k in TARGET_COLUMNS if k != "fraudulent"}
        doc = build_document_text(row_dict)
        doc = clean_text(doc)
        if mask_leakage:
            doc = mask_label_leakage(doc)
        doc = truncate_text(doc, max_chars=max_text_chars)
        feats: dict[str, object] = {}
        feats.update(salary_features(row_dict.get("salary_range", "")))
        feats.update(location_features(row_dict.get("location", "")))
        return append_feature_tokens(doc, feats=feats)

    df["text"] = df.apply(_row_to_text, axis=1)

    sal = df["salary_range"].apply(lambda s: salary_features(str(s)))
    df["salary_low"] = sal.apply(lambda d: d["salary_low"])
    df["salary_high"] = sal.apply(lambda d: d["salary_high"])
    df["salary_mid"] = sal.apply(lambda d: d["salary_mid"])
    df["has_salary"] = sal.apply(lambda d: d["has_salary"])

    locf = df["location"].apply(lambda s: location_features(str(s)))
    df["has_location"] = locf.apply(lambda d: d["has_location"])
    df["location_len"] = locf.apply(lambda d: d["location_len"])

    out = df.loc[
        :,
        [
            "text",
            "fraudulent",
            "salary_low",
            "salary_high",
            "salary_mid",
            "has_salary",
            "has_location",
            "location_len",
            "source",
        ],
    ].copy()

    out["text_len"] = out["text"].astype(str).apply(len)
    out = out.loc[out["text_len"] >= int(min_text_chars)].reset_index(drop=True)

    out["text_hash"] = out["text"].astype(str).apply(
        lambda t: hashlib.sha256(t.encode("utf-8", errors="ignore")).hexdigest()
    )
    if dedupe:
        before = len(out)
        out = out.drop_duplicates(subset=["text_hash"]).reset_index(drop=True)
        after = len(out)
        if before != after:
            print(f"[전처리] 중복 제거: {before - after}개 제거")
    return out


def _print_source_stats(df, *, label: str) -> None:
    if "source" not in df.columns:
        return
    grouped = df.groupby("source")["fraudulent"].value_counts(dropna=False).unstack(fill_value=0)
    print(f"[전처리] {label} source별 라벨 분포:")
    for source_name, row in grouped.iterrows():
        pos = int(row.get(1, 0))
        neg = int(row.get(0, 0))
        print(f"  - {source_name}: size={pos+neg}, pos={pos}, neg={neg}")


def preprocess_to_splits(
    *,
    inputs: Sequence[Path],
    out_dir: Path,
    nrows: int | None,
    max_text_chars: int,
    min_text_chars: int,
    mask_leakage: bool,
    dedupe: bool,
    random_state: int,
    test_size: float,
    valid_size: float,
    fakepos_multiplier: float,
    linkedin_multiplier: float,
) -> dict[str, Path]:
    """
    정책 기반 전처리:
    - test는 RecruitmentScam ONLY
    - FakeJobPostings(all-positive), LinkedIn(all-negative)는 train augment only
    """
    existing = [p for p in inputs if p.exists()]
    if not existing:
        raise SystemExit("[전처리] 오류: 입력 CSV가 없습니다. training/data/raw/를 확인하세요.")

    raw_by_name = {p.stem: p for p in existing}
    if "RecruitmentScam" not in raw_by_name:
        raise SystemExit("[전처리] 오류: RecruitmentScam.csv가 필요합니다(평가용 test 생성).")

    # 1) load supervised base
    base = _load_recruitment_scam(raw_by_name["RecruitmentScam"], nrows=nrows)

    # 2) finalize 전체를 먼저 만들고(=text_hash 포함), 그 다음 split 합니다.
    # 이렇게 해야 동일/유사 텍스트가 train/test에 같이 들어가는 누수를 줄일 수 있습니다.
    base_all_out = _finalize_rows(
        base,
        max_text_chars=max_text_chars,
        min_text_chars=min_text_chars,
        mask_leakage=mask_leakage,
        dedupe=dedupe,
    )

    from sklearn.model_selection import train_test_split

    y = base_all_out["fraudulent"]
    base_train_out, base_test_out = train_test_split(
        base_all_out,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )

    base_valid_out = None
    if valid_size > 0:
        y2 = base_train_out["fraudulent"]
        base_train_out, base_valid_out = train_test_split(
            base_train_out,
            test_size=float(valid_size),
            random_state=int(random_state),
            stratify=y2,
        )

    # test는 반드시 pos/neg 둘 다 있어야 의미 있음
    test_pos = int((base_test_out["fraudulent"] == 1).sum())
    test_neg = int((base_test_out["fraudulent"] == 0).sum())
    if test_pos == 0 or test_neg == 0:
        raise SystemExit(
            f"[전처리] 오류: test.csv가 단일 클래스입니다(pos={test_pos}, neg={test_neg}). "
            "RecruitmentScam 라벨/분포를 확인하세요."
        )

    # split 후에도 교집합이 있으면 (dedupe off 등) 실패시켜 누수를 강제로 차단
    inter_base = set(base_train_out["text_hash"].astype(str)) & set(base_test_out["text_hash"].astype(str))
    if inter_base:
        raise SystemExit(
            f"[전처리] 오류: RecruitmentScam train/test 텍스트 중복(해시 교집합)={len(inter_base)}. "
            "중복 제거(dedupe)를 켜고 다시 실행하세요."
        )

    # 3) augment datasets (train only)
    aug_frames = [base_train_out]
    base_pos = int((base_train_out["fraudulent"] == 1).sum())
    base_neg = int((base_train_out["fraudulent"] == 0).sum())

    if "FakeJobPostings" in raw_by_name:
        fake = _load_fakejob_pos_only(raw_by_name["FakeJobPostings"], nrows=nrows)
        fake_out = _finalize_rows(
            fake,
            max_text_chars=max_text_chars,
            min_text_chars=min_text_chars,
            mask_leakage=mask_leakage,
            dedupe=dedupe,
        )
        # all-positive면 증강용으로만 샘플링
        if (fake_out["source"] == "FakeJobPostings_PosOnly").any():
            max_fake = int(round(base_pos * float(fakepos_multiplier)))
            if max_fake > 0:
                fake_sample = fake_out.sample(
                    n=min(max_fake, len(fake_out)),
                    random_state=int(random_state),
                    replace=False,
                )
                aug_frames.append(fake_sample)

    if "LinkedInPostings" in raw_by_name:
        li = _load_linkedin_neg_only(raw_by_name["LinkedInPostings"], nrows=nrows)
        li_out = _finalize_rows(
            li,
            max_text_chars=max_text_chars,
            min_text_chars=min_text_chars,
            mask_leakage=mask_leakage,
            dedupe=dedupe,
        )
        max_li = int(round(base_neg * float(linkedin_multiplier)))
        if max_li > 0:
            li_sample = li_out.sample(
                n=min(max_li, len(li_out)),
                random_state=int(random_state),
                replace=False,
            )
            aug_frames.append(li_sample)

    import pandas as pd  # type: ignore

    combined_train = pd.concat(aug_frames, axis=0, ignore_index=True)

    # split 중복/누수 점검: train/test 해시 교집합은 0이어야 함
    inter = set(combined_train["text_hash"].astype(str)) & set(base_test_out["text_hash"].astype(str))
    if inter:
        raise SystemExit(f"[전처리] 오류: train/test 텍스트 중복(해시 교집합)={len(inter)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    combined_train_path = out_dir / "combined_train.csv"
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"

    combined_train.to_csv(combined_train_path, index=False)
    combined_train.to_csv(train_path, index=False)
    base_test_out.to_csv(test_path, index=False)

    if base_valid_out is not None:
        valid_path = out_dir / "valid.csv"
        base_valid_out.to_csv(valid_path, index=False)

    _print_source_stats(combined_train, label="TRAIN(augment 포함)")
    _print_source_stats(base_test_out, label="TEST(RecruitmentScam only)")

    print(f"[전처리] train.csv 저장: {train_path}")
    print(f"[전처리] test.csv 저장: {test_path}")

    # 재현성용 preprocess signature 저장 (train/evaluate에서 metadata로 끌어올 수 있음)
    signature: dict[str, Any] = {
        "template": "v1_bracketed",
        "max_text_chars": int(max_text_chars),
        "min_text_chars": int(min_text_chars),
        "mask_leakage": bool(mask_leakage),
        "dedupe": bool(dedupe),
        "split_seed": int(random_state),
        "test_size": float(test_size),
        "valid_size": float(valid_size),
        "fakepos_multiplier": float(fakepos_multiplier),
        "linkedin_multiplier": float(linkedin_multiplier),
        "policy": {
            "test_sources": ["RecruitmentScam"],
            "augment_only_sources": ["FakeJobPostings_PosOnly", "LinkedIn_NegOnly"],
        },
        "outputs": {
            "combined_train_csv": str(combined_train_path),
            "train_csv": str(train_path),
            "test_csv": str(test_path),
            **({"valid_csv": str(out_dir / "valid.csv")} if base_valid_out is not None else {}),
        },
    }
    sig_path = out_dir / "preprocess_signature.json"
    sig_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[전처리] preprocess_signature 저장: {sig_path}")

    return {
        "combined_train_csv": combined_train_path,
        "train_csv": train_path,
        "test_csv": test_path,
        **({"valid_csv": out_dir / "valid.csv"} if base_valid_out is not None else {}),
    }


def make_dataset(
    *,
    inputs: Sequence[Path],
    out: Path,
    nrows: int | None = None,
    max_text_chars: int = 6000,
    min_text_chars: int = 50,
    mask_leakage: bool = True,
    dedupe: bool = True,
    random_state: int = 42,
    test_size: float = 0.2,
    valid_size: float = 0.0,
    fakepos_multiplier: float = 2.0,
    linkedin_multiplier: float = 3.0,
) -> int:
    # out는 combined_train.csv 경로로 취급하고, 나머지(train/test/valid)는 out.parent에 저장합니다.
    out_dir = out.parent
    preprocess_to_splits(
        inputs=inputs,
        out_dir=out_dir,
        nrows=nrows,
        max_text_chars=max_text_chars,
        min_text_chars=min_text_chars,
        mask_leakage=mask_leakage,
        dedupe=dedupe,
        random_state=random_state,
        test_size=test_size,
        valid_size=valid_size,
        fakepos_multiplier=fakepos_multiplier,
        linkedin_multiplier=linkedin_multiplier,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset preprocessing (unify schema).")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help=(
            "Input CSV paths. Default: training/data/raw/FakeJobPostings.csv, "
            "training/data/raw/LinkedInPostings.csv, training/data/raw/RecruitmentScam.csv (if present)."
        ),
    )
    parser.add_argument(
        "--out",
        default="training/data/processed/combined_train.csv",
        help="Output combined_train.csv path (default: training/data/processed/combined_train.csv).",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="(debug) Read only first N rows per input file.",
    )
    parser.add_argument("--max-text-chars", type=int, default=6000)
    parser.add_argument("--min-text-chars", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--valid-size", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--fakepos-multiplier", type=float, default=2.0)
    parser.add_argument("--linkedin-multiplier", type=float, default=3.0)
    # 유지: leakage/dedupe 옵션
    parser.add_argument(
        "--no-leakage-mask",
        action="store_true",
        help="Disable masking obvious label leakage keywords.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable exact duplicate removal via text_hash.",
    )

    args = parser.parse_args()
    if args.inputs is None or len(args.inputs) == 0:
        inputs = default_input_paths()
    else:
        inputs = [Path(p) for p in args.inputs]

    return make_dataset(
        inputs=inputs,
        out=Path(args.out),
        nrows=args.nrows,
        max_text_chars=int(args.max_text_chars),
        min_text_chars=int(args.min_text_chars),
        mask_leakage=not bool(args.no_leakage_mask),
        dedupe=not bool(args.no_dedupe),
        random_state=int(args.random_state),
        test_size=float(args.test_size),
        valid_size=float(args.valid_size),
        fakepos_multiplier=float(args.fakepos_multiplier),
        linkedin_multiplier=float(args.linkedin_multiplier),
    )


if __name__ == "__main__":
    raise SystemExit(main())
