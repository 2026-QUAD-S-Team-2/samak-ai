import csv
from pathlib import Path

import pytest


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_preprocess_test_excludes_augment_sources(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")  # training extra

    # Minimal RecruitmentScam (mixed labels)
    _write_csv(
        tmp_path / "raw" / "RecruitmentScam.csv",
        header=["title", "location", "description", "fraudulent"],
        rows=[
            ["t1", "US", "desc1", "t"],
            ["t2", "US", "desc2", "f"],
            ["t3", "US", "desc3", "t"],
            ["t4", "US", "desc4", "f"],
        ],
    )

    # FakeJobPostings (all-positive)
    _write_csv(
        tmp_path / "raw" / "FakeJobPostings.csv",
        header=["title", "description", "fraudulent"],
        rows=[
            ["fp1", "fakepos1", 1],
            ["fp2", "fakepos2", 1],
        ],
    )

    # LinkedIn (neg-only)
    _write_csv(
        tmp_path / "raw" / "LinkedInPostings.csv",
        header=["job_posting_url", "skills_desc", "title", "description", "location"],
        rows=[
            ["http://x", "skill", "li1", "normal1", "US"],
            ["http://y", "skill2", "li2", "normal2", "US"],
        ],
    )

    from training import preprocess

    out_dir = tmp_path / "processed"
    preprocess.preprocess_to_splits(
        inputs=[
            tmp_path / "raw" / "RecruitmentScam.csv",
            tmp_path / "raw" / "FakeJobPostings.csv",
            tmp_path / "raw" / "LinkedInPostings.csv",
        ],
        out_dir=out_dir,
        nrows=None,
        max_text_chars=2000,
        min_text_chars=1,
        mask_leakage=True,
        dedupe=True,
        random_state=42,
        test_size=0.5,
        valid_size=0.0,
        fakepos_multiplier=2.0,
        linkedin_multiplier=2.0,
    )

    test_df = pd.read_csv(out_dir / "test.csv")
    assert set(test_df["source"].unique()) == {"RecruitmentScam"}

    # Sanity: test must be mixed label for meaningful evaluation
    vc = test_df["fraudulent"].value_counts().to_dict()
    assert 1 in vc and 0 in vc

