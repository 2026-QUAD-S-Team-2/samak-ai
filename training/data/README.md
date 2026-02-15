# Datasets (gitignored)

학습 데이터셋 파일은 `training/data/` 아래에 둡니다.

- 기본적으로 데이터 파일은 git에 올리지 않기 위해 `training/data/*`가 ignore 됩니다.
- 안내용으로 이 `training/data/README.md`만 추적합니다.

구조:
- `training/data/raw/` (원본 다운로드 파일)
- `training/data/processed/` (전처리/통합 결과)

raw:
- `training/data/raw/FakeJobPostings.csv`
- `training/data/raw/RecruitmentScam.csv`
- `training/data/raw/LinkedInPostings.csv` (정상 데이터로 처리)

전처리/스플릿 생성:
```bash
python training/preprocess.py
```
