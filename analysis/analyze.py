import csv
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "nc_30games.csv"
OUTPUT_DIR = ROOT / "analysis"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"
PLOT_PATH = OUTPUT_DIR / "scatter.png"
EXPECTED_COLUMNS = ["date", "opponent", "hits", "runs", "url"]


def load_and_validate():
    with INPUT.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"CSV columns must be {EXPECTED_COLUMNS}; found {reader.fieldnames}"
            )
        rows = list(reader)

    if len(rows) != 30:
        raise ValueError(f"Expected 30 game rows; found {len(rows)}")

    seen_urls = set()
    for line_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"Unexpected extra column on CSV line {line_number}")
        if any(row[column] is None or not row[column].strip() for column in EXPECTED_COLUMNS):
            raise ValueError(f"Missing value on CSV line {line_number}")
        try:
            datetime.strptime(row["date"], "%Y.%m.%d")
        except ValueError as error:
            raise ValueError(f"Invalid date on CSV line {line_number}") from error
        if not row["url"].startswith("https://"):
            raise ValueError(f"Invalid URL on CSV line {line_number}")
        if row["url"] in seen_urls:
            raise ValueError(f"Duplicate game URL on CSV line {line_number}")
        seen_urls.add(row["url"])
        for column in ("hits", "runs"):
            try:
                value = int(row[column])
            except ValueError as error:
                raise ValueError(
                    f"{column} must be an integer on CSV line {line_number}"
                ) from error
            if value < 0:
                raise ValueError(
                    f"{column} must not be negative on CSV line {line_number}"
                )
            row[column] = value

    return rows


def main():
    rows = load_and_validate()
    hits = [row["hits"] for row in rows]
    runs = [row["runs"] for row in rows]
    count = len(rows)

    hits_mean = statistics.mean(hits)
    runs_mean = statistics.mean(runs)
    hits_std = statistics.stdev(hits)
    runs_std = statistics.stdev(runs)
    hits_median = statistics.median(hits)
    runs_median = statistics.median(runs)

    covariance = sum(
        (hit - hits_mean) * (run - runs_mean)
        for hit, run in zip(hits, runs)
    ) / (count - 1)
    correlation = covariance / (hits_std * runs_std)
    slope = covariance / (hits_std**2)
    intercept = runs_mean - slope * hits_mean

    hits_population_std = statistics.pstdev(hits)
    runs_population_std = statistics.pstdev(runs)
    outliers = []
    for row in rows:
        hits_z = (row["hits"] - hits_mean) / hits_population_std
        runs_z = (row["runs"] - runs_mean) / runs_population_std
        if abs(hits_z) > 3 or abs(runs_z) > 3:
            outliers.append((row, hits_z, runs_z))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    plt.scatter(hits, runs, alpha=0.8, label="Games")
    line_x = [min(hits), max(hits)]
    line_y = [slope * value + intercept for value in line_x]
    plt.plot(line_x, line_y, color="crimson", label="Linear trend line")
    plt.title("2026 NC Dinos: Hits and Runs (30 games)")
    plt.xlabel("Hits (count)")
    plt.ylabel("Runs (runs)")
    plt.xticks(range(min(hits), max(hits) + 1))
    plt.yticks(range(min(runs), max(runs) + 1, 2))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160)
    plt.close()

    outlier_text = (
        "\n".join(
            f"- {row['date']} vs {row['opponent']}: "
            f"안타 z={hits_z:.3f}, 득점 z={runs_z:.3f}"
            for row, hits_z, runs_z in outliers
        )
        if outliers
        else "- 없음"
    )
    summary = f"""# 2026 NC 다이노스 안타 수와 득점 분석

## 데이터 및 품질 점검

- 입력: `data/nc_30games.csv` (원본은 수정하지 않음)
- 데이터 행: {count}경기 (헤더 제외)
- 열: `date`, `opponent`, `hits`, `runs`, `url`
- 자료형: 날짜(문자열 `YYYY.MM.DD`), 상대팀(문자열), 안타·득점(정수), URL(문자열)
- 결측치: 0건
- 중복 경기 URL: 0건
- 안타 또는 득점의 음수·정수 형식 오류: 0건
- 분석은 모든 검증을 통과한 데이터에 대해서만 수행했습니다.

## 기술통계

| 변수 | 평균 | 중앙값 | 표본 표준편차 | 범위 |
|---|---:|---:|---:|---:|
| 안타 수 (개) | {hits_mean:.3f} | {hits_median:.1f} | {hits_std:.3f} | {min(hits)}–{max(hits)} |
| 득점 (점) | {runs_mean:.3f} | {runs_median:.1f} | {runs_std:.3f} | {min(runs)}–{max(runs)} |

## 상관관계와 산점도

- 피어슨 상관계수: **r = {correlation:.3f}**
- 회귀 추세선: 득점 = {slope:.3f} × 안타 수 {intercept:+.3f}
- 그래프: [`scatter.png`](./scatter.png)

이 30경기에서는 안타 수와 득점 사이에 양의 선형 상관관계가 관찰됩니다(r = {correlation:.3f}). 즉, 이 표본에서는 안타 수가 많은 경기일수록 득점도 높은 경향이 있습니다. 다만 상관관계만으로 안타가 득점 증가의 원인이라고 결론 내릴 수는 없습니다.

## z점수 이상치 점검

각 변수의 모집단 표준편차를 사용해 z점수를 계산했으며, 안타 또는 득점의 z점수 절댓값이 3을 초과하는 경기를 확인했습니다.

- 해당 관측치: **{len(outliers)}경기**
{outlier_text}

## 해석의 범위

분석 대상은 시즌 전체에서 무작위로 추출한 표본이 아니라 날짜순으로 선택된 30경기입니다. 따라서 이 결과를 2026 NC 다이노스 정규시즌 전체로 일반화할 수 없습니다.
"""
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(f"Analyzed {count} validated games.")
    print(f"Pearson r = {correlation:.6f}; outlier games = {len(outliers)}")
    print(f"Wrote {SUMMARY_PATH} and {PLOT_PATH}")


if __name__ == "__main__":
    main()
