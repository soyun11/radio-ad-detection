import os
import re
import csv
import argparse
import subprocess
from dataclasses import dataclass
from typing import List, Tuple

OUT_DIR = os.environ.get("OUT_DIR", os.path.join(os.getcwd(), "results"))

MIN_MATCH_SCORE = 8
MIN_MATCH_PERCENT = 40.0
MIN_MATCHED_DAYS = 2


@dataclass
class ClipResult:
    clip_file: str
    clip_start: float
    clip_end: float
    matched_days: int
    matched_ids: List[str]
    best_score: float
    best_percent: float

def run_panako_query(audio_path: str) -> str:
    cmd = [
        "panako", "query",
        "--audio", audio_path,
        "--max-matches", "5",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    return proc.stdout


def parse_panako_output(out: str) -> Tuple[int, List[str], float, float]:
    matched_ids = set()

    for m in re.finditer(r"Matches\s+(\d{8})\s+\(id\)\s+Filtered hits:\s+(\d+)", out):
        if int(m.group(2)) > 0:
            matched_ids.add(m.group(1))

    best_score = -1.0
    best_percent = -1.0

    for line in out.splitlines():
        if ";" not in line:
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 10:
            continue

        nums = []
        for p in parts:
            try:
                nums.append(float(p.replace("%", "")))
            except:
                pass

        for n in nums:
            if 0.0 <= n <= 100.0:
                best_percent = max(best_percent, n)
            elif n > best_score:
                best_score = max(best_score, n)

    return len(matched_ids), sorted(matched_ids), best_score, best_percent

def parse_clip_time(filename: str) -> Tuple[float, float]:
    """
    clip_002374_007122.mp3 → (2374, 7122) ms → seconds
    """
    m = re.search(r"_(\d+)_(\d+)\.mp3$", filename)
    if not m:
        raise ValueError(f"Invalid clip filename: {filename}")

    start_ms = int(m.group(1))
    end_ms = int(m.group(2))
    return start_ms / 1000.0, end_ms / 1000.0


def detect_clips(input_dir: str) -> List[ClipResult]:
    results: List[ClipResult] = []

    clips = sorted(f for f in os.listdir(input_dir) if f.endswith(".mp3"))
    print(f"[INFO] clips found: {len(clips)}")

    for idx, clip in enumerate(clips, 1):
        clip_path = os.path.join(input_dir, clip)
        clip_start, clip_end = parse_clip_time(clip)

        print(f"[{idx}/{len(clips)}] querying {clip}")

        out = run_panako_query(clip_path)
        matched_days, matched_ids, best_score, best_percent = parse_panako_output(out)

        if (
            matched_days >= MIN_MATCHED_DAYS
            and best_score >= MIN_MATCH_SCORE
            and best_percent >= MIN_MATCH_PERCENT
        ):
            results.append(
                ClipResult(
                    clip_file=clip,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    matched_days=matched_days,
                    matched_ids=matched_ids,
                    best_score=best_score,
                    best_percent=best_percent,
                )
            )

    return results

def save_csv(date: str, results: List[ClipResult]):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"panako_clips_{date}.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "date",
            "clip_file",
            "clip_start",
            "clip_end",
            "clip_sec",
            "matched_days",
            "matched_ids",
            "best_score",
            "best_percent",
        ])
        for r in results:
            w.writerow([
                date,
                r.clip_file,
                f"{r.clip_start:.2f}",
                f"{r.clip_end:.2f}",
                f"{(r.clip_end - r.clip_start):.2f}",
                r.matched_days,
                ",".join(r.matched_ids),
                f"{r.best_score:.2f}",
                f"{r.best_percent:.2f}",
            ])

    print(f"[DONE] saved → {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Panako ad detector (clip-based)")
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--input-dir", required=True, help="directory with mp3 clips")
    args = ap.parse_args()

    results = detect_clips(args.input_dir)
    print(f"[INFO] matched clips: {len(results)}")

    save_csv(args.date, results)


if __name__ == "__main__":
    main()

