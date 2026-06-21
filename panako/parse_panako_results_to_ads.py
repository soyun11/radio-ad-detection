import csv
import os
import re
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set

MIN_MATCHED_DAYS = 3
MIN_SECONDS_MATCH = 0.6
MIN_MATCH_SCORE = 200

MERGE_GAP_SEC = 6.0
MIN_AD_DURATION = 15.0


@dataclass
class ClipMatch:
    clip_path: str
    clip_start: float
    clip_end: float
    matched_days: Set[str]
    best_score: float
    best_seconds_match: float


@dataclass
class AdBlock:
    start: float
    end: float
    duration: float
    clip_count: int
    matched_days: int
    confidence: float


def parse_clip_time(clip_path: str) -> (float, float):
    """
    clip_000132_000396.mp3 → (132, 396)
    """
    m = re.search(r"clip_(\d+)_(\d+)", os.path.basename(clip_path))
    if not m:
        raise ValueError(f"Invalid clip filename: {clip_path}")
    return float(m.group(1)), float(m.group(2))


def load_and_group_by_clip(csv_path: str) -> Dict[str, ClipMatch]:
    clips = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["Query Path"]].append(r)

    for clip_path, entries in grouped.items():
        clip_start, clip_end = parse_clip_time(clip_path)

        matched_days = set()
        best_score = 0
        best_seconds = 0.0

        for e in entries:
            matched_days.add(e["Match ID"])
            best_score = max(best_score, float(e["Match Score"]))
            best_seconds = max(best_seconds, float(e["Seconds with Match (%)"]))

        clips[clip_path] = ClipMatch(
            clip_path=clip_path,
            clip_start=clip_start,
            clip_end=clip_end,
            matched_days=matched_days,
            best_score=best_score,
            best_seconds_match=best_seconds
        )

    return clips


def filter_ad_clips(clips: Dict[str, ClipMatch]) -> List[ClipMatch]:
    ads = []
    for c in clips.values():
        if (
            len(c.matched_days) >= MIN_MATCHED_DAYS
            and c.best_seconds_match >= MIN_SECONDS_MATCH
            and c.best_score >= MIN_MATCH_SCORE
        ):
            ads.append(c)

    return sorted(ads, key=lambda x: x.clip_start)


def merge_clips_to_blocks(clips: List[ClipMatch]) -> List[AdBlock]:
    if not clips:
        return []

    blocks = []
    cur_start = clips[0].clip_start
    cur_end = clips[0].clip_end
    days = set(clips[0].matched_days)
    count = 1

    for c in clips[1:]:
        if c.clip_start <= cur_end + MERGE_GAP_SEC:
            cur_end = max(cur_end, c.clip_end)
            days |= c.matched_days
            count += 1
        else:
            duration = cur_end - cur_start
            if duration >= MIN_AD_DURATION:
                blocks.append(
                    AdBlock(
                        start=cur_start,
                        end=cur_end,
                        duration=duration,
                        clip_count=count,
                        matched_days=len(days),
                        confidence=min(1.0, len(days) / 7.0),
                    )
                )
            cur_start = c.clip_start
            cur_end = c.clip_end
            days = set(c.matched_days)
            count = 1

    duration = cur_end - cur_start
    if duration >= MIN_AD_DURATION:
        blocks.append(
            AdBlock(
                start=cur_start,
                end=cur_end,
                duration=duration,
                clip_count=count,
                matched_days=len(days),
                confidence=min(1.0, len(days) / 7.0),
            )
        )

    return blocks


def save_blocks(date: str, blocks: List[AdBlock]):
    out = f"{date}-ad-blocks.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "date", "start_sec", "end_sec", "duration_sec",
            "clip_count", "matched_days", "confidence"
        ])
        for b in blocks:
            w.writerow([
                date,
                f"{b.start:.2f}",
                f"{b.end:.2f}",
                f"{b.duration:.2f}",
                b.clip_count,
                b.matched_days,
                f"{b.confidence:.2f}"
            ])
    print(f"[DONE] saved → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--input", required=True, help="Panako results CSV")
    args = ap.parse_args()

    clips = load_and_group_by_clip(args.input)
    ad_clips = filter_ad_clips(clips)
    blocks = merge_clips_to_blocks(ad_clips)

    print(f"[INFO] ad clips: {len(ad_clips)}")
    print(f"[INFO] ad blocks: {len(blocks)}")

    save_blocks(args.date, blocks)


if __name__ == "__main__":
    main()

