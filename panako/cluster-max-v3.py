import re
import pandas as pd
import argparse
import os

SCORE_THRESHOLD = 100
OPENING_FILTER  = 0
GAP_THRESHOLD   = 30
CLIP_DURATION   = 20
CROSS_MATCH_THRESHOLD = 3600


def extract_query_start(query_path: str) -> float:
    match = re.search(r"clip_\d+_(\d+)\.mp3", str(query_path))
    return int(match.group(1)) / 1000.0 if match else 0.0


def is_overlapping(a_start, a_stop, b_start, b_stop):
    return not (a_stop <= b_start or a_start >= b_stop)


def find_non_overlapping_scores(sorted_cluster, cluster_id):
    non_overlapping = []
    used_intervals  = [] 
    for row in sorted_cluster:
        q_start = row["_query_start"]
        q_stop  = row["_query_stop"]
        m_start = row["Match Start"]

        overlapped = any(
            is_overlapping(q_start, q_stop, qs, qe)
            and m_start == ms
            for qs, qe, ms in used_intervals
        )

        if not overlapped:
            r = row.copy()
            r["Cluster ID"] = cluster_id
            non_overlapping.append(r)
            used_intervals.append((q_start, q_stop, m_start))

    return non_overlapping


def find_high_scores_in_clusters(data, gap_threshold):
    data = data.sort_values(by="_query_start").reset_index(drop=True)

    clusters = []
    current  = []

    for _, row in data.iterrows():
        r = row.to_dict()
        if not current:
            current.append(r)
            continue
        last = current[-1]
        if r["_query_start"] <= last["_query_stop"] + gap_threshold:
            current.append(r)
        else:
            clusters.append(current)
            current = [r]
    if current:
        clusters.append(current)

    cluster_info    = []
    all_non_overlap = []

    for cid, cluster in enumerate(clusters, start=1):
        c_start = min(r["_query_start"] for r in cluster)
        c_stop  = max(r["_query_stop"]  for r in cluster)

        cluster_info.append({
            "Cluster ID"    : cid,
            "Cluster Start" : round(c_start, 2),
            "Cluster Stop"  : round(c_stop,  2),
            "Cluster Size"  : len(cluster),
            "Duration"      : round(c_stop - c_start, 2),
        })

        sorted_cluster = sorted(cluster, key=lambda x: x["Match Score"], reverse=True)
        kept = find_non_overlapping_scores(sorted_cluster, cid)
        all_non_overlap.extend(kept)

    return all_non_overlap, cluster_info


def main(input_file, score_threshold, opening_filter, gap_threshold):
    data = pd.read_csv(input_file)
    print(f"\n📂 입력: {input_file}")
    print(f"   원본 매칭 수: {len(data)}개")
    print(f"   설정: score_threshold={score_threshold}, "
          f"opening_filter={opening_filter}, gap_threshold={gap_threshold}")
    print(f"   NMS: Match Start 동일할 때만 중복으로 판단")

    required = ["Query Path", "Match Start", "Match Stop", "Match Score"]
    for c in required:
        if c not in data.columns:
            raise ValueError(f"Missing column: {c}")

    data["Match Start"] = pd.to_numeric(data["Match Start"])
    data["Match Stop"]  = pd.to_numeric(data["Match Stop"])
    data["Match Score"] = pd.to_numeric(data["Match Score"])

    data["_query_start"] = data["Query Path"].apply(extract_query_start)
    data["_query_stop"]  = data["_query_start"] + CLIP_DURATION

    before = len(data)
    data = data[data["Match Score"] >= score_threshold]
    print(f"   Score < {score_threshold} 제거: {before - len(data)}개 → {len(data)}개 남음")

    before = len(data)
    data["_time_diff"] = abs(data["_query_start"] - data["Match Start"])
    data = data[data["_time_diff"] <= CROSS_MATCH_THRESHOLD]
    print(f"   교차매칭 필터 제거: {before - len(data)}개 → {len(data)}개 남음")

    if opening_filter > 0:
        before = len(data)
        data = data[data["_query_start"] >= opening_filter]
        print(f"   Opening 필터 제거: {before - len(data)}개 → {len(data)}개 남음")

    if len(data) == 0:
        print("⚠️ 남은 매칭이 없습니다.")
        out = os.path.splitext(input_file)[0] + "-ad-result.csv"
        pd.DataFrame(columns=["Cluster ID", "Match Start", "Match Stop",
                               "Duration", "Match Score", "Query Path"]).to_csv(out, index=False)
        return

    data["Duration"] = data["Match Stop"] - data["Match Start"]

    non_overlapping, cluster_info = find_high_scores_in_clusters(data, gap_threshold)
    print(f"   클러스터(광고 블록) 수: {len(cluster_info)}개")
    print(f"   최종 광고 매칭: {len(non_overlapping)}개")

    print(f"\n   [클러스터 상세 - Query 시간 기준]")
    for ci in cluster_info:
        print(f"   - 클러스터 {ci['Cluster ID']}: "
              f"{ci['Cluster Start']:.1f}~{ci['Cluster Stop']:.1f}초 "
              f"(duration={ci['Duration']:.1f}초, 매칭={ci['Cluster Size']}개)")

    output_base       = os.path.splitext(input_file)[0]
    ad_result_file    = output_base + "-ad-result.csv"
    cluster_info_file = output_base + "-cluster-info.csv"

    result = pd.DataFrame(non_overlapping)
    result = result.sort_values(by="_query_start")
    result["Duration"] = round(result["Match Stop"] - result["Match Start"], 2)

    save_cols = ["Cluster ID", "Match Start", "Match Stop", "Duration",
                 "Match Score", "Query Path"]
    result[save_cols].to_csv(ad_result_file, index=False)

    pd.DataFrame(cluster_info).to_csv(cluster_info_file, index=False)

    print(f"\n💾 광고 결과  : {ad_result_file}")
    print(f"💾 클러스터   : {cluster_info_file}")
    print("\n✅ 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cluster-max v3: Match Start 동일할 때만 중복 판단"
    )
    parser.add_argument("input_file",        type=str)
    parser.add_argument("--score_threshold", type=int, default=SCORE_THRESHOLD)
    parser.add_argument("--opening_filter",  type=int, default=OPENING_FILTER)
    parser.add_argument("--gap_threshold",   type=int, default=GAP_THRESHOLD)
    args = parser.parse_args()

    main(args.input_file, args.score_threshold, args.opening_filter, args.gap_threshold)