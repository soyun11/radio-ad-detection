import argparse
import os
import numpy as np
import pandas as pd

BROADCASTER_DATES = {
    "noon": [
        "20260101", "20260102", "20260108", "20260109", "20260110",
        "20260111", "20260112", "20260113", "20260114", "20260115",
    ],
    "baechulsu": [
        "20241124", "20241125", "20241126",
        "20260101", "20260102", "20260103", "20260104",
        "20260105", "20260106", "20260107",
    ],
    "movie": [
        "20241120", "20241121", "20241122", "20241123", "20241124",
        "20241125", "20241126", "20241127", "20241128", "20241129",
    ],
}

AD_DETECT_THRESHOLD = 0.5  

def load_gt(date: str, gt_dir: str):
    for suffix in ["truth_block.csv", "truth_macro.csv", "truth_all_block.csv"]:
        path = os.path.join(gt_dir, f"{date}-{suffix}")
        if os.path.exists(path):
            break
    else:
        return [], []

    df = pd.read_csv(path)
    df["ad_start"] = pd.to_numeric(df["ad_start"], errors="coerce")
    df["ad_end"]   = pd.to_numeric(df["ad_end"],   errors="coerce")
    df = df.dropna(subset=["ad_start", "ad_end"])

    df = df[df["ad_end"] > df["ad_start"]]
    return list(zip(df["ad_start"], df["ad_end"])), df.to_dict("records")

def load_baseline_blocks(date: str, broadcaster: str, base_dir: str) -> list:
    path = os.path.join(base_dir, broadcaster, date, "transcript", f"{date}-blocks.csv")
    if not os.path.exists(path):
        return []

    df = pd.read_csv(path)
    if "block_type" not in df.columns:
        return []

    ad_rows = df[df["block_type"] == "AD"]
    blocks = []
    for _, r in ad_rows.iterrows():
        s, e = float(r["start"]), float(r["end"])
        if e > s:
            blocks.append({"start": s, "end": e})
    return sorted(blocks, key=lambda x: x["start"])

def overlap_ratio(a_start, a_end, b_start, b_end) -> float:
    dur = a_end - a_start
    if dur <= 0:
        return 0.0
    ov = max(0, min(a_end, b_end) - max(a_start, b_start))
    return ov / dur

def evaluate_ads(gt_intervals, gt_info, pred_blocks, threshold=0.5):
    ad_results = []
    matched_blocks = set()
    for i, (s, e) in enumerate(gt_intervals):
        best_ratio, best_block = 0.0, None
        for j, b in enumerate(pred_blocks):
            r = overlap_ratio(s, e, b["start"], b["end"])
            if r > best_ratio:
                best_ratio, best_block = r, j

        detected = best_ratio >= threshold
        if detected and best_block is not None:
            matched_blocks.add(best_block)

        ad_results.append({
            "ad_idx": i + 1, "ad_start": s, "ad_end": e,
            "company": gt_info[i].get("company", "") if gt_info else "",
            "overlap_ratio": round(best_ratio, 3), "detected": detected,
        })

    ad_tp  = sum(1 for r in ad_results if r["detected"])
    ad_fn  = sum(1 for r in ad_results if not r["detected"])
    ad_fp_blocks = sum(1 for j in range(len(pred_blocks)) if j not in matched_blocks)
    return ad_results, ad_tp, ad_fn, ad_fp_blocks


def evaluate_date(date, broadcaster, base_dir, gt_dir, log):
    log(f"\n{'='*65}\n📅 [{date}] {broadcaster} 베이스라인 평가\n{'='*65}")

    gt_intervals, gt_info = load_gt(date, gt_dir)
    if not gt_intervals:
        log(f"⚠️ {date}의 GT 파일을 찾을 수 없습니다.")
        return None

    pred_blocks = load_baseline_blocks(date, broadcaster, base_dir)

    log(f"  GT 아이템      : {len(gt_intervals)}개")
    log(f"  베이스라인 블록 : {len(pred_blocks)}개")

    if not pred_blocks:
        log("⚠️ 탐지된 베이스라인 블록이 없습니다.")
        return None

    max_time = max(max(e for _, e in gt_intervals), max(b["end"] for b in pred_blocks)) + 10
    N = int(max_time)
    gt_arr   = np.zeros(N, dtype=np.int8)
    pred_arr = np.zeros(N, dtype=np.int8)

    for s, e in gt_intervals:
        gt_arr[int(s):int(e)] = 1
    for b in pred_blocks:
        pred_arr[int(b["start"]):int(b["end"])] = 1

    tp = int(np.sum((gt_arr == 1) & (pred_arr == 1)))
    fp = int(np.sum((gt_arr == 0) & (pred_arr == 1)))
    fn = int(np.sum((gt_arr == 1) & (pred_arr == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    ad_results, ad_tp, ad_fn, ad_fp_blocks = evaluate_ads(
        gt_intervals, gt_info, pred_blocks, AD_DETECT_THRESHOLD
    )
    ad_precision = ad_tp / (ad_tp + ad_fp_blocks) if (ad_tp + ad_fp_blocks) > 0 else 0.0
    ad_recall    = ad_tp / (ad_tp + ad_fn)         if (ad_tp + ad_fn)         > 0 else 0.0
    ad_f1        = (2 * ad_precision * ad_recall / (ad_precision + ad_recall)) if (ad_precision + ad_recall) > 0 else 0.0

    log(f"\n[정답 아이템별 탐지율]")
    log(f"  {'#':>3}  {'시작':>7} {'끝':>7}  {'TP':>4} {'FN':>4}  {'탐지율':>6}  {'개별판정':>6}  회사")
    log("  " + "-" * 68)
    for i, (s, e) in enumerate(gt_intervals):
        ov   = int(np.sum(pred_arr[int(s):int(e)] == 1))
        miss = int(e) - int(s) - ov
        pct  = ov / (int(e) - int(s)) * 100 if (int(e) - int(s)) > 0 else 0
        log(f"  {i+1:>3}  {s:>7.1f} {e:>7.1f}  {ov:>4} {miss:>4}  {pct:>5.1f}% "
            f"{'✓' if pct >= 50 else '✗'}  "
            f"{'✅탐지' if ad_results[i]['detected'] else '❌누락'}  "
            f"{gt_info[i].get('company', '')}")

  
    log(f"\n[베이스라인 예측 블록별 정확도]")
    log(f"  {'#':>3}  {'시작':>7} {'끝':>7}  {'TP':>4} {'FP':>4}  {'정확도':>6}  판정")
    log("  " + "-" * 57)
    for i, b in enumerate(pred_blocks):
        tp_b = int(np.sum(gt_arr[int(b["start"]):int(b["end"])] == 1))
        dur  = int(b["end"]) - int(b["start"])
        fp_b = dur - tp_b
        acc  = tp_b / dur * 100 if dur > 0 else 0
        log(f"  {i+1:>3}  {b['start']:>7.1f} {b['end']:>7.1f}  {tp_b:>4} {fp_b:>4}  {acc:>5.1f}%  "
            f"{'✓ 블록' if acc >= 50 else '✗ FP'}")

    log(f"\n[{date} 블록 단위 결과]")
    log(f"  TP={tp}초  FP={fp}초  FN={fn}초")
    log(f"  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "ad_total":      len(gt_intervals),
        "ad_tp":         ad_tp,
        "ad_fn":         ad_fn,
        "ad_fp_blocks":  ad_fp_blocks,
        "ad_precision":  round(ad_precision, 4),
        "ad_recall":     round(ad_recall, 4),
        "ad_f1":         round(ad_f1, 4),
        "_ad_results":   ad_results,
        "_date":         date,
    }

def evaluate_broadcaster(broadcaster, dates, base_dir, gt_dir, out_dir, log):
    log(f"\n{'#'*65}")
    log(f"🎙️  [{broadcaster.upper()}] 베이스라인 평가")
    log(f"{'#'*65}")

    total = {"tp": 0, "fp": 0, "fn": 0,
             "ad_total": 0, "ad_tp": 0, "ad_fn": 0, "ad_fp_blocks": 0}
    date_results, all_ad_detail = [], []

    for date in dates:
        r = evaluate_date(date, broadcaster, base_dir, gt_dir, log)
        if r:
            _date = r.pop("_date", date)
            _ads  = r.pop("_ad_results", [])
            for k in total:
                total[k] += r[k]
            for ad in _ads:
                ad["date"] = _date
                ad["broadcaster"] = broadcaster
                all_ad_detail.append(ad)
            date_results.append({"broadcaster": broadcaster, "date": date, **r})

    if not date_results:
        log(f"\n⚠️ [{broadcaster}] 유효한 날짜가 없습니다.\n")
        return None, []

    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    ad_tp, ad_fp, ad_fn = total["ad_tp"], total["ad_fp_blocks"], total["ad_fn"]
    ad_precision = ad_tp / (ad_tp + ad_fp) if (ad_tp + ad_fp) > 0 else 0.0
    ad_recall    = ad_tp / (ad_tp + ad_fn) if (ad_tp + ad_fn) > 0 else 0.0
    ad_f1        = (2 * ad_precision * ad_recall / (ad_precision + ad_recall)) if (ad_precision + ad_recall) > 0 else 0.0

    log(f"\n{'='*65}")
    log(f"📊 [{broadcaster}] 전체 결과 ({len(date_results)}일)")
    log(f"{'='*65}")
    log(f"  [블록 단위] TP: {tp}초 | FP: {fp}초 | FN: {fn}초")
    log(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    log(f"  [아이템 단위] 탐지: {ad_tp} | 누락: {ad_fn} | 놓친블록(FP): {ad_fp}")
    log(f"  AD Precision: {ad_precision:.4f} | AD Recall: {ad_recall:.4f} | AD F1: {ad_f1:.4f}\n")


    tag = f"{broadcaster}_baseline"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(date_results).to_csv(
        os.path.join(out_dir, f"{tag}_by_date.csv"), index=False, encoding="utf-8-sig")
    if all_ad_detail:
        pd.DataFrame(all_ad_detail).to_csv(
            os.path.join(out_dir, f"{tag}_ad_detail.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "broadcaster":   broadcaster,
        "days":          len(date_results),
        "TP":            tp, "FP": fp, "FN": fn,
        "Precision":     round(precision, 4),
        "Recall":        round(recall, 4),
        "F1":            round(f1, 4),
        "ad_total":      total["ad_total"],
        "AD_TP":         ad_tp, "AD_FN": ad_fn, "AD_FP": ad_fp,
        "AD_Precision":  round(ad_precision, 4),
        "AD_Recall":     round(ad_recall, 4),
        "AD_F1":         round(ad_f1, 4),
    }
    return summary, date_results

def main():
    ap = argparse.ArgumentParser(description="베이스라인 광고 블록 탐지 평가 (방송 3사)")
    ap.add_argument("--base_dir",    default="")
    ap.add_argument("--broadcaster", default=None,
                    help="특정 방송사만 평가 (noon / baechulsu / movie). 생략 시 전체")
    ap.add_argument("--out_dir",     default="")
    args = ap.parse_args()

    base_dir = args.base_dir
    out_dir  = args.out_dir if args.out_dir else os.path.join(base_dir, "ad_evaluation", "results", "baseline")

    broadcasters = [args.broadcaster] if args.broadcaster else list(BROADCASTER_DATES.keys())

    report_lines = []
    def log(msg=""):
        print(msg)
        report_lines.append(str(msg) + "\n")

    log("=" * 65)
    log("🎯 베이스라인 광고 블록 탐지 평가 (방송 3사)")
    log(f"   base_dir : {base_dir}")
    log(f"   out_dir  : {out_dir}")
    log("=" * 65)

    all_summaries = []
    all_date_results = []

    for broadcaster in broadcasters:
        dates  = BROADCASTER_DATES[broadcaster]
        gt_dir = os.path.join(base_dir, "ad_evaluation", "ground_truth", broadcaster)

        summary, date_results = evaluate_broadcaster(
            broadcaster, dates, base_dir, gt_dir, out_dir, log
        )
        if summary:
            all_summaries.append(summary)
            all_date_results.extend(date_results)

    # 전체 합산 출력
    if len(all_summaries) > 1:
        total_tp  = sum(s["TP"]  for s in all_summaries)
        total_fp  = sum(s["FP"]  for s in all_summaries)
        total_fn  = sum(s["FN"]  for s in all_summaries)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        ad_tp = sum(s["AD_TP"] for s in all_summaries)
        ad_fp = sum(s["AD_FP"] for s in all_summaries)
        ad_fn = sum(s["AD_FN"] for s in all_summaries)
        ad_precision = ad_tp / (ad_tp + ad_fp) if (ad_tp + ad_fp) > 0 else 0.0
        ad_recall    = ad_tp / (ad_tp + ad_fn) if (ad_tp + ad_fn) > 0 else 0.0
        ad_f1        = (2 * ad_precision * ad_recall / (ad_precision + ad_recall)) if (ad_precision + ad_recall) > 0 else 0.0

        log(f"\n{'='*65}")
        log(f"🏆 전체 3사 합산 결과")
        log(f"{'='*65}")
        log(f"  [블록 단위] TP: {total_tp}초 | FP: {total_fp}초 | FN: {total_fn}초")
        log(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        log(f"  [아이템 단위] 탐지: {ad_tp} | 누락: {ad_fn} | 놓친블록(FP): {ad_fp}")
        log(f"  AD Precision: {ad_precision:.4f} | AD Recall: {ad_recall:.4f} | AD F1: {ad_f1:.4f}\n")

        all_summaries.append({
            "broadcaster": "ALL",
            "days": sum(s["days"] for s in all_summaries),
            "TP": total_tp, "FP": total_fp, "FN": total_fn,
            "Precision": round(precision, 4), "Recall": round(recall, 4), "F1": round(f1, 4),
            "ad_total": sum(s["ad_total"] for s in all_summaries),
            "AD_TP": ad_tp, "AD_FN": ad_fn, "AD_FP": ad_fp,
            "AD_Precision": round(ad_precision, 4),
            "AD_Recall": round(ad_recall, 4),
            "AD_F1": round(ad_f1, 4),
        })

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(all_summaries).to_csv(
        os.path.join(out_dir, "baseline_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(all_date_results).to_csv(
        os.path.join(out_dir, "baseline_all_by_date.csv"), index=False, encoding="utf-8-sig")

    report_path = os.path.join(out_dir, "baseline_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    log(f"💾 결과가 {out_dir} 에 저장되었습니다.")
    log(f"   - baseline_summary.csv")
    log(f"   - baseline_all_by_date.csv")
    log(f"   - {{broadcaster}}_baseline_by_date.csv")
    log(f"   - baseline_report.txt")


if __name__ == "__main__":
    main()