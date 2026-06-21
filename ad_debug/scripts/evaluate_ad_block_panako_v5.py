import argparse
import os
import re
import numpy as np
import pandas as pd

CLIP_DURATION = 20
AD_DETECT_THRESHOLD = 0.5


PREV_DATE = {

    "20260102": "20260101",
    "20260108": "20260102",
    "20260109": "20260108",
    "20260110": "20260109",
    "20260111": "20260110",
    "20260112": "20260111",
    "20260113": "20260112",
    "20260114": "20260113",
    "20260115": "20260114",


    "20241125": "20241124",
    "20241126": "20241125",
    "20260102": "20260101",
    "20260103": "20260102",
    "20260104": "20260103",
    "20260105": "20260104",
    "20260106": "20260105",
    "20260107": "20260106",
    

    "20241121": "20241120",
    "20241122": "20241121",
    "20241123": "20241122",
    "20241124": "20241123",
    "20241125": "20241124",
    "20241126": "20241125",
    "20241127": "20241126",
    "20241128": "20241127",
    "20241129": "20241128",
    
}

def extract_query_time(query_path) -> float:
    match = re.search(r"clip_\d+_(\d+)\.mp3", str(query_path))
    return int(match.group(1)) / 1000.0 if match else 0.0

def load_panako_segments(date: str, result_dir: str) -> list:
    prev = PREV_DATE.get(date)
    if not prev: return []
    filename = f"{date}-{prev}-compare-ad-result.csv"
    candidates = [
        os.path.join(result_dir, filename),
        os.path.join(result_dir, f"{date}_vs_{prev}", filename),
        os.path.join(result_dir, date, filename),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path: return []
    
    df = pd.read_csv(path)
    return [{"query_start": extract_query_time(r["Query Path"]), 
             "query_end": extract_query_time(r["Query Path"]) + CLIP_DURATION} 
            for _, r in df.iterrows()]

def merge_segments(segs: list, gap: float = 30) -> list:
    if not segs: return []
    segs = sorted(segs, key=lambda x: x["query_start"])
    merged = [{"start": segs[0]["query_start"], "end": segs[0]["query_end"]}]
    for s in segs[1:]:
        last = merged[-1]
        if s["query_start"] <= last["end"] + gap:
            last["end"] = max(last["end"], s["query_end"])
        else:
            merged.append({"start": s["query_start"], "end": s["query_end"]})
    return merged

def load_gt(date: str, gt_dir: str):

    for suffix in ["truth_block.csv", "truth_macro.csv", "truth_all_block.csv"]:
        path = os.path.join(gt_dir, f"{date}-{suffix}")
        if os.path.exists(path): break
    else: return [], []

    df = pd.read_csv(path)
    df["ad_start"] = pd.to_numeric(df["ad_start"], errors="coerce")
    df["ad_end"]   = pd.to_numeric(df["ad_end"],   errors="coerce")
    df = df.dropna(subset=["ad_start", "ad_end"])
    return list(zip(df["ad_start"], df["ad_end"])), df.to_dict("records")

def overlap_ratio(a_start, a_end, b_start, b_end) -> float:
    dur = a_end - a_start
    if dur <= 0: return 0.0
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
    return ad_results, sum(1 for r in ad_results if r["detected"]), sum(1 for r in ad_results if not r["detected"]), sum(1 for j in range(len(pred_blocks)) if j not in matched_blocks)

def evaluate_date(date, result_dir, gt_dir, log, gap=30):
    log(f"\n{'='*65}\n📅 [{date}] 평가 (통합 블록 기준 - v5)\n{'='*65}")

    gt_intervals, gt_info = load_gt(date, gt_dir)
    if not gt_intervals: 
        log(f"⚠️ {date}의 GT 파일을 찾을 수 없습니다.")
        return None

    segs = load_panako_segments(date, result_dir)
    pred_blocks = merge_segments(segs, gap)

    log(f"  GT 아이템   : {len(gt_intervals)}개")
    log(f"  Panako 블록 : {len(pred_blocks)}개 (gap={gap}s)")

    if not pred_blocks:
        log("⚠️ 탐지된 Panako 블록이 없습니다.")
        return None

    max_time = max(max(e for _, e in gt_intervals), max(b["end"] for b in pred_blocks)) + 10
    N = int(max_time)
    gt_arr, pred_arr = np.zeros(N, dtype=np.int8), np.zeros(N, dtype=np.int8)

    for s, e in gt_intervals: gt_arr[int(s):int(e)] = 1
    for b in pred_blocks: pred_arr[int(b["start"]):int(b["end"])] = 1

    tp = int(np.sum((gt_arr == 1) & (pred_arr == 1)))
    fp = int(np.sum((gt_arr == 0) & (pred_arr == 1)))
    fn = int(np.sum((gt_arr == 1) & (pred_arr == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    ad_results, ad_tp, ad_fn, ad_fp_blocks = evaluate_ads(gt_intervals, gt_info, pred_blocks, AD_DETECT_THRESHOLD)
    ad_precision = ad_tp / (ad_tp + ad_fp_blocks) if (ad_tp + ad_fp_blocks) > 0 else 0.0
    ad_recall    = ad_tp / (ad_tp + ad_fn) if (ad_tp + ad_fn) > 0 else 0.0
    ad_f1        = (2 * ad_precision * ad_recall / (ad_precision + ad_recall)) if (ad_precision + ad_recall) > 0 else 0.0

    log(f"\n[정답 아이템별 탐지율]")
    log(f"  {'#':>3}  {'시작':>7} {'끝':>7}  {'TP':>4} {'FN':>4}  {'탐지율':>6}  {'개별판정':>6}  회사")
    log("  " + "-" * 68)
    for i, (s, e) in enumerate(gt_intervals):
        ov, miss = int(np.sum(pred_arr[int(s):int(e)] == 1)), int(e) - int(s) - int(np.sum(pred_arr[int(s):int(e)] == 1))
        pct = ov / (int(e) - int(s)) * 100 if (int(e) - int(s)) > 0 else 0
        log(f"  {i+1:>3}  {s:>7.1f} {e:>7.1f}  {ov:>4} {miss:>4}  {pct:>5.1f}% {'✓' if pct>=50 else '✗'}  {'✅탐지' if ad_results[i]['detected'] else '❌누락'}  {gt_info[i].get('company', '')}")

    log(f"\n[Panako 예측 블록별 정확도]")
    log(f"  {'#':>3}  {'시작':>7} {'끝':>7}  {'TP':>4} {'FP':>4}  {'정확도':>6}  판정")
    log("  " + "-" * 57)
    for i, b in enumerate(pred_blocks):
        tp_b = int(np.sum(gt_arr[int(b["start"]):int(b["end"])] == 1))
        dur, fp_b = int(b["end"]) - int(b["start"]), int(b["end"]) - int(b["start"]) - int(np.sum(gt_arr[int(b["start"]):int(b["end"])] == 1))
        acc = tp_b / dur * 100 if dur > 0 else 0
        log(f"  {i+1:>3}  {b['start']:>7.1f} {b['end']:>7.1f}  {tp_b:>4} {fp_b:>4}  {acc:>5.1f}%  {'✓ 블록' if acc>=50 else '✗ FP'}")

    log(f"\n[{date} 블록 단위 결과]")
    log(f"  TP={tp}초  FP={fp}초  FN={fn}초")
    log(f"  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")

    return {
        "tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "ad_total": len(gt_intervals), "ad_tp": ad_tp, "ad_fn": ad_fn, "ad_fp_blocks": ad_fp_blocks,
        "ad_precision": round(ad_precision, 4), "ad_recall": round(ad_recall, 4), "ad_f1": round(ad_f1, 4),
        "_ad_results": ad_results, "_date": date
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", required=True)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--gap", type=int, default=30)
    args = ap.parse_args()

    out_dir = args.out_dir if args.out_dir else f"../results/{args.program}"
    os.makedirs(out_dir, exist_ok=True)
    
    total = {"tp": 0, "fp": 0, "fn": 0, "ad_total": 0, "ad_tp": 0, "ad_fn": 0, "ad_fp_blocks": 0}
    date_results, all_ad_detail = [], []
    
    report_lines = []
    def log(msg=""): 
        print(msg)
        report_lines.append(str(msg) + "\n")

    log("=" * 65)
    log(f"🎯 [{args.program}] Panako 통합 블록 탐지 평가 v5")
    log(f"   (ignore 로직 제거됨 / GAP = {args.gap}s)")
    log("=" * 65)

    for date in args.dates:
        r = evaluate_date(date, args.result_dir, args.gt_dir, log, gap=args.gap)
        if r:
            _date, _ads = r.pop("_date", date), r.pop("_ad_results", [])
            for k in total: total[k] += r[k]
            for ad in _ads:
                ad["date"] = _date
                all_ad_detail.append(ad)
            date_results.append({"date": date, **r})

    if not date_results: return

    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    ad_tp, ad_fp, ad_fn = total["ad_tp"], total["ad_fp_blocks"], total["ad_fn"]
    ad_precision = ad_tp / (ad_tp + ad_fp) if (ad_tp + ad_fp) > 0 else 0.0
    ad_recall    = ad_tp / (ad_tp + ad_fn) if (ad_tp + ad_fn) > 0 else 0.0
    ad_f1        = (2 * ad_precision * ad_recall / (ad_precision + ad_recall)) if (ad_precision + ad_recall) > 0 else 0.0

    log(f"\n{'='*65}\n📊 전체 결과 ({len(date_results)}일)\n{'='*65}")
    log(f"  [블록 단위] TP: {tp}초 | FP: {fp}초 | FN: {fn}초")
    log(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    log(f"  [아이템 단위] 탐지: {ad_tp} | 누락: {ad_fn} | 놓친블록(FP): {ad_fp}")
    log(f"  AD Precision: {ad_precision:.4f} | AD Recall: {ad_recall:.4f} | AD F1: {ad_f1:.4f}\n")

    tag = f"{args.program}_v5_gap{args.gap}"
    pd.DataFrame(date_results).to_csv(os.path.join(out_dir, f"{tag}_by_date.csv"), index=False, encoding="utf-8-sig")
    
    summary = {
        "program": args.program, "gap": args.gap, "days": len(date_results),
        "TP": tp, "FP": fp, "FN": fn, "Precision": round(precision, 4), "Recall": round(recall, 4), "F1": round(f1, 4),
        "ad_total": total["ad_total"], "AD_TP": ad_tp, "AD_FN": ad_fn, "AD_FP": ad_fp,
        "AD_Precision": round(ad_precision, 4), "AD_Recall": round(ad_recall, 4), "AD_F1": round(ad_f1, 4),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, f"{tag}_summary.csv"), index=False, encoding="utf-8-sig")
    
    if all_ad_detail:
        pd.DataFrame(all_ad_detail).to_csv(os.path.join(out_dir, f"{tag}_ad_detail.csv"), index=False, encoding="utf-8-sig")
    
    report_path = os.path.join(out_dir, f"{tag}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
        
    log(f"💾 결과가 {out_dir} 폴더에 정상적으로 저장되었습니다.")

if __name__ == "__main__":
    main()