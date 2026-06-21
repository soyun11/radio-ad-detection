import os
import re
import json
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")

BASE_DIR = ""

BROADCASTER_CONFIG = {
    "baechulsu": {
        "name": "배철수의 음악캠프 (MBC)",
        "audio_dir":  f"{BASE_DIR}/baechulsu",
        "gt_dir":     f"{BASE_DIR}/ad_evaluation/ground_truth/baechulsu",
        "panako_dir": f"{BASE_DIR}/panako/baechulsu/results",
        "prev_date": {
            "20241125": "20241124",
            "20241126": "20241125",
            "20260102": "20260101",
            "20260103": "20260102",
            "20260104": "20260103",
            "20260105": "20260104",
            "20260106": "20260105",
            "20260107": "20260106",
        },
    },
    "movie": {
        "name": "박하선의 씨네타운 (SBS)",
        "audio_dir":  f"{BASE_DIR}/movie",
        "gt_dir":     f"{BASE_DIR}/ad_evaluation/ground_truth/movie",
        "panako_dir": f"{BASE_DIR}/panako/movie/results",
        "prev_date": {
            "20241121": "20241120",
            "20241122": "20241121",
            "20241123": "20241122",
            "20241124": "20241123",
            "20241125": "20241124",
            "20241126": "20241125",
            "20241127": "20241126",
            "20241128": "20241127",
            "20241129": "20241128",
        },
    },
    "noon": {
        "name": "이은지의 가요광장 (KBS)",
        "audio_dir":  f"{BASE_DIR}/noon",
        "gt_dir":     f"{BASE_DIR}/ad_evaluation/ground_truth/noon",
        "panako_dir": f"{BASE_DIR}/panako/noon/results",
        "prev_date": {
            "20260102": "20260101",
            "20260108": "20260102",
            "20260109": "20260108",
            "20260110": "20260109",
            "20260111": "20260110",
            "20260112": "20260111",
            "20260113": "20260112",
            "20260114": "20260113",
            "20260115": "20260114",
        },
    },
}

BASE_DIR = ""

BROADCASTER_CONFIG = {
    "baechulsu": {
        "name": "배철수의 음악캠프 (MBC)",
        "audio_dir":  f"{BASE_DIR}/baechulsu",
        "gt_dir":     f"{BASE_DIR}/ad_evaluation/ground_truth/baechulsu",
        "panako_dir": f"{BASE_DIR}/panako/baechulsu/results",

        "prev_date": {
            "20241125": "20241124",
            "20241126": "20241125",
            "20260102": "20260101",
            "20260103": "20260102",
            "20260104": "20260103",
            "20260105": "20260104",
            "20260106": "20260105",
            "20260107": "20260106",
        },
    },
    "movie": {
        "name": "박하선의 씨네타운 (SBS)",
        "audio_dir":  f"{BASE_DIR}/movie",
        "gt_dir":     f"{BASE_DIR}/ad_evaluation/ground_truth/movie",
        "panako_dir": f"{BASE_DIR}/panako/movie/results",
        "prev_date": {
            "20241121": "20241120",
            "20241122": "20241121",
            "20241123": "20241122",
            "20241124": "20241123",
            "20241125": "20241124",
            "20241126": "20241125",
            "20241127": "20241126",
            "20241128": "20241127",
            "20241129": "20241128",
        },
    },
    "noon": {
        "name": "이은지의 가요광장 (KBS)",
        "audio_dir":  f"{BASE_DIR}/noon",
        "gt_dir":     f"{BASE_DIR}/ad_evaluation/ground_truth/noon",
        "panako_dir": f"{BASE_DIR}/panako/noon/results",
        "prev_date": {
            "20260102": "20260101",
            "20260108": "20260102",
            "20260109": "20260108",
            "20260110": "20260109",
            "20260111": "20260110",
            "20260112": "20260111",
            "20260113": "20260112",
            "20260114": "20260113",
            "20260115": "20260114",
        },
    },
}

OUTPUT_DIR      = f"{BASE_DIR}/ad_evaluation/results/individual_ads_v4_new"
CLIP_DURATION   = 20    
OVERLAP_THRESH  = 0.5   
SIM_THRESHOLD   = 0.75  
GAP_THRESHOLD   = 30    

@dataclass
class DetectedAd:

    gt_start:    float   
    gt_end:      float  
    gt_company:  str     
    gt_product:  str     
    overlap:     float   
    transcript:  str = ""        
    pred_company: str = ""       
    pred_product: str = ""      
    company_sim:  float = 0.0   
    product_sim:  float = 0.0  
    company_correct: bool = False  
    product_correct: bool = False 

class IndividualAdEvaluatorV4New:

    def __init__(self, broadcaster: str):
        self.broadcaster      = broadcaster
        self.config           = BROADCASTER_CONFIG[broadcaster]
        self._embedding_model = None
        self._llm_client      = None
        self._inference_cache = {}

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            print("🔄 Embedding 모델 로드 중...")
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embedding_model

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = OpenAI()
        return self._llm_client

    def load_pred_blocks(self, date: str) -> List[Dict]:
        prev_date = self.config["prev_date"].get(date)
        if not prev_date:
            print(f"  ⚠️ prev_date 없음: {date}")
            return []

        filename = f"{date}-{prev_date}-compare-ad-result.csv"
        candidates = [
            os.path.join(self.config["panako_dir"], filename),
            os.path.join(self.config["panako_dir"], f"{date}_vs_{prev_date}", filename),
            os.path.join(self.config["panako_dir"], date, filename),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if not path:
            print(f"  ⚠️ ad-result.csv 없음: {filename}")
            return []

        df = pd.read_csv(path)

        def extract_start(qpath: str) -> float:
            m = re.search(r'clip_\d+_(\d+)\.mp3', str(qpath))
            return int(m.group(1)) / 1000.0 if m else 0.0

        segs = [
            {"query_start": extract_start(row["Query Path"]),
             "query_end":   extract_start(row["Query Path"]) + CLIP_DURATION}
            for _, row in df.iterrows()
        ]

        if not segs:
            return []
        segs = sorted(segs, key=lambda x: x["query_start"])
        merged = [{"start": segs[0]["query_start"], "end": segs[0]["query_end"]}]
        for s in segs[1:]:
            last = merged[-1]
            if s["query_start"] <= last["end"] + GAP_THRESHOLD:
                last["end"] = max(last["end"], s["query_end"])
            else:
                merged.append({"start": s["query_start"], "end": s["query_end"]})

        print(f"  → {len(df)}개 클립 → {len(merged)}개 블록")
        return merged

    def load_ground_truth(self, date: str) -> List[Dict]:
        gt_path = os.path.join(self.config["gt_dir"], f"{date}-truth_block.csv")
        if not os.path.exists(gt_path):
            print(f"  ⚠️ GT 없음: {gt_path}")
            return []
        df = pd.read_csv(gt_path)
        df["ad_start"] = pd.to_numeric(df["ad_start"], errors="coerce")
        df["ad_end"]   = pd.to_numeric(df["ad_end"],   errors="coerce")
        df = df.dropna(subset=["ad_start", "ad_end"])
        df = df[df["ad_end"] > df["ad_start"]]
        return [
            {"start":   float(row["ad_start"]),
             "end":     float(row["ad_end"]),
             "company": str(row.get("company", "")).strip(),
             "product": str(row.get("product", "")).strip()}
            for _, row in df.iterrows()
        ]

    def evaluate_detection(
        self, gt_ads: List[Dict], pred_blocks: List[Dict]
    ) -> Tuple[List[DetectedAd], List[Dict], int]:
        
        detected_ads = []
        missed_ads   = []
        matched_blocks = set()

        for gt in gt_ads:
            gt_dur = gt["end"] - gt["start"]
            if gt_dur <= 0:
                continue

            best_ratio, best_block_idx = 0.0, -1
            for j, b in enumerate(pred_blocks):
                overlap = max(0, min(gt["end"], b["end"]) - max(gt["start"], b["start"]))
                ratio   = overlap / gt_dur
                if ratio > best_ratio:
                    best_ratio, best_block_idx = ratio, j

            if best_ratio >= OVERLAP_THRESH and best_block_idx >= 0:
                matched_blocks.add(best_block_idx)
                detected_ads.append(DetectedAd(
                    gt_start=gt["start"], gt_end=gt["end"],
                    gt_company=gt["company"], gt_product=gt["product"],
                    overlap=round(best_ratio, 3),
                ))
            else:
                missed_ads.append(gt)

        fp_blocks = sum(1 for j in range(len(pred_blocks)) if j not in matched_blocks)
        return detected_ads, missed_ads, fp_blocks


    def load_inference_csv(self, date: str) -> pd.DataFrame:
 
        if date in self._inference_cache:
            return self._inference_cache[date]
        path = os.path.join(
            self.config["audio_dir"], date, "transcript",
            f"{date}-inference_result_ratio.csv"
        )
        if not os.path.exists(path):
            print(f"  ⚠️ inference 파일 없음: {path}")
            self._inference_cache[date] = pd.DataFrame()
            return pd.DataFrame()
        df = pd.read_csv(path)
        self._inference_cache[date] = df
        return df

    def get_transcript(self, date: str, start: float, end: float) -> str:

        df = self.load_inference_csv(date)
        if df.empty:
            return ""
        texts = []
        for _, row in df.iterrows():
            if float(row["Stop Time"]) > start and float(row["Start Time"]) < end:
                t = row.get("Transcript", "")
                if pd.notna(t) and str(t).strip():
                    texts.append(str(t).strip())
        return " ".join(texts).strip()

    def extract_entities(self, transcript: str) -> Tuple[str, str]:

        prompt = f"""다음은 라디오 방송에서 오디오 핑거프린팅으로 탐지된 구간의 음성을 텍스트로 변환한 것입니다.
이 구간의 회사명(광고주 또는 방송사/프로그램)과 제품명(광고 제품 또는 프로그램/코너명)을 추출해주세요.

[추출 규칙]
1. 일반 광고: 회사명 = 광고주, 제품명 = 광고 제품/서비스
   예) "현대해상 하이카 자동차보험" → {{"company": "현대해상", "product": "하이카"}}
   예) "신한은행 퇴직연금" → {{"company": "신한은행", "product": "퇴직연금"}}
   예) "LG전자 힐링미 MX9" → {{"company": "LG전자", "product": "힐링미 MX9"}}

2. 방송사 자체 광고/프로모션: 회사명 = 방송사, 제품명 = 프로그램/서비스명
   예) "MBC FM4U 날씨와 생활" → {{"company": "MBC", "product": "날씨와생활"}}
   예) "SBS 고릴라 앱 광고" → {{"company": "SBS", "product": "고릴라광고"}}
   예) "KBS플러스 채널 소개" → {{"company": "KBS플러스", "product": "KBS플러스"}}

3. 협찬/경품 안내: 회사명 = 프로그램명, 제품명 = 협찬경품안내
   예) "씨네타운 협찬 경품 안내입니다" → {{"company": "씨네타운", "product": "협찬경품안내"}}
   예) "가요광장 협찬 경품" → {{"company": "가요광장", "product": "협찬경품안내"}}

4. 프로그램 오프닝/BGM/코너: 회사명 = 프로그램명, 제품명 = 오프닝곡/BGM/코너명
   예) "배철수의 음악캠프 오프닝" → {{"company": "배철수의라디오", "product": "오프닝곡"}}
   예) "박하선의 씨네타운 시작" → {{"company": "박하선의씨네타운", "product": "오프닝곡"}}

5. 캠페인/공익광고: 회사명 = 주관 기관, 제품명 = 캠페인
   예) "환경부 환경책임보험 캠페인" → {{"company": "환경부와 환경책임보험사업단", "product": "캠페인"}}

광고 텍스트:
{transcript}

JSON 형식으로만 답변해주세요:
{{"company": "회사명", "product": "제품명"}}"""

        for attempt in range(5):
            try:
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content.strip()
                m = re.search(r'\{[^}]+\}', content)
                if m:
                    result = json.loads(m.group())
                    return result.get("company", ""), result.get("product", "")
                return "", ""
            except Exception as e:
                err = str(e)
                if "rate_limit_exceeded" in err or "429" in err:
                    # 에러 메시지에서 대기시간 파싱, 없으면 지수 백오프
                    wait = 2 ** attempt
                    m_wait = re.search(r'try again in (\d+(?:\.\d+)?)s', err)
                    if m_wait:
                        wait = float(m_wait.group(1)) + 0.5
                    print(f"    ⏳ Rate limit, {wait:.1f}초 후 재시도 ({attempt+1}/5)...")
                    time.sleep(wait)
                else:
                    print(f"    ⚠️ LLM 오류: {e}")
                    break
        return "", ""


    def compute_similarity(self, pred: str, gt: str) -> float:

        if not pred or not gt:
            return 0.0
        if pred.strip() == gt.strip():
            return 1.0
        if pred in gt or gt in pred:
            return 0.9
        return float(cosine_similarity(
            self.embedding_model.encode([pred]),
            self.embedding_model.encode([gt])
        )[0, 0])


    def evaluate(self, date: str, log) -> Dict:
        log(f"\n{'='*65}")
        log(f"📅 [{date}] 개별 광고 평가 (v4_new)")
        log(f"   방송사: {self.config['name']}")
        log(f"{'='*65}")


        log("\n[Step 1] ad-result.csv → pred_blocks")
        pred_blocks = self.load_pred_blocks(date)
        if not pred_blocks:
            log("  ⚠️ pred_blocks 없음, 스킵")
            return {}

        gt_ads = self.load_ground_truth(date)
        if not gt_ads:
            log("  ⚠️ GT 없음, 스킵")
            return {}
        log(f"  GT 광고: {len(gt_ads)}개 | pred_blocks: {len(pred_blocks)}개")

        log(f"\n[Step 2] 탐지 평가 (overlap >= {OVERLAP_THRESH})")
        detected_ads, missed_ads, fp_blocks = self.evaluate_detection(gt_ads, pred_blocks)
        ad_tp = len(detected_ads)
        ad_fn = len(missed_ads)
        ad_fp = fp_blocks

        ad_prec = ad_tp / (ad_tp + ad_fp) if (ad_tp + ad_fp) > 0 else 0
        ad_rec  = ad_tp / (ad_tp + ad_fn) if (ad_tp + ad_fn) > 0 else 0
        ad_f1   = 2*ad_prec*ad_rec / (ad_prec+ad_rec) if (ad_prec+ad_rec) > 0 else 0

        log(f"  TP={ad_tp} FN={ad_fn} FP(미매칭블록)={ad_fp}")
        log(f"  Detection P={ad_prec:.4f} R={ad_rec:.4f} F1={ad_f1:.4f}")

        log(f"\n[Step 3] 탐지된 광고 엔티티 추출 ({len(detected_ads)}개)")
        for i, ad in enumerate(detected_ads):
            ad.transcript   = self.get_transcript(date, ad.gt_start, ad.gt_end)
            ad.pred_company, ad.pred_product = self.extract_entities(ad.transcript)
            log(f"  [{i+1:3d}] {ad.gt_start:.0f}~{ad.gt_end:.0f}초 "
                f"| 예측: {ad.pred_company} / {ad.pred_product} "
                f"| GT: {ad.gt_company} / {ad.gt_product}")

        log(f"\n[Step 4] 엔티티 F1 평가 (sim >= {SIM_THRESHOLD}, company OR product)")
        entity_tp = entity_fp = 0
        company_tp = company_fp = product_tp = product_fp = 0

        for ad in detected_ads:
            ad.company_sim = self.compute_similarity(ad.pred_company, ad.gt_company)
            ad.product_sim = self.compute_similarity(ad.pred_product, ad.gt_product)
            ad.company_correct = ad.company_sim >= SIM_THRESHOLD
            ad.product_correct = ad.product_sim >= SIM_THRESHOLD

          
            if ad.company_correct: company_tp += 1
            else:                  company_fp += 1
            if ad.product_correct: product_tp += 1
            else:                  product_fp += 1

           
            if ad.company_correct or ad.product_correct:
                entity_tp += 1
            else:
                entity_fp += 1

   
        entity_fn = len(missed_ads)

     
        e_prec = entity_tp / (entity_tp + entity_fp) if (entity_tp + entity_fp) > 0 else 0
        e_rec  = entity_tp / (entity_tp + entity_fn) if (entity_tp + entity_fn) > 0 else 0
        e_f1   = 2*e_prec*e_rec / (e_prec+e_rec) if (e_prec+e_rec) > 0 else 0

        c_prec = company_tp / (company_tp + company_fp) if (company_tp + company_fp) > 0 else 0
        c_rec  = company_tp / (company_tp + entity_fn)  if (company_tp + entity_fn)  > 0 else 0
        c_f1   = 2*c_prec*c_rec / (c_prec+c_rec) if (c_prec+c_rec) > 0 else 0

        p_prec = product_tp / (product_tp + product_fp) if (product_tp + product_fp) > 0 else 0
        p_rec  = product_tp / (product_tp + entity_fn)  if (product_tp + entity_fn)  > 0 else 0
        p_f1   = 2*p_prec*p_rec / (p_prec+p_rec) if (p_prec+p_rec) > 0 else 0

     
        total_detected = len(detected_ads)
        company_acc = company_tp / total_detected if total_detected > 0 else 0
        product_acc = product_tp / total_detected if total_detected > 0 else 0
        entity_acc  = entity_tp  / total_detected if total_detected > 0 else 0

        log(f"  [OR 기준] TP={entity_tp} FP={entity_fp} FN={entity_fn}")
        log(f"    Entity   P={e_prec:.4f} R={e_rec:.4f} F1={e_f1:.4f} | Acc={entity_acc:.4f}")
        log(f"  [개별 참고]")
        log(f"    Company  P={c_prec:.4f} R={c_rec:.4f} F1={c_f1:.4f} | Acc={company_acc:.4f}")
        log(f"    Product  P={p_prec:.4f} R={p_rec:.4f} F1={p_f1:.4f} | Acc={product_acc:.4f}")

        log(f"\n[{date} 결과]")
        log(f"  GT={len(gt_ads)} | Detection TP={ad_tp} FP={ad_fp} FN={ad_fn}")
        log(f"  Detection  P={ad_prec:.4f} R={ad_rec:.4f} F1={ad_f1:.4f}")
        log(f"  Entity(OR) P={e_prec:.4f} R={e_rec:.4f} F1={e_f1:.4f}")
        log(f"  Company    P={c_prec:.4f} R={c_rec:.4f} F1={c_f1:.4f}")
        log(f"  Product    P={p_prec:.4f} R={p_rec:.4f} F1={p_f1:.4f}")

        return {
            "date": date, "broadcaster": self.broadcaster,
            "gt_total":    len(gt_ads),
            "pred_blocks": len(pred_blocks),
            
            "detection_tp":        ad_tp,
            "detection_fp":        ad_fp,
            "detection_fn":        ad_fn,
            "detection_precision": round(ad_prec, 4),
            "detection_recall":    round(ad_rec,  4),
            "detection_f1":        round(ad_f1,   4),
           
            "entity_tp":        entity_tp,
            "entity_fp":        entity_fp,
            "entity_fn":        entity_fn,
            "entity_precision": round(e_prec, 4),
            "entity_recall":    round(e_rec,  4),
            "entity_f1":        round(e_f1,   4),
            "entity_acc":       round(entity_acc, 4),
          
            "company_tp":        company_tp,
            "company_fp":        company_fp,
            "company_precision": round(c_prec, 4),
            "company_recall":    round(c_rec,  4),
            "company_f1":        round(c_f1,   4),
            "company_acc":       round(company_acc, 4),
          
            "product_tp":        product_tp,
            "product_fp":        product_fp,
            "product_precision": round(p_prec, 4),
            "product_recall":    round(p_rec,  4),
            "product_f1":        round(p_f1,   4),
            "product_acc":       round(product_acc, 4),
            "_detected_ads": detected_ads,
        }

    def save_results(self, result: Dict, out_dir: str):
        if not result:
            return
        os.makedirs(out_dir, exist_ok=True)
        date = result["date"]

        rows = []
        for ad in result.get("_detected_ads", []):
            rows.append({
                "date":            date,
                "broadcaster":     self.broadcaster,
                "gt_start":        ad.gt_start,
                "gt_end":          ad.gt_end,
                "overlap":         ad.overlap,
                "gt_company":      ad.gt_company,
                "gt_product":      ad.gt_product,
                "pred_company":    ad.pred_company,
                "pred_product":    ad.pred_product,
                "company_sim":     round(ad.company_sim, 4),
                "product_sim":     round(ad.product_sim, 4),
                "company_correct": ad.company_correct,
                "product_correct": ad.product_correct,
            })
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, f"{self.broadcaster}_{date}_matches.csv"),
                index=False, encoding="utf-8-sig"
            )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broadcaster", default="all",
                    choices=["baechulsu", "movie", "noon", "all"])
    ap.add_argument("--date", default=None)
    args = ap.parse_args()

    broadcasters = (["baechulsu", "movie", "noon"]
                    if args.broadcaster == "all" else [args.broadcaster])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(str(msg) + "\n")

    log("=" * 65)
    log("🎯 개별 광고 탐지 + 엔티티 추출 평가 (v4_new)")
    log("   탐지: ad-result.csv → merge_segments → evaluate_ads 방식")
    log("   엔티티: 탐지된 GT 광고 구간 inference_result_ratio.csv → GPT-4o")
    log("=" * 65)

    all_results = []

    for broadcaster in broadcasters:
        log(f"\n{'#'*65}")
        log(f"# {BROADCASTER_CONFIG[broadcaster]['name']}")
        log(f"{'#'*65}")

        evaluator = IndividualAdEvaluatorV4New(broadcaster=broadcaster)
        dates = ([args.date] if args.date
                 else list(BROADCASTER_CONFIG[broadcaster]["prev_date"].keys()))

        b_total = dict(gt=0, tp=0, fp=0, fn=0,
                       etp=0, efp=0, efn=0,
                       ctp=0, cfp=0, ptp=0, pfp=0)

        for date in dates:
            try:
                result = evaluator.evaluate(date, log)
                if not result:
                    continue
                evaluator.save_results(result, OUTPUT_DIR)

                b_total["gt"]  += result["gt_total"]
                b_total["tp"]  += result["detection_tp"]
                b_total["fp"]  += result["detection_fp"]
                b_total["fn"]  += result["detection_fn"]
                b_total["etp"] += result["entity_tp"]
                b_total["efp"] += result["entity_fp"]
                b_total["efn"] += result["entity_fn"]
                b_total["ctp"] += result["company_tp"]
                b_total["cfp"] += result["company_fp"]
                b_total["ptp"] += result["product_tp"]
                b_total["pfp"] += result["product_fp"]

                all_results.append({k: v for k, v in result.items()
                                    if not k.startswith("_")})
            except Exception as e:
                log(f"  ❌ [{date}] 오류: {e}")
                import traceback; traceback.print_exc()

        tp, fp, fn = b_total["tp"], b_total["fp"], b_total["fn"]
        p  = tp/(tp+fp) if (tp+fp)>0 else 0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0

        etp, efp, efn = b_total["etp"], b_total["efp"], b_total["efn"]
        ep = etp/(etp+efp) if (etp+efp)>0 else 0
        er = etp/(etp+efn) if (etp+efn)>0 else 0
        ef = 2*ep*er/(ep+er) if (ep+er)>0 else 0

        ctp, cfp = b_total["ctp"], b_total["cfp"]
        ptp, pfp = b_total["ptp"], b_total["pfp"]
        cp = ctp/(ctp+cfp) if (ctp+cfp)>0 else 0
        cr = ctp/(ctp+efn) if (ctp+efn)>0 else 0
        cf = 2*cp*cr/(cp+cr) if (cp+cr)>0 else 0
        pp = ptp/(ptp+pfp) if (ptp+pfp)>0 else 0
        pr = ptp/(ptp+efn) if (ptp+efn)>0 else 0
        pf = 2*pp*pr/(pp+pr) if (pp+pr)>0 else 0

        log(f"\n{'='*65}")
        log(f"📊 [{broadcaster}] 소계")
        log(f"  GT={b_total['gt']} TP={tp} FP={fp} FN={fn}")
        log(f"  Detection    P={p:.4f} R={r:.4f} F1={f1:.4f}")
        log(f"  Entity(OR)   P={ep:.4f} R={er:.4f} F1={ef:.4f}")
        log(f"  Company      P={cp:.4f} R={cr:.4f} F1={cf:.4f}")
        log(f"  Product      P={pp:.4f} R={pr:.4f} F1={pf:.4f}")

    if all_results:
        df = pd.DataFrame(all_results)

        log(f"\n{'='*65}")
        log("🏆 전체 3사 합산")
        log(f"{'='*65}")

        for bc in df["broadcaster"].unique():
            sub = df[df["broadcaster"] == bc]
            tp  = sub["detection_tp"].sum()
            fp  = sub["detection_fp"].sum()
            fn  = sub["detection_fn"].sum()
            p   = tp/(tp+fp) if (tp+fp)>0 else 0
            r   = tp/(tp+fn) if (tp+fn)>0 else 0
            f1  = 2*p*r/(p+r) if (p+r)>0 else 0

            etp = sub["entity_tp"].sum()
            efp = sub["entity_fp"].sum()
            efn = sub["entity_fn"].sum()
            ep  = etp/(etp+efp) if (etp+efp)>0 else 0
            er  = etp/(etp+efn) if (etp+efn)>0 else 0
            ef  = 2*ep*er/(ep+er) if (ep+er)>0 else 0

            ctp = sub["company_tp"].sum()
            cfp = sub["company_fp"].sum()
            ptp = sub["product_tp"].sum()
            pfp = sub["product_fp"].sum()
            cp  = ctp/(ctp+cfp) if (ctp+cfp)>0 else 0
            cr  = ctp/(ctp+efn) if (ctp+efn)>0 else 0
            cf  = 2*cp*cr/(cp+cr) if (cp+cr)>0 else 0
            pp  = ptp/(ptp+pfp) if (ptp+pfp)>0 else 0
            pr  = ptp/(ptp+efn) if (ptp+efn)>0 else 0
            pf  = 2*pp*pr/(pp+pr) if (pp+pr)>0 else 0

            log(f"  [{bc}]")
            log(f"    Detection    P:{p:.4f} R:{r:.4f} F1:{f1:.4f}")
            log(f"    Entity(OR)   P:{ep:.4f} R:{er:.4f} F1:{ef:.4f}")
            log(f"    Company      P:{cp:.4f} R:{cr:.4f} F1:{cf:.4f}")
            log(f"    Product      P:{pp:.4f} R:{pr:.4f} F1:{pf:.4f}")

        df.to_csv(os.path.join(OUTPUT_DIR, "all_results_summary.csv"),
                  index=False, encoding="utf-8-sig")
        with open(os.path.join(OUTPUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
            f.writelines(report_lines)

        log(f"\n💾 결과 저장: {OUTPUT_DIR}")

    log(f"\n{'='*65}")
    log("✅ v4_new 평가 완료!")
    log(f"{'='*65}")


if __name__ == "__main__":
    main()