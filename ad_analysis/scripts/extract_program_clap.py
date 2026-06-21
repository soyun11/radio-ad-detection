"""
extract_program_clap.py

[목적]
광고 블록 사이의 프로그램 구간(음악/DJ)을 추출해서 CLAP 임베딩 생성.
광고 직전 프로그램 분위기를 파악해서 맥락에 맞는 광고 생성에 활용.

흐름:
  광고 타임라인 → 광고 사이 구간 = 프로그램 구간
  → 프로그램 구간 클립 추출 → CLAP 임베딩
  → 광고 임베딩과 코사인 유사도 계산
  → 가장 유사한 광고 타입 Tags → heartlib 생성
"""

import os
import numpy as np
import pandas as pd
import laion_clap
import torch
import librosa
from pydub import AudioSegment
import warnings
warnings.filterwarnings('ignore')

CSV_PATH   = "ad_debug/results/noon/noon_v5_gap30_ad_detail.csv"
NOON_DIR   = "noon"
OUTPUT_DIR = "ad_analysis/clips/program"
OUTPUT_NPY = "ad_analysis/results/program_clap_embeddings.npy"
OUTPUT_CSV = "ad_analysis/results/program_clap_embeddings.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────
# 1. 프로그램 구간 추출
# 광고와 광고 사이 구간을 프로그램 구간으로 정의
# 최소 30초 이상인 구간만 사용 (너무 짧으면 의미없음)
# ──────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df[df["detected"] == True].sort_values(["date", "ad_start"]).reset_index(drop=True)

program_segments = []
for date, group in df.groupby("date"):
    group = group.sort_values("ad_start").reset_index(drop=True)
    for i in range(len(group) - 1):
        prog_start = group.loc[i, "ad_end"]
        prog_end   = group.loc[i+1, "ad_start"]
        duration   = prog_end - prog_start
        # 다음 광고 직전 구간만 사용 (30초 ~ 300초)
        if 30 <= duration <= 300:
            program_segments.append({
                "date": str(int(date)),
                "prog_start": prog_start,
                "prog_end": prog_end,
                "duration": duration,
                "next_ad_company": group.loc[i+1, "company"],
                "next_ad_tags": None  # 나중에 ad_features.csv에서 매핑
            })

prog_df = pd.DataFrame(program_segments)
print(f"총 {len(prog_df)}개 프로그램 구간 추출")

# ──────────────────────────────────────────
# 2. CLAP 모델 로드
# ──────────────────────────────────────────
print("LAION-CLAP 모델 로드 중...")
model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
model.load_ckpt()
model.eval()
print("모델 로드 완료!")

# ──────────────────────────────────────────
# 3. 프로그램 구간 클립 추출 + CLAP 임베딩
# ──────────────────────────────────────────
embeddings   = []
valid_indices = []

for i, row in prog_df.iterrows():
    date     = row["date"]
    mp3_path = os.path.join(NOON_DIR, date, "mp3", f"{date}.mp3")

    if not os.path.exists(mp3_path):
        continue

    clip_name = f"{date}_prog_{int(row['prog_start'])}_{int(row['prog_end'])}.mp3"
    clip_path = os.path.join(OUTPUT_DIR, clip_name)

    # 클립 추출 (없으면 자르기)
    if not os.path.exists(clip_path):
        try:
            audio = AudioSegment.from_mp3(mp3_path)
            clip  = audio[int(row["prog_start"]*1000):int(row["prog_end"]*1000)]
            clip.export(clip_path, format="mp3")
        except Exception as e:
            print(f"클립 추출 실패: {clip_name} - {e}")
            continue

    # CLAP 임베딩 추출
    try:
        audio, sr = librosa.load(clip_path, sr=48000, mono=True)
        # 최대 30초만 사용
        audio = audio[:48000*30]
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        with torch.no_grad():
            emb = model.get_audio_embedding_from_data(x=audio_tensor, use_tensor=True)
        embeddings.append(emb.squeeze().numpy())
        valid_indices.append(i)
        print(f"완료: {clip_name}")
    except Exception as e:
        print(f"임베딩 실패: {clip_name} - {e}")

# ──────────────────────────────────────────
# 4. 저장
# ──────────────────────────────────────────
emb_array = np.array(embeddings)
np.save(OUTPUT_NPY, emb_array)
print(f"\n임베딩 shape: {emb_array.shape}")

prog_valid = prog_df.loc[valid_indices].reset_index(drop=True)
emb_cols   = {f'clap_{i}': emb_array[:, i] for i in range(emb_array.shape[1])}
result_df  = pd.concat([prog_valid, pd.DataFrame(emb_cols)], axis=1)
result_df.to_csv(OUTPUT_CSV, index=False)
print(f"저장 완료: {OUTPUT_CSV}")
