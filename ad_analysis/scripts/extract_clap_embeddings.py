"""
extract_clap_embeddings.py

[전체 목적]
LAION-CLAP 딥러닝 모델을 사용해서 광고 클립별 오디오 임베딩을 추출하는 스크립트.

[왜 CLAP 임베딩이 필요하냐?]
extract_ad_features.py의 librosa 피처는 BPM, 에너지 같은 수치를 직접 계산하는
"룰 기반" 방식이라 음악의 "분위기"나 "감성" 같은 추상적 특징을 잘 못 잡음.
예를 들어 BPM이 같아도 발라드와 재즈는 전혀 다른 느낌인데 librosa로는 구분이 안 됨.

CLAP(Contrastive Language-Audio Pretraining)은 오디오와 텍스트를 같은 임베딩 공간에
매핑하도록 학습된 딥러닝 모델로, 음악적 의미와 감성까지 포착할 수 있음.
→ 각 광고 클립을 512차원 벡터로 변환하면 비슷한 분위기의 광고끼리 가까운 위치에 놓임.

[LAION-CLAP vs HeartCLAP]
원래 heartlib의 HeartCLAP을 쓰려 했지만 아직 미출시 상태 (소스코드 없음).
→ LAION-CLAP이 HeartCLAP과 동일한 역할(오디오-텍스트 통합 임베딩)을 하므로 대안으로 사용.
→ HeartCLAP이 공개되면 이 스크립트를 HeartCLAP으로 교체 예정.

[사용 모델]
  - LAION-CLAP, HTSAT-tiny 버전
  - HTSAT-base는 체크포인트 아키텍처 불일치 오류로 사용 불가 (size mismatch)
  - CPU로 실행 (서버 CUDA 드라이버 버전 불일치로 GPU 사용 불가)

[입력 파일]
  - ad_features.csv : extract_ad_features.py 결과 (클립 경로 포함)

[출력 파일]
  - ad_clap_embeddings.npy : 307 x 512 임베딩 행렬 (numpy 바이너리, 빠른 로드용)
  - ad_clap_embeddings.csv : 임베딩 + 메타데이터 (company, date, tags 등)

[활용 방법]
  - 코사인 유사도로 라디오 프로그램 구간과 광고 간 유사도 측정
  - CLAP 임베딩 기반 K-Means / HDBSCAN 클러스터링으로 더 정교한 광고 분류
  - heartlib 광고 생성 후 생성된 광고와 주변 프로그램 구간의 유사도 평가
"""

import os
import numpy as np
import pandas as pd
import laion_clap
import torch
import librosa
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────

# 입력: extract_ad_features.py에서 생성한 피처 CSV (클립 경로 포함)
FEATURE_CSV   = "ad_analysis/results/ad_features.csv"

# 출력: 512차원 임베딩 행렬 (numpy 바이너리)
# shape: (광고 수, 512) → 예: (307, 512)
OUTPUT_NPY    = "ad_analysis/results/ad_clap_embeddings.npy"

# 출력: 임베딩 + 메타데이터 CSV
# 컬럼: date, company, ad_start, ad_end, tags, clip_path, clap_0 ~ clap_511
OUTPUT_CSV    = "ad_analysis/results/ad_clap_embeddings.csv"


# ──────────────────────────────────────────
# 1. LAION-CLAP 모델 로드
# ──────────────────────────────────────────

print("LAION-CLAP 모델 로드 중...")

# HTSAT-tiny 사용 (HTSAT-base는 체크포인트 아키텍처 불일치 오류 발생)
# enable_fusion=False: 오디오 전용 임베딩 (텍스트 fusion 비활성화)
model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')

# 첫 실행 시 HuggingFace에서 체크포인트 자동 다운로드 (~500MB)
model.load_ckpt()
model.eval()  # 추론 모드 (dropout 비활성화)
print("모델 로드 완료!")


# ──────────────────────────────────────────
# 2. 광고 클립 목록 로드
# ──────────────────────────────────────────

df = pd.read_csv(FEATURE_CSV)
print(f"총 {len(df)}개 광고 클립 처리 시작")


# ──────────────────────────────────────────
# 3. 클립별 임베딩 추출
# ──────────────────────────────────────────

embeddings = []   # 추출된 임베딩 저장 리스트
valid_indices = []  # 성공적으로 처리된 행 인덱스

for i, row in df.iterrows():
    clip_path = row['clip_path']

    if not os.path.exists(clip_path):
        print(f"파일 없음 스킵: {clip_path}")
        continue

    try:
        # CLAP 모델은 48kHz 모노 오디오를 입력으로 받음
        # sr=48000: CLAP 모델 요구 샘플링 레이트
        # mono=True: 스테레오 → 모노 변환
        audio, sr = librosa.load(clip_path, sr=48000, mono=True)

        # 최대 10초로 자르기
        # 이유: 광고는 보통 15~30초인데 너무 길면 메모리/시간 부담
        #       앞 10초가 광고의 핵심 특징을 담고 있다고 가정
        max_samples = 48000 * 10
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # 배치 형태로 변환: (samples,) → (1, samples)
        # CLAP은 배치 입력을 받으므로 차원 추가
        audio_tensor = torch.tensor(audio).unsqueeze(0)

        # 임베딩 추출 (512차원 벡터)
        # torch.no_grad(): 역전파 비활성화 → 메모리 절약, 속도 향상
        with torch.no_grad():
            embedding = model.get_audio_embedding_from_data(
                x=audio_tensor,
                use_tensor=True
            )

        # (1, 512) → (512,) 으로 squeeze 후 numpy 변환
        embeddings.append(embedding.squeeze().numpy())
        valid_indices.append(i)

        if len(embeddings) % 50 == 0:
            print(f"진행중: {len(embeddings)}/{len(df)}")

    except Exception as e:
        print(f"임베딩 추출 실패: {clip_path} - {e}")


# ──────────────────────────────────────────
# 4. 결과 저장
# ──────────────────────────────────────────

# (광고 수, 512) 형태의 numpy 배열로 변환
embeddings_array = np.array(embeddings)
print(f"\n임베딩 shape: {embeddings_array.shape}")  # 예: (307, 512)

# npy로 저장 (빠른 로드용)
# 나중에 np.load()로 빠르게 불러올 수 있음
np.save(OUTPUT_NPY, embeddings_array)
print(f"임베딩 저장: {OUTPUT_NPY}")

# 메타데이터 + 임베딩을 합쳐서 CSV로도 저장
# clap_0 ~ clap_511 컬럼으로 512차원 임베딩 값 저장
df_valid = df.loc[valid_indices].reset_index(drop=True)
emb_cols = {f'clap_{i}': embeddings_array[:, i] for i in range(embeddings_array.shape[1])}
df_emb = pd.concat([
    df_valid[['date', 'company', 'ad_start', 'ad_end', 'tags', 'clip_path']],
    pd.DataFrame(emb_cols)
], axis=1)
df_emb.to_csv(OUTPUT_CSV, index=False)
print(f"메타데이터 포함 CSV 저장: {OUTPUT_CSV}")
print("\n완료!")