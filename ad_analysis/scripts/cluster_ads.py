"""
cluster_ads.py

[전체 목적]
extract_ad_features.py에서 뽑은 오디오 피처(ad_features.csv)를 기반으로
광고를 3가지 유형으로 자동 분류하는 스크립트.

[왜 클러스터링을 하냐?]
heartlib으로 광고를 생성할 때 "어떤 타입의 광고를 만들어야 하는지" 알아야 함.
예를 들어 라디오 프로그램 직전에 흐르던 음악이 잔잔한 발라드였다면,
나레이션 위주의 조용한 광고를 생성하는 게 맥락상 자연스러움.
→ 광고를 타입별로 분류해두면 맥락에 맞는 타입을 골라서 생성할 수 있음.

[분류 기준 - 회의에서 논의된 3가지 유형]
  1. Narration     : 나레이션 위주. voice_ratio 높고 에너지 낮음.
                     예) 공공기관 광고, 방송국 자체 광고 (KBS쿨FM, 경기주택도시공사)
  2. CM_Song       : CM송 위주. BPM 높고 에너지 높고 voice_ratio 낮음.
                     예) 쇼핑몰, 식품 광고 (이은지의가요광장, 리퍼브)
  3. BGM_Narration : 배경음악 + 나레이션 혼합. 중간 수준.
                     예) 서비스/금융 광고 (기프트인포, 경리나라)

[사용 피처 - 17차원 벡터]
  - voice_ratio    : 나레이션 비율 (SRT 기반)
  - bpm            : 템포
  - rms_energy     : 에너지/음량
  - spectral_centroid : 음색 밝기
  - mfcc_0 ~ mfcc_12  : 음색 특징 13개

[알고리즘]
  1. StandardScaler로 피처 정규화 (단위가 다른 피처들을 동일 스케일로)
  2. K-Means (k=3)으로 클러스터링
  3. 클러스터별 평균 voice_ratio, BPM으로 유형 이름 자동 부여
  4. PCA로 17차원 → 2차원 압축 후 시각화

[입력 파일]
  - ad_features.csv : extract_ad_features.py 결과

[출력 파일]
  - ad_clusters.csv     : 광고별 클러스터 레이블(ad_type) 추가된 결과
  - ad_clusters_pca.png : PCA 2D 시각화 이미지

[한계]
  - K-Means는 클러스터 수(k=3)를 미리 지정해야 함
  - 클러스터 결과가 실제로 의미있는지 사람이 직접 대표 클립을 들어보고 검증 필요
  - librosa 피처 기반이라 "감성/분위기" 같은 추상적 특징은 반영 어려움
    → 더 정교한 분류가 필요하면 CLAP 임베딩(extract_clap_embeddings.py) 활용
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # 서버 환경에서 GUI 없이 이미지 저장
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────

# 입력: extract_ad_features.py에서 생성한 피처 CSV
FEATURE_CSV = "ad_analysis/results/ad_features.csv"

# 출력: 클러스터 레이블(ad_type)이 추가된 CSV
OUTPUT_CSV  = "ad_analysis/results/ad_clusters.csv"

# 출력: PCA 2D 시각화 이미지
PLOT_PATH   = "ad_analysis/results/ad_clusters_pca.png"


# ──────────────────────────────────────────
# 1. 데이터 로드 및 피처 선택
# ──────────────────────────────────────────

df = pd.read_csv(FEATURE_CSV)
print(f"총 {len(df)}개 광고 로드 완료")

# 클러스터링에 사용할 피처 컬럼 선택
# voice_ratio + bpm + rms_energy + spectral_centroid + mfcc 13개 = 총 17차원
feature_cols = (
    ['voice_ratio', 'bpm', 'rms_energy', 'spectral_centroid'] +
    [f'mfcc_{i}' for i in range(13)]
)

X = df[feature_cols].copy()


# ──────────────────────────────────────────
# 2. 정규화 (StandardScaler)
# ──────────────────────────────────────────

# 피처마다 단위/스케일이 달라서 정규화 필수
# 예: bpm은 100~160 범위, rms_energy는 0.05~0.18 범위
# → 정규화 안 하면 BPM 같은 큰 값이 클러스터링을 지배함
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ──────────────────────────────────────────
# 3. K-Means 클러스터링 (k=3)
# ──────────────────────────────────────────

# n_clusters=3: 회의에서 논의한 3가지 광고 유형 기준
# random_state=42: 재현 가능한 결과를 위한 시드 고정
# n_init=10: 초기값 10번 시도 후 가장 좋은 결과 선택 (안정성 향상)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)


# ──────────────────────────────────────────
# 4. 클러스터별 특징 분석 → 유형 이름 자동 부여
# ──────────────────────────────────────────

# 클러스터별 평균 피처 확인
cluster_stats = df.groupby('cluster')[['voice_ratio', 'bpm', 'rms_energy']].mean()
print("\n=== 클러스터별 평균 피처 ===")
print(cluster_stats.round(3))

# 클러스터에 유형 이름 자동 부여
# 규칙:
#   voice_ratio가 가장 높은 클러스터 → Narration (나레이션 위주)
#   voice_ratio 평균 이하이면서 bpm이 가장 높은 클러스터 → CM_Song (CM송 위주)
#   나머지 → BGM_Narration (배경음악 + 나레이션 혼합)
vr  = cluster_stats['voice_ratio']
bpm = cluster_stats['bpm']

narration_cluster = vr.idxmax()
cmsong_cluster    = bpm[vr < vr.mean()].idxmax() if (vr < vr.mean()).any() else bpm.idxmax()
bgm_cluster       = [c for c in [0, 1, 2] if c not in [narration_cluster, cmsong_cluster]][0]

label_map = {
    narration_cluster: "Narration",
    cmsong_cluster:    "CM_Song",
    bgm_cluster:       "BGM_Narration"
}
df['ad_type'] = df['cluster'].map(label_map)


# ──────────────────────────────────────────
# 5. 결과 출력
# ──────────────────────────────────────────

print("\n=== 광고 유형 분포 ===")
print(df['ad_type'].value_counts())

print("\n=== 유형별 주요 피처 평균 ===")
print(df.groupby('ad_type')[['voice_ratio', 'bpm', 'rms_energy', 'spectral_centroid']].mean().round(3))

print("\n=== 유형별 대표 회사 (상위 5개) ===")
for ad_type in df['ad_type'].unique():
    companies = df[df['ad_type'] == ad_type]['company'].value_counts().head(5)
    print(f"\n[{ad_type}]")
    print(companies.to_string())


# ──────────────────────────────────────────
# 6. PCA 시각화 (2D)
# ──────────────────────────────────────────

# PCA로 17차원 → 2차원으로 압축해서 시각화
# PC1, PC2가 전체 분산의 몇 %를 설명하는지도 함께 표시
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.rcParams['font.family'] = 'DejaVu Sans'

# 유형별 색상 지정
colors = {
    'Narration'    : '#E74C3C',  # 빨강
    'CM_Song'      : '#3498DB',  # 파랑
    'BGM_Narration': '#2ECC71'   # 초록
}

fig, ax = plt.subplots(figsize=(10, 7))
for ad_type, color in colors.items():
    mask = df['ad_type'] == ad_type
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=color, label=ad_type, alpha=0.7, s=60)

ax.set_title('Radio Ad Clustering (PCA 2D)', fontsize=14)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
print(f"\nPCA 시각화 저장: {PLOT_PATH}")


# ──────────────────────────────────────────
# 7. 결과 저장
# ──────────────────────────────────────────

# ad_type 컬럼이 추가된 전체 데이터 저장
# 이후 heartlib 광고 생성 시 타입별 Tags 참조용으로 활용
df.to_csv(OUTPUT_CSV, index=False)
print(f"클러스터 결과 저장: {OUTPUT_CSV}")