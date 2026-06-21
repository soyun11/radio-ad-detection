"""
extract_ad_features.py

[전체 목적]
라디오 광고 생성 파이프라인의 Step 1 — 광고 분석.
heartlib으로 광고를 생성하려면 "어떤 스타일로 만들어줘"라는 입력(Tags)이 필요한데,
그 Tags를 자동으로 만들기 위해 기존 광고들의 오디오 특징을 분석하는 스크립트.

흐름:
  기존 광고 분석 → Tags 추출 → (Step 2) heartlib에 넣어서 새 광고 생성

구체적으로 하는 일:
  1. noon_v5_gap30_ad_detail.csv에서 panako로 탐지된 광고(detected=True)만 필터링
  2. 날짜별 방송 mp3에서 광고 구간(ad_start ~ ad_end)을 잘라 클립 mp3로 저장
  3. 각 클립에서 librosa로 오디오 피처(BPM, 에너지, 음색 등) 추출
  4. whisper SRT 자막 파일로 나레이션 비율(voice_ratio) 계산
  5. 피처들을 heartlib Tags 형식("upbeat,energetic,voice_heavy")으로 변환
  6. 결과를 ad_features.csv로 저장

[입력 파일]
  - noon_v5_gap30_ad_detail.csv : panako + 화자분리 기반 광고 탐지 결과
      컬럼: ad_idx, ad_start, ad_end, company, overlap_ratio, detected, date
  - {date}.mp3     : 날짜별 방송 원본 (~/radio2/noon/YYYYMMDD/mp3/)
  - {date}_vo.srt  : whisper 전사 자막 (~/radio2/noon/YYYYMMDD/transcript/)

[출력 파일]
  - clips/         : 광고별 클립 mp3 (~/radio2/ad_analysis/clips/)
  - ad_features.csv: 광고별 피처 + Tags (~/radio2/ad_analysis/results/)
      주요 컬럼: company, bpm, rms_energy, spectral_centroid, mfcc_0~12,
                voice_ratio, tags, clip_path

[출력 Tags 형식 예시]
  "upbeat,energetic,voice_heavy"  → 빠르고 에너지 높은 나레이션 위주 광고
  "medium,moderate,music_heavy"   → 중간 템포, 음악 위주 광고
  "slow,quiet,voice_heavy"        → 조용하고 느린 나레이션 광고

[한계]
  - librosa 피처는 룰 기반이라 "분위기/감성" 같은 추상적 특징은 잘 못 잡음
  - 더 정교한 임베딩이 필요하면 extract_clap_embeddings.py 사용
"""

import os
import re
import pandas as pd
import librosa
import numpy as np
from pydub import AudioSegment


# ──────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────

# 광고 블록 탐지 결과 CSV
# panako(오디오 핑거프린트) + 화자분리 조합으로 탐지된 결과
# detected=True인 것만 실제 광고로 간주 (blocks.csv는 룰 기반이라 부정확, 이걸 사용)
CSV_PATH = os.path.expanduser("~/radio2/ad_debug/results/noon/noon_v5_gap30_ad_detail.csv")

# 날짜별 라디오 방송 원본이 있는 루트 폴더
# 하위 구조: {NOON_DIR}/{date}/mp3/{date}.mp3
#            {NOON_DIR}/{date}/transcript/{date}_vo.srt
NOON_DIR = os.path.expanduser("~/radio2/noon")

# 광고 클립 mp3를 저장할 폴더
# 파일명 형식: {date}_ad{ad_idx}_{company}.mp3
# 예: 20260102_ad3_경기주택도시공사.mp3
OUTPUT_DIR = os.path.expanduser("~/radio2/ad_analysis/clips")

# 광고별 피처 + Tags 결과를 저장할 CSV
FEATURE_CSV = os.path.expanduser("~/radio2/ad_analysis/results/ad_features.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────
# 1. 광고 클립 추출
# ──────────────────────────────────────────

def extract_clip(mp3_path, start_sec, end_sec, out_path):
    """
    라디오 방송 전체 mp3에서 광고 구간만 잘라 저장.

    pydub의 AudioSegment를 사용해 밀리초 단위로 슬라이싱.
    이미 클립이 존재하면 main()에서 스킵하므로 중복 실행 걱정 없음.

    Args:
        mp3_path (str)   : 원본 방송 mp3 경로
        start_sec (float): 광고 시작 시간 (초 단위, CSV의 ad_start)
        end_sec (float)  : 광고 끝 시간 (초 단위, CSV의 ad_end)
        out_path (str)   : 저장할 클립 mp3 경로
    """
    audio = AudioSegment.from_mp3(mp3_path)
    # pydub은 밀리초 단위 → 초 × 1000
    start_ms = int(start_sec * 1000)
    end_ms = int(end_sec * 1000)
    clip = audio[start_ms:end_ms]
    clip.export(out_path, format="mp3")


# ──────────────────────────────────────────
# 2. 음성 비율 계산 (SRT 활용)
# ──────────────────────────────────────────

def get_voice_ratio(srt_path, ad_start, ad_end):
    """
    whisper가 생성한 SRT 자막 파일을 파싱해서,
    해당 광고 구간에서 실제 사람 목소리(나레이션)가 차지하는 비율을 계산.

    [왜 ZCR 대신 SRT를 쓰냐?]
    Zero Crossing Rate(ZCR)로 음성/음악을 구분하려 했지만,
    라디오 광고의 ZCR 분포가 0.04~0.11로 너무 좁아서 기준값 설정이 어려움.
    → whisper가 이미 전체 방송의 음성 구간을 SRT로 뽑아놨으므로
      광고 구간과 SRT 자막 구간의 겹치는 시간을 계산하면
      나레이션 비율을 훨씬 정확하게 구할 수 있음.

    계산 방식:
        voice_ratio = 광고 구간 내 SRT 자막 겹침 시간 합계 / 광고 전체 길이

    값 해석:
        0에 가까울수록 → 음악 위주 광고 (music_heavy 태그)
        1에 가까울수록 → 나레이션 위주 광고 (voice_heavy 태그)
        기준값 0.5: 절반 이상이 음성이면 voice_heavy로 분류

    Args:
        srt_path (str)  : whisper SRT 파일 경로 (예: 20260102_vo.srt)
        ad_start (float): 광고 시작 시간 (초)
        ad_end (float)  : 광고 끝 시간 (초)

    Returns:
        float: voice_ratio (0.0 ~ 1.0). SRT 파일 없으면 0.0 반환.
    """
    if not os.path.exists(srt_path):
        return 0.0

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # SRT 타임스탬프 패턴 예시: "00:10:23,456 --> 00:10:25,789"
    pattern = r'(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)'
    matches = re.findall(pattern, content)

    def to_sec(t):
        """
        SRT 타임스탬프 문자열을 초 단위 float로 변환.
        예: "00:10:23,456" → 623.456
        쉼표(,)를 점(.)으로 바꿔서 float 변환 처리.
        """
        h, m, s = t.replace(',', '.').split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    # 광고 구간과 각 SRT 자막 구간의 교집합(겹치는 시간) 합산
    voice_duration = 0
    for start_t, end_t in matches:
        s, e = to_sec(start_t), to_sec(end_t)
        # 두 구간 [ad_start, ad_end]와 [s, e]의 교집합 길이
        # min(끝점) - max(시작점) 이 양수이면 겹치는 구간 존재
        overlap = max(0, min(e, ad_end) - max(s, ad_start))
        voice_duration += overlap

    ad_duration = ad_end - ad_start
    if ad_duration <= 0:
        return 0.0
    return voice_duration / ad_duration


# ──────────────────────────────────────────
# 3. 오디오 피처 추출 (librosa)
# ──────────────────────────────────────────

def extract_features(clip_path):
    """
    librosa를 사용해 광고 클립에서 오디오 피처를 추출.

    [각 피처 설명]
    bpm (Beats Per Minute):
        - 음악의 빠르기. beat_track 알고리즘으로 추정.
        - 70 미만: slow, 70~120: medium, 120 이상: upbeat 태그로 변환됨.

    rms_energy (Root Mean Square Energy):
        - 전체 프레임의 평균 음량. 클수록 소리가 크고 강렬함.
        - 0.05 미만: quiet, 0.05~0.15: moderate, 0.15 이상: energetic 태그.

    spectral_centroid (스펙트럼 무게중심):
        - 주파수 스펙트럼의 무게중심. Hz 단위.
        - 높을수록 밝고 날카로운 소리 (고음역대).
        - 낮을수록 어둡고 묵직한 소리 (저음역대, 나레이션).
        - 현재는 클러스터링에만 활용.

    zero_crossing_rate (ZCR):
        - 파형이 0을 교차하는 비율.
        - 원래 voice/music 구분에 쓰려 했으나 분포가 좁아 voice_ratio로 대체.
        - 클러스터링 참고용으로만 보존.

    mfcc_0 ~ mfcc_12 (Mel-Frequency Cepstral Coefficients):
        - 사람 귀의 청각 특성을 모방한 13차원 음색 벡터.
        - 악기 구성, 톤, 음색 등을 종합적으로 표현.
        - K-Means 클러스터링(cluster_ads.py)의 주요 입력 피처.

    Args:
        clip_path (str): 광고 클립 mp3 경로

    Returns:
        dict: 피처 딕셔너리
              {"bpm": float, "rms_energy": float, "spectral_centroid": float,
               "zero_crossing_rate": float, "mfcc_0": float, ..., "mfcc_12": float}
    """
    y, sr = librosa.load(clip_path, sr=None)

    # 템포 추정 (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # RMS 에너지: 전체 프레임의 평균
    rms = float(np.mean(librosa.feature.rms(y=y)))

    # 스펙트럼 무게중심 (밝기)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # Zero Crossing Rate (보조 참고용)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    # MFCC 13개 계수의 시간 평균
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = {f"mfcc_{i}": float(np.mean(mfccs[i])) for i in range(13)}

    return {
        "bpm": float(tempo),
        "rms_energy": rms,
        "spectral_centroid": centroid,
        "zero_crossing_rate": zcr,
        **mfcc_means
    }


# ──────────────────────────────────────────
# 4. 피처 → heartlib Tags 변환
# ──────────────────────────────────────────

def bpm_to_tag(bpm):
    """
    BPM을 heartlib 태그로 변환.

    heartlib은 "slow", "medium", "upbeat" 같은 텍스트 태그로
    음악 스타일을 제어하므로 수치를 태그로 변환해야 함.

    기준값:
        slow   : BPM < 70   (느린 발라드, 잔잔한 광고)
        medium : 70 ~ 120   (중간 템포 광고)
        upbeat : BPM >= 120 (빠른 댄스, 활기찬 CM송)
    """
    if bpm < 70:
        return "slow"
    elif bpm < 120:
        return "medium"
    else:
        return "upbeat"


def energy_to_tag(rms):
    """
    RMS 에너지를 heartlib 태그로 변환.

    기준값 (실제 데이터 분포 기반, 307개 광고 기준):
        quiet     : RMS < 0.05  (조용한 나레이션 광고)
        moderate  : 0.05 ~ 0.15 (일반적인 라디오 광고 음량, 평균 0.12)
        energetic : RMS >= 0.15 (강렬한 CM송, 효과음 많은 광고)
    """
    if rms < 0.05:
        return "quiet"
    elif rms < 0.15:
        return "moderate"
    else:
        return "energetic"


def features_to_tags(features):
    """
    추출된 피처들을 heartlib이 이해하는 Tags 문자열로 변환.

    heartlib 태그 형식: "태그1,태그2,태그3" (쉼표 구분, 공백 없음)
    예시: "upbeat,energetic,voice_heavy"

    Tags 구성 (3개):
        1. 템포 태그   : slow / medium / upbeat       (bpm 기반)
        2. 에너지 태그 : quiet / moderate / energetic  (rms_energy 기반)
        3. 음성 태그   : voice_heavy / music_heavy     (voice_ratio 기반)
                        voice_ratio > 0.5 → 나레이션이 절반 이상 → voice_heavy

    Args:
        features (dict): extract_features() 결과 + voice_ratio가 추가된 딕셔너리

    Returns:
        str: heartlib Tags 문자열. 예: "upbeat,moderate,voice_heavy"
    """
    tags = []
    tags.append(bpm_to_tag(features["bpm"]))
    tags.append(energy_to_tag(features["rms_energy"]))

    # voice_ratio가 없으면 0으로 처리 (SRT 파일 없는 경우 대비)
    if features.get("voice_ratio", 0) > 0.5:
        tags.append("voice_heavy")
    else:
        tags.append("music_heavy")

    return ",".join(tags)


# ──────────────────────────────────────────
# 5. 메인 파이프라인
# ──────────────────────────────────────────

def main():
    """
    전체 파이프라인 실행.

    실행 순서:
        1. CSV 로드 → detected=True 광고만 필터링
        2. 날짜별 mp3에서 광고 클립 추출 (이미 있으면 스킵)
        3. 각 클립에서 librosa 오디오 피처 추출
        4. SRT 기반 voice_ratio 계산
        5. heartlib Tags 생성
        6. 전체 결과를 ad_features.csv로 저장

    참고:
        - 클립이 이미 존재하면 자르는 과정을 스킵 → 재실행해도 안전.
        - mp3 파일이 없는 날짜는 자동으로 건너뜀.
        - 피처 추출 실패한 클립은 에러 출력 후 건너뜀 (전체 중단 없음).
    """
    # detected=True인 광고만 사용 (panako로 실제 확인된 광고)
    df = pd.read_csv(CSV_PATH)
    df = df[df["detected"] == True].reset_index(drop=True)
    print(f"총 {len(df)}개 광고 처리 시작")

    results = []
    for _, row in df.iterrows():
        date = str(int(row["date"]))

        # 원본 방송 mp3 경로
        # 예: ~/radio2/noon/20260102/mp3/20260102.mp3
        mp3_path = os.path.join(NOON_DIR, date, "mp3", f"{date}.mp3")

        # whisper SRT 경로
        # 예: ~/radio2/noon/20260102/transcript/20260102_vo.srt
        srt_path = os.path.join(NOON_DIR, date, "transcript", f"{date}_vo.srt")

        if not os.path.exists(mp3_path):
            print(f"mp3 없음 스킵: {mp3_path}")
            continue

        # 클립 파일명: {date}_ad{ad_idx}_{company}.mp3
        # 예: 20260102_ad3_경기주택도시공사.mp3
        clip_name = f"{date}_ad{int(row['ad_idx'])}_{row['company']}.mp3"
        clip_path = os.path.join(OUTPUT_DIR, clip_name)

        # 클립이 없을 때만 자르기 (이미 있으면 스킵 → 재실행 안전)
        if not os.path.exists(clip_path):
            try:
                extract_clip(mp3_path, row["ad_start"], row["ad_end"], clip_path)
            except Exception as e:
                print(f"클립 추출 실패: {clip_name} - {e}")
                continue

        # 오디오 피처 추출 + voice_ratio 계산 + Tags 생성
        try:
            features = extract_features(clip_path)
            features["voice_ratio"] = get_voice_ratio(srt_path, row["ad_start"], row["ad_end"])
            tags = features_to_tags(features)

            results.append({
                "date": date,
                "ad_idx": row["ad_idx"],
                "company": row["company"],
                "ad_start": row["ad_start"],
                "ad_end": row["ad_end"],
                "duration": row["ad_end"] - row["ad_start"],
                "clip_path": clip_path,
                "tags": tags,
                **features  # bpm, rms_energy, spectral_centroid, zcr, mfcc_0~12, voice_ratio
            })
            print(f"완료: {clip_name} | voice_ratio: {features['voice_ratio']:.2f} | tags: {tags}")
        except Exception as e:
            print(f"피처 추출 실패: {clip_name} - {e}")

    # 전체 결과 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(FEATURE_CSV, index=False)
    print(f"\n완료! 결과 저장: {FEATURE_CSV}")
    print(result_df[["company", "bpm", "rms_energy", "voice_ratio", "tags"]].head(10))


if __name__ == "__main__":
    main()