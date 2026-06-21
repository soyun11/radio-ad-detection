cd panako
source .venv/bin/activate

export OLAF_MIN_HITS_FILTERED=3
export OLAF_MIN_MATCH_DURATION=3
export OLAF_MIN_SEC_WITH_MATCH=0.1


BASE_DIR="noon"


WORK_DIR="panako/noon"
mkdir -p $WORK_DIR/clips
mkdir -p $WORK_DIR/results

DATES=(20260101 20260102 20260108 20260109 20260110 20260111 20260112 20260113 20260114 20260115)

echo "========================================"
echo "🚀 KBS noon Panako 광고 탐지 (20초 클립)"
echo "========================================"
echo "날짜: ${DATES[*]}"
echo "작업 폴더: $WORK_DIR"
echo ""

echo "[Step 1] 모든 날짜의 클립 생성 (20초 윈도우)"
echo "========================================"
for DATE in "${DATES[@]}"; do

    if [ -d "$WORK_DIR/clips/$DATE" ]; then
        echo "🗑️ $DATE 기존 클립 삭제..."
        rm -rf "$WORK_DIR/clips/$DATE"
    fi
    
    echo "🔪 $DATE 클립 생성 중 (20초)..."
    MP3_PATH="$BASE_DIR/$DATE/mp3/$DATE.mp3"
    if [ -f "$MP3_PATH" ]; then
        python mp3-segmentation-noon.py $DATE --output_dir $WORK_DIR/clips/$DATE --segment_length 20
    else
        echo "⚠️ MP3 없음: $MP3_PATH"
    fi
done
echo ""

echo "[Step 1.5] 기존 결과 삭제"
echo "========================================"
rm -f $WORK_DIR/results/*.csv
echo "✅ 기존 결과 삭제 완료"
echo ""

echo "[Step 2] 날짜별 매칭 시작"
echo "========================================"

for i in "${!DATES[@]}"; do
    if [ $i -eq 0 ]; then
        echo "⏭️ ${DATES[$i]} - 첫 번째 날짜라 건너뜀"
        continue
    fi
    
    QUERY_DATE=${DATES[$i]}
    DB_DATE=${DATES[$((i-1))]}
    
    echo ""
    echo "----------------------------------------"
    echo "📅 $QUERY_DATE vs $DB_DATE 비교"
    echo "----------------------------------------"
    

    bash delete_db.sh
    
    DB_MP3="$BASE_DIR/$DB_DATE/mp3/$DB_DATE.mp3"
    if [ -f "$DB_MP3" ]; then
        echo "📦 Building DB from $DB_MP3"
        panako store "$DB_MP3"
    else
        echo "⚠️ DB MP3 없음: $DB_MP3"
        continue
    fi
    

    echo "🔍 매칭 중..."
    python seg-lookup3.py $WORK_DIR/clips/$QUERY_DATE $DB_DATE
    

    COMPARE_FILE="${QUERY_DATE}-${DB_DATE}-compare.csv"
    
    if [ ! -f "$COMPARE_FILE" ]; then
        echo "⚠️ 매칭 없음"
        continue
    fi
    

    echo "📊 클러스터링 (gap_threshold=30)..."
    python cluster-max-v3.py $COMPARE_FILE --gap_threshold 30

    AD_RESULT_FILE="${QUERY_DATE}-${DB_DATE}-compare-ad-result.csv"
    if [ -f "$AD_RESULT_FILE" ]; then
        mv ${QUERY_DATE}-${DB_DATE}-*.csv $WORK_DIR/results/
        echo "✅ 완료: $WORK_DIR/results/"
    fi


    echo "🎤 Whisper 전사 중..."
    python whisper_ad_faster.py \
        --input $WORK_DIR/results/${QUERY_DATE}-${DB_DATE}-compare-ad-result.csv \
        --output $WORK_DIR/results/${QUERY_DATE}-whisper.csv

done

echo ""
echo "========================================"
echo "[Step 3] 전체 결과 요약"
echo "========================================"

RESULT_COUNT=$(ls $WORK_DIR/results/*-ad-result.csv 2>/dev/null | wc -l)
echo "총 결과 파일: $RESULT_COUNT 개"

for f in $WORK_DIR/results/*-ad-result.csv; do
    if [ -f "$f" ]; then
        COUNT=$(($(wc -l < "$f") - 1))
        echo "  $(basename $f): $COUNT 개 광고 클립"
    fi
done

echo ""
echo "Whisper 전사 결과:"
for f in $WORK_DIR/results/*-whisper.csv; do
    if [ -f "$f" ]; then
        COUNT=$(($(wc -l < "$f") - 1))
        echo "  $(basename $f): $COUNT 개 전사"
    fi
done

echo ""
echo "🎉 KBS noon Panako 완료!"
echo "결과 폴더: $WORK_DIR/results/"