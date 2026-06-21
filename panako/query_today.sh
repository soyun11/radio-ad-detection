source ./config.sh

if [ $# -ne 1 ]; then
    echo "Usage: query_today.sh YYYYMMDD"
    exit 1
fi

DATE=$1
MP3="$BASE_DIR/$DATE/mp3/$DATE.mp3"
OUT_TXT="$OUT_DIR/panako_$DATE.txt"

echo "🔍 Querying $MP3"
$PANAKO_BIN query "$MP3" > "$OUT_TXT"

echo "✅ Saved: $OUT_TXT"

