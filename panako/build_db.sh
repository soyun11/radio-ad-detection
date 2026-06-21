source ./config.sh

if [ $# -ne 1 ]; then
    echo "Usage: build_db.sh YYYYMMDD"
    exit 1
fi

TARGET_DATE=$1

echo "📦 Building Panako DB (before $TARGET_DATE, $HISTORY_DAYS days)"

for i in $(seq $HISTORY_DAYS -1 1); do
    DATE=$(date -d "$TARGET_DATE -$i day" +%Y%m%d)
    MP3="$BASE_DIR/$DATE/mp3/$DATE.mp3"

    if [ -f "$MP3" ]; then
        echo "➕ Storing $MP3"
        $PANAKO_BIN store "$MP3"
    else
        echo "⚠️ Missing $MP3"
    fi
done

