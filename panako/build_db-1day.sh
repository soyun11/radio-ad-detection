source ./config.sh

if [ $# -ne 1 ]; then
    echo "Usage: build_db.sh YYYYMMDD"
    exit 1
fi

DATE=$1

echo "📦 Building Panako DB ($DATE)"

MP3="$BASE_DIR/$DATE/mp3/$DATE.mp3"

if [ -f "$MP3" ]; then
    echo "➕ Storing $MP3"
    
    $PANAKO_BIN store "$MP3"
else
    echo "⚠️ Missing $MP3"
fi