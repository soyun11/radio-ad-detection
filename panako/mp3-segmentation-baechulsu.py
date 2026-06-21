import argparse
import os
import subprocess
from os.path import join, exists

BASE_DIR = "baechulsu"

def split_mp3(input_file, output_dir, segment_length, step):
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_file
    ]
    total_duration = float(subprocess.check_output(cmd).decode().strip())

    t = 0.0
    idx = 0
    while t + segment_length <= total_duration:
        out_name = f"clip_{idx:06d}_{int(t*1000):06d}.mp3"
        out_path = join(output_dir, out_name)

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", str(t),
            "-t", str(segment_length),
            "-i", input_file,
            "-c", "copy",
            out_path
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        t += step
        idx += 1

    print(f"[DONE] generated {idx} clips in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split baechulsu radio MP3 into sliding-window clips"
    )
    parser.add_argument("date", type=str, help="Date (YYYYMMDD)")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save clips (default: clips_baechulsu/<date>)"
    )
    parser.add_argument(
        "--segment_length", type=int, default=20,
        help="Window length in seconds (default: 20)"
    )
    parser.add_argument(
        "--step", type=int, default=3,
        help="Sliding step in seconds (default: 3)"
    )

    args = parser.parse_args()
    date = args.date

    input_file = join(BASE_DIR, date, "mp3", f"{date}.mp3")

    if not exists(input_file):
        raise FileNotFoundError(f"MP3 not found: {input_file}")

    output_dir = args.output_dir or join("clips_baechulsu", date)

    print(f"[INFO] input : {input_file}")
    print(f"[INFO] output: {output_dir}")
    print(f"[INFO] window={args.segment_length}s step={args.step}s")

    split_mp3(
        input_file,
        output_dir,
        args.segment_length,
        args.step
    )