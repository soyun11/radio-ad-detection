import argparse
import os
import subprocess
from os.path import join, exists


BASE_DIR_ROOT = ""

BROADCASTER_CONFIG = {
    "baechulsu": {
        "base_dir": join(BASE_DIR_ROOT, "baechulsu"),
        "clips_dir": "clips_baechulsu",
    },
    "movie": {
        "base_dir": join(BASE_DIR_ROOT, "movie"),
        "clips_dir": "clips_movie",
    },
    "noon": {
        "base_dir": join(BASE_DIR_ROOT, "noon"),
        "clips_dir": "clips_noon",
    },
}


def split_mp3(input_file, output_dir, segment_length, step):
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_file
    ]
    total_duration = float(subprocess.check_output(cmd).decode().strip())
    total_clips = int((total_duration - segment_length) / step) + 1
    print(f"[INFO] 총 방송 길이: {total_duration:.1f}초, 예상 클립 수: {total_clips}개")

    t = 0.0
    idx = 0
    while t + segment_length <= total_duration:
        out_name = f"clip_{idx:06d}_{int(t * 1000):06d}.mp3"
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

        if idx % 100 == 0:
            pct = t / total_duration * 100
            print(f"  [{idx:>4}/{total_clips}] {t:>7.1f}s / {total_duration:.1f}s  ({pct:.1f}%)",
                  flush=True)

        t += step
        idx += 1

    print(f"[DONE] generated {idx} clips in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split radio broadcast MP3 into sliding-window clips for Panako fingerprinting"
    )
    parser.add_argument(
        "broadcaster", type=str,
        choices=["baechulsu", "movie", "noon"],
        help="Broadcaster name (baechulsu=MBC, movie=SBS, noon=KBS)"
    )
    parser.add_argument(
        "date", type=str,
        help="Broadcast date (YYYYMMDD)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save clips (default: clips_{broadcaster}/{date})"
    )
    parser.add_argument(
        "--segment_length", type=int, default=20,
        help="Clip length in seconds (default: 20)"
    )
    parser.add_argument(
        "--step", type=int, default=3,
        help="Sliding step in seconds (default: 3)"
    )

    args = parser.parse_args()
    config = BROADCASTER_CONFIG[args.broadcaster]

    input_file = join(config["base_dir"], args.date, "mp3", f"{args.date}.mp3")
    if not exists(input_file):
        raise FileNotFoundError(f"MP3 not found: {input_file}")

    output_dir = args.output_dir or join(config["clips_dir"], args.date)

    print(f"[INFO] broadcaster : {args.broadcaster}")
    print(f"[INFO] input       : {input_file}")
    print(f"[INFO] output      : {output_dir}")
    print(f"[INFO] window={args.segment_length}s  step={args.step}s")

    split_mp3(input_file, output_dir, args.segment_length, args.step)