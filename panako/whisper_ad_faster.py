import csv
import os
import argparse
from tqdm import tqdm
from faster_whisper import WhisperModel

DEFAULT_MODEL = "large-v3"   
DEVICE = "cuda"             
COMPUTE_TYPE = "float16" 


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def transcribe_audio(model, audio_path):

    try:

        segments, info = model.transcribe(
            audio_path,
            language="ko",   
            beam_size=5,        
            vad_filter=True      
        )

        texts = []
        avg_logprobs = []

        for seg in segments:
            texts.append(seg.text.strip())
            if seg.avg_logprob is not None:
                avg_logprobs.append(seg.avg_logprob)

        return {
            "transcript": " ".join(texts),  
            "language": info.language,       
            "avg_logprob": round(sum(avg_logprobs) / len(avg_logprobs), 4) if avg_logprobs else ""
        }

    except Exception as e:
        return {
            "transcript": "",
            "language": "",
            "avg_logprob": "",
            "error": str(e)
        }


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Panako ad result CSV (ad-result.csv)")
    ap.add_argument("--output", required=True, help="Output CSV with Whisper transcripts")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model size")
    args = ap.parse_args()

    rows = load_rows(args.input)
    print(f"[INFO] loaded {len(rows)} ad clips")

    print(f"[INFO] loading faster-whisper model: {args.model}")
    model = WhisperModel(
        args.model,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )

    out_rows = []

    for r in tqdm(rows, desc="Whisper (faster)"):
        audio_path = r["Query Path"]  

        if not os.path.isfile(audio_path):
            r["transcript"] = ""
            r["language"] = ""
            r["avg_logprob"] = ""
            r["error"] = "file not found"
            out_rows.append(r)
            continue

        tr = transcribe_audio(model, audio_path)

        r["transcript"] = tr.get("transcript", "")
        r["language"] = tr.get("language", "")
        r["avg_logprob"] = tr.get("avg_logprob", "")
        r["error"] = tr.get("error", "")

        out_rows.append(r)

    fieldnames = list(out_rows[0].keys())
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"[DONE] saved → {args.output}")


if __name__ == "__main__":
    main()