import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = ""

STYLE = {
    "DJ":    {"facecolor": "#AAAAAA", "edgecolor": "black", "linewidth": 1, "hatch": None},  
    "AD":    {"facecolor": "#FFB3B3", "edgecolor": "black", "linewidth": 1, "hatch": "///"}, 
    "MUSIC": {"facecolor": "#6BA3E8", "edgecolor": "black", "linewidth": 1, "hatch": None},   
}

DATE_CONFIG = {
    "20241125": ("baechulsu", "20241124", "MBC"),
    "20241121": ("movie",     "20241120", "SBS"),
    "20260102": ("noon",      "20260101", "KBS"),
}

FIXED_MAX_SEC = 7200  

def safe_int(x):
    return int(max(0, np.floor(float(x))))

def normalize_label(x):
    x = str(x).strip().upper()
    if "GUEST" in x:
        return "DJ"
    if x in ["DJ"]:
        return "DJ"
    if x in ["MUSIC"]:
        return "MUSIC"
    if x in ["AD"]:
        return "AD"
    return "DJ"   

def build_timeline(blocks_path, music_path, ad_path, date):

    print(f"\n📅 Processing {date}")

    blocks = pd.read_csv(blocks_path)
    music = pd.read_csv(music_path)
    ad = pd.read_csv(ad_path)

    blocks["block_type"] = blocks["block_type"].apply(normalize_label)

    ad = ad.rename(columns={"Cluster Start": "start",
                            "Cluster Stop": "end"})

    max_time = max(
        blocks["end"].max(),
        music["end"].max(),
        ad["end"].max()
    )

    data_T = int(np.ceil(max_time)) + 1
    T = max(data_T, FIXED_MAX_SEC)

    timeline = np.zeros(T)           
    timeline[:data_T] = 1            
    
    for _, r in blocks.iterrows():
        s, e = safe_int(r["start"]), safe_int(r["end"])
        if r["block_type"] == "MUSIC":
            timeline[s:e] = 2
        elif r["block_type"] == "AD":
            timeline[s:e] = 3

    for _, r in music.iterrows():
        s, e = safe_int(r["start"]), safe_int(r["end"])
        timeline[s:e] = 2

    for _, r in ad.iterrows():
        s, e = safe_int(r["start"]), safe_int(r["end"])
        timeline[s:e] = 3

    blocks_out = []
    current = timeline[0]
    start = 0

    for t in range(1, len(timeline)):
        if timeline[t] != current:
            blocks_out.append((current, start, t))
            current = timeline[t]
            start = t

    blocks_out.append((current, start, len(timeline)))

    label_map = {1: "DJ", 2: "MUSIC", 3: "AD"}

    rows = []
    for l, s, e in blocks_out:
        l = int(l)
        if l == 0:
            continue  
        rows.append({
            "block_type": label_map[l],
            "start": float(s),
            "end": float(e),
            "duration": float(e - s),
        })

    df_out = pd.DataFrame(rows)

    out_dir = os.path.join(BASE_DIR, "ad_evaluation", "results", "timeline")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{date}-timeline-3labels.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"   ✅ CSV saved: {csv_path}")

    fig, ax = plt.subplots(figsize=(18, 2.5))

    for _, r in df_out.iterrows():
        bt = r["block_type"]
        st = STYLE[bt]

        ax.barh(
            0,
            r["end"] - r["start"],
            left=r["start"],
            height=0.8,
            color=st["facecolor"],
            edgecolor=st["edgecolor"],
            linewidth=st["linewidth"],
            hatch=st["hatch"] if st["hatch"] else None
        )

    ax.set_xlim(0, FIXED_MAX_SEC)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    broadcaster = DATE_CONFIG[date][2]
    ax.set_title(
        f"{broadcaster} Radio Broadcast Timeline ({date})",
        fontsize=18,
        fontweight='bold',
        pad=12
    )

    legend_patches = [
        mpatches.Patch(
            facecolor=STYLE["DJ"]["facecolor"],
            edgecolor="black",
            linewidth=1,
            label="DJ"
        ),
        mpatches.Patch(
            facecolor=STYLE["MUSIC"]["facecolor"],
            edgecolor="black",
            linewidth=1,
            label="MUSIC"
        ),
        mpatches.Patch(
            facecolor=STYLE["AD"]["facecolor"],
            edgecolor="black",
            linewidth=1,
            hatch=STYLE["AD"]["hatch"],
            label="AD"
        ),
    ]

    ax.legend(handles=legend_patches, loc="upper right")

    png_path = os.path.join(out_dir, f"{date}-timeline-3labels.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"   ✅ PNG saved: {png_path}")

def main():

    for date, (broadcaster, prev_date, _) in DATE_CONFIG.items():

        blocks_path = os.path.join(BASE_DIR, broadcaster, date, "transcript", f"{date}-blocks.csv")

        music_path = os.path.join(BASE_DIR, "ad_evaluation", "ground_truth", broadcaster, f"{date}-selection_music.csv")

        ad_path = os.path.join(BASE_DIR, "panako", broadcaster, "results", f"{date}-{prev_date}-compare-cluster-info.csv")

        for label, path in [("blocks", blocks_path), ("music", music_path), ("ad", ad_path)]:
            if not os.path.exists(path):
                print(f"⚠️ Skipping {date} (missing {label}: {path})")
                break
        else:
            build_timeline(blocks_path, music_path, ad_path, date)

    print("\n🎉 All timelines generated!")


if __name__ == "__main__":
    main()