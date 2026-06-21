import os
import subprocess
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse


def query_panako(file_path):
    try:
        result = subprocess.run(
            ["panako", "query", file_path],  
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return file_path, result.stdout
    except Exception as e:
        print(f"Error querying {file_path}: {e}")
        return file_path, None


def parse_query_result(result):
    matches = []
    lines = result.splitlines()

    for line in lines:

        if not line.strip() or "null" in line or "-1.000" in line:
            continue

        fields = line.split(";")
        
        if len(fields) >= 13 and fields[0].strip().isdigit():
            try:
                query_path = fields[2].strip()          
                query_start = float(fields[3].strip())  
                query_stop = float(fields[4].strip())   
                match_path = fields[5].strip()          
                match_id = fields[6].strip()            
                match_start = float(fields[7].strip())  
                match_stop = float(fields[8].strip())   
                match_score = int(fields[9].strip())   
                time_factor = float(fields[10].strip().replace("%", ""))
                frequency_factor = float(fields[11].strip().replace("%", ""))
                seconds_with_match = float(fields[12].strip())

                matches.append({
                    "Query Path": query_path,
                    "Query Start": query_start,
                    "Query Stop": query_stop,
                    "Match Path": match_path,
                    "Match ID": match_id,
                    "Match Start": match_start,     
                    "Match Stop": match_stop,        
                    "Match Score": match_score,      
                    "Time Factor (%)": time_factor,
                    "Frequency Factor (%)": frequency_factor,
                    "Seconds with Match (%)": seconds_with_match
                })
            except Exception as e:
               
                print(f"\n⚠️ 데이터 파싱 건너뜀: {line} | 사유: {e}")

    return matches if matches else None


def process_file(file_path):
    file_path, query_result = query_panako(file_path)
    if query_result:
        matches = parse_query_result(query_result)
        return matches
    return None


def process_directory_parallel(input_dir, output_csv):
    mp3_files = [
        os.path.join(input_dir, f) 
        for f in sorted(os.listdir(input_dir)) 
        if f.endswith(".mp3")
    ]
    print(f"Found {len(mp3_files)} MP3 files in {input_dir}")

    results = []
    with ThreadPoolExecutor() as executor:

        future_to_file = {
            executor.submit(process_file, file): file 
            for file in mp3_files
        }

        for future in tqdm(as_completed(future_to_file), total=len(mp3_files), desc="Processing files"):
            try:
                matches = future.result()
                if matches:
                    results.extend(matches) 
            except Exception as e:
                print(f"Error processing file: {e}")

    if results:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=[
                "Query Path", "Query Start", "Query Stop",
                "Match Path", "Match ID", "Match Start", "Match Stop",
                "Match Score", "Time Factor (%)", "Frequency Factor (%)", "Seconds with Match (%)"
            ])
            writer.writeheader()
            writer.writerows(results)

        print(f"\n🎉 성공! Results saved to {output_csv}")
    else:
        print("\nNo matches found. CSV file not created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process MP3 files in a directory using Panako and save the results."
    )
    parser.add_argument("input_dir", type=str, 
                        help="Path to the input directory containing MP3 files.")
    parser.add_argument("compare_date", type=str, 
                        help="compare date in panako.")
    args = parser.parse_args()

    output_csv = (
        os.path.basename(args.input_dir.rstrip("/")) 
        + "-" 
        + os.path.basename(args.compare_date) 
        + "-compare.csv"
    )
    
    process_directory_parallel(args.input_dir, output_csv)