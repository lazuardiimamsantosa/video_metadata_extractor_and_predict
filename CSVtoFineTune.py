import pandas as pd
import json

def csv_to_gemini_jsonl(csv_file_path, jsonl_output_path):
    df = pd.read_csv(csv_file_path)

    # Hapus koma ribuan di kolom numerik lalu ubah jadi int
    for col in ['Video_Bitrate', 'Video_Width(Pixels)', 'Audio_Bitrate', 'Video_Height(Pixels)', 'Overall_bitrate']:
        df[col] = df[col].astype(str).str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    with open(jsonl_output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            input_text = (
                f"Video bitrate: {row['Video_Bitrate']} kbps, "
                f"Width: {row['Video_Width(Pixels)']} px, "
                f"Height: {row['Video_Height(Pixels)']} px, "
                f"Audio bitrate: {row['Audio_Bitrate']} kbps, "
                f"Overall bitrate: {row['Overall_bitrate']} kbps, "
                f"Format: {row['Format_profile']}, "
                f"Codec ID: {row['Codec_ID']}, "
                f"Writing app: {row['Writing_application']}, "
                f"Movie name: {row['Movie_name']}, "
                f"Video profile: {row['Video_Format_profile']}, "
                f"Format settings: {row['Video_Format_settings']}, "
                f"Color range: {row['Video_Color_range']}, "
                f"Color primaries: {row['Video_Color_primaries']}, "
                f"Transfer char: {row['Video_Transfer_characteristics']}, "
                f"Matrix coeff: {row['Video_Matrix_coefficients']}, "
                f"Box sequence: {row['BOX_Sequence']}, "
                f"Audio title: {row['Audio_Title']}"
            )

            example = {
                "input": input_text,
                "output": row["label_folder"]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"✅ Konversi selesai! File JSONL disimpan di: {jsonl_output_path}")

if __name__ == "__main__":
    csv_to_gemini_jsonl("fitur_output.csv", "data_gemini_format.json")
