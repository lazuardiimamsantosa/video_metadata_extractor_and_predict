import os
import json
import csv
import pandas as pd
from pathlib import Path
import pickle
import numpy as np

from video.ftyp import *
from video.moof import *
from video.moov import *
from video.moov_subatom import *
from video.nal_parser import *
from video.featureExtraction import videoParsing, featureExtraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder




def ambil_label_dari_folder(path_dataset):
    return sorted([
        folder.name for folder in Path(path_dataset).iterdir()
        if folder.is_dir()
    ])

LABELS = ambil_label_dari_folder(Path("D:/local Disk D/UGM/Semester 1/Penelitian/video_source_identifier-main/Dataset/Dataset"))
features = [
    'BOX_Sequence', 'Video_Format_settings', 'Writing_application',
    'Video_Format_profile', 'Movie_name', 'Video_Title', 'Format_profile'
]


def extract_video_metadata(video_path: str) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"File video tidak ditemukan di: {video_path}")

    metadata = {}

    # Label folder
    parent_directory_name = Path(video_path).parent.name
    metadata['label_folder'] = parent_directory_name

    if parent_directory_name not in LABELS:
        print(f"Peringatan: Nama folder '{parent_directory_name}' tidak ada dalam daftar LABELS yang ditentukan.")

    # Parsing video
    AtomDict = {}  # pastikan ini dict kosong
    boxSequence = []  # pastikan ini list kosong

    try:
        # Tambahkan reset struktur jika perlu
        codec_check = videoParsing(video_path, {}, [])  # pakai dict dan list baru

        # Atau jika `videoParsing` mengisi AtomDict/boxSequence:
        AtomDict.clear()
        boxSequence.clear()
        codec_check = videoParsing(video_path, AtomDict, boxSequence)

        df = featureExtraction(video_path, AtomDict.copy(), codec_check, boxSequence[:])

        metadata.update(df.to_dict(orient='records')[0])
        print(f"✅ Metadata berhasil diekstraksi dari: {video_path}")
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat parsing video: {e}")
        raise

    return metadata



def simpan_ke_csv(list_metadata: list, nama_file: str = "output_metadata.csv"):
    df = pd.DataFrame(list_metadata)
    df.to_csv(nama_file, index=False)
    print(f"Data metadata berhasil disimpan ke {nama_file}")


def proses_semua_video(direktori: str):
    hasil_metadata = []

    for root, dirs, files in os.walk(direktori):
        for file in files:
            if file.lower().endswith((".mp4", ".mov", ".avi")):
                full_path = os.path.join(root, file)
                try:
                    metadata = extract_video_metadata(full_path)
                    hasil_metadata.append(metadata)
                except Exception as e:
                    print(f"Gagal memproses {full_path}: {e}")

    if hasil_metadata:
        simpan_ke_csv(hasil_metadata)
    else:
        print("Tidak ada metadata yang berhasil diekstraksi.")


def simple_tokenizer(texts, word_index, max_len):
    sequences = []
    for text in texts:
        tokens = text.lower().split()
        seq = [word_index.get(token, 0) for token in tokens]
        if len(seq) < max_len:
            seq = [0] * (max_len - len(seq)) + seq
        else:
            seq = seq[:max_len]
        sequences.append(seq)
    return np.array(sequences)



def preprocessing_only(df, features, predefined_vocabularies=None, n_components=10):
    if predefined_vocabularies is None:
        predefined_vocabularies = {}  # kosongkan jika tidak ada vocab khusus

    for feature in features:
        texts = []
        for data in df[feature]:
            data = 'nan' if data == 'None' else str(data)
            text = data.replace('.', ' ').replace('_', ' ').strip().lower()
            texts.append(text)

        # Buang teks kosong
        texts = [t for t in texts if t.strip()]

        if len(texts) < 2:
            print(f"⚠️  Lewati fitur '{feature}' karena data kosong atau hanya 1 baris.")
            continue

        try:
            if feature in predefined_vocabularies:
                # 🔁 Gunakan vocab predefined
                vocab = predefined_vocabularies[feature]
                encoder = OrdinalEncoder(categories=[vocab], handle_unknown='use_encoded_value', unknown_value=-1)
                encoded = encoder.fit_transform(pd.DataFrame(texts))
                df[f"{feature}_encoded"] = encoded
            else:
                # 🧠 Default: CountVectorizer + PCA
                vectorizer = CountVectorizer()
                tokenized = vectorizer.fit_transform(texts)

                if tokenized.shape[1] == 0:
                    print(f"⚠️  Lewati fitur '{feature}' karena tidak ada vocabulary yang valid.")
                    continue

                pca = PCA(n_components=min(n_components, tokenized.shape[1], tokenized.shape[0] - 1))
                transformed = pca.fit_transform(tokenized.toarray())

                transformed_df = pd.DataFrame(transformed, columns=[f"{feature}_{i}" for i in range(transformed.shape[1])])
                df = pd.concat([df.reset_index(drop=True), transformed_df], axis=1)

        except Exception as e:
            print(f"⚠️  Gagal memproses feature '{feature}': {e}")
            continue

    numeric_cols = [
        'Video_Bitrate', 'Video_Width(Pixels)', 'Audio_Bitrate',
        'Video_Height(Pixels)', 'Overall_bitrate',
        'Video_Matrix_coefficients', 'Codec_ID'
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]
    encoded_cols = [c for c in df.columns if '_' in c and c not in available_cols]
    df = df[available_cols + encoded_cols]
    return df


def extract_features_only(video_path):
    AtomDict, boxSequence = {}, []
    codec_check = videoParsing(video_path, AtomDict, boxSequence)
    df = featureExtraction(video_path, AtomDict, codec_check, boxSequence)

    features = [
        'BOX_Sequence', 'Video_Format_settings', 'Writing_application',
        'Video_Format_profile', 'Movie_name', 'Video_Title', 'Format_profile'
    ]
    x_data = preprocessing_only(df, features, n_components=10)

    # ✅ Tambahkan label_folder dari parent folder
    label_folder = Path(video_path).parent.name
    x_data['label_folder'] = label_folder

    return x_data




if __name__ == "__main__":
    video_dir = Path("D:/local Disk D/UGM/Semester 1/Penelitian/video_source_identifier-main/Dataset/Dataset")

    all_features = []

    for video_path in video_dir.rglob("*.mp4"):
        try:
            fitur = extract_features_only(str(video_path))
            # Hapus baris ini kalau tidak mau include nama file
            # fitur['file'] = video_path.name  
            all_features.append(fitur)
            print(f"✅ Berhasil ekstraksi fitur dari {video_path}")
        except Exception as e:
            print(f"❌ Gagal ekstraksi dari {video_path}: {e}")

    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)

        # Simpan ke CSV
        final_df.to_csv("fitur_output.csv", index=False)
        print("📁 Semua fitur disimpan ke fitur_output.csv")

        # Simpan ke Excel
        final_df.to_excel("fitur_output.xlsx", index=False)
        print("📁 Semua fitur juga disimpan ke fitur_output.xlsx")
