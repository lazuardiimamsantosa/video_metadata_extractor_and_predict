# -*- coding: utf-8 -*-
"""
Skrip Inferensi Gemma 2-2B-IT untuk Klasifikasi Metadata Video menggunakan QLoRA.
Versi yang diperbaiki dengan penanganan error yang lebih baik dan logika yang disederhanakan.
"""

# --- Impor Pustaka ---
import torch
import pandas as pd
import numpy as np
import logging
import os
import re
from fuzzywuzzy import process

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("transformers").setLevel(logging.WARNING)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from datasets import Dataset, ClassLabel, Features, Value
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm

# --- Fungsi untuk membersihkan label string ---
def clean_label_string(label_str, possible_labels=None):
    """
    Membersihkan string label yang dihasilkan model dan memetakannya ke
    salah satu label yang valid dari daftar `possible_labels`.
    """
    if label_str is None or label_str == "":
        return ""

    # Konversi ke string, huruf kecil, dan hapus spasi awal/akhir
    cleaned = str(label_str).lower().strip()

    # Hapus konten setelah <end_of_turn> jika ada
    if "<end_of_turn>" in cleaned:
        cleaned = cleaned.split("<end_of_turn>")[0].strip()

    # Ganti karakter non-alfanumerik dengan spasi
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
    
    # Ganti multiple spasi dengan spasi tunggal
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if not possible_labels:
        return cleaned

    # Konversi possible_labels ke lowercase untuk perbandingan
    lower_possible_labels = [l.lower() for l in possible_labels]
    
    # Kecocokan persis
    if cleaned in lower_possible_labels:
        return cleaned

    # Fuzzy matching
    try:
        best_match_fuzzy, score = process.extractOne(cleaned, lower_possible_labels)
        if score >= 85:  # Threshold yang lebih rendah untuk lebih fleksibel
            logging.debug(f"Fuzzy Match: '{cleaned}' -> '{best_match_fuzzy}' (Score: {score})")
            return best_match_fuzzy
    except Exception as e:
        logging.warning(f"Fuzzy matching error: {e}")

    # Pencarian substring/prefix
    sorted_labels = sorted(lower_possible_labels, key=len, reverse=True)
    for label in sorted_labels:
        if label in cleaned or cleaned.startswith(label):
            return label

    # Ambil kata pertama yang valid
    words = cleaned.split()
    for word in words:
        if word in lower_possible_labels:
            return word

    # Fallback: ambil kata pertama
    if words:
        return words[0]

    return cleaned

# --- Fungsi untuk memformat prompt ---
def format_prompt(entry, id_to_name_map):
    """
    Memformat entri data metadata menjadi format prompt untuk model Gemma.
    """
    # Metadata penting untuk klasifikasi
    metadata_points = {
        "Video Bitrate": f"{entry.get('Video_Bitrate', 'N/A')} kbps",
        "Resolution": f"{entry.get('Video_Width(Pixels)', 'N/A')}x{entry.get('Video_Height(Pixels)', 'N/A')}",
        "Overall Bitrate": f"{entry.get('Overall_bitrate', 'N/A')} kbps",
        "Codec ID": f"'{entry.get('Codec_ID', 'N/A')}'",
        "Format Profile": f"'{entry.get('Format_profile', 'N/A')}'",
        "Writing App": f"'{entry.get('Writing_application', 'N/A')}'",
        "Video Format Profile": f"'{entry.get('Video_Format_profile', 'N/A')}'",
        "Video Format Settings": f"'{entry.get('Video_Format_settings', 'N/A')}'",
        "Video Color Range": f"'{entry.get('Video_Color_range', 'N/A')}'",
        "Video Color Primaries": f"'{entry.get('Video_Color_primaries', 'N/A')}'",
        "Video Transfer Characteristics": f"'{entry.get('Video_Transfer_characteristics', 'N/A')}'",
        "Audio Bitrate": f"{entry.get('Audio_Bitrate', 'N/A')} kbps",
    }
    
    # Filter metadata yang tidak N/A
    metadata_str = ", ".join([
        f"{key}: {value}" for key, value in metadata_points.items() 
        if value != "'N/A'" and "N/A" not in value
    ])
    
    # Label yang mungkin
    possible_labels = [
        "band", "discord", "facebook", "kakao", "line", "qqmesenger", 
        "session", "signal", "slack", "snap", "teams", "telegram", 
        "viber", "wechat", "whatsapp", "wire"
    ]
    labels_str = ", ".join(possible_labels)

    instruction = (
        f"Analyze the following video metadata and identify the EXACT matching folder label. "
        f"Metadata: {metadata_str}. "
        f"The label MUST be one of the following: ({labels_str}). "
        f"Respond ONLY with the exact label name, without any additional words, phrases, or punctuation."
    )
    
    # Dapatkan true label dari mapping
    try:
        true_label_str = id_to_name_map[entry['label_folder']].lower()
    except (TypeError, IndexError, KeyError) as e:
        logging.warning(f"Tidak dapat memetakan label ID {entry.get('label_folder', 'N/A')}: {e}")
        true_label_str = str(entry['label_folder']).lower()

    formatted_string = (
        f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n{true_label_str}<end_of_turn>"
    )
    
    return {"text": formatted_string, "true_label": true_label_str}

# --- MAIN PROGRAM ---
def main():
    # Verifikasi GPU
    logging.info("Memulai skrip inferensi Gemma 2-2B-IT.")
    if not torch.cuda.is_available():
        logging.error("CUDA tidak tersedia. QLoRA membutuhkan GPU.")
        raise SystemError("CUDA tidak tersedia.")

    print(f"CUDA tersedia: {torch.cuda.is_available()}")
    print(f"Nama Perangkat: {torch.cuda.get_device_name(0)}")
    print(f"Kemampuan Komputasi: {torch.cuda.get_device_capability(0)}")

    # Konfigurasi kuantisasi
    print("\n⚙️ Mengkonfigurasi kuantisasi 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model dan tokenizer
    model_id = "google/gemma-2b-it"
    finetuned_model_path = "./gemma-2-2b-finetuned-video-metadata/final_model"
    
    print(f"\n📥 Memuat model dari: {finetuned_model_path}")
    
    if not os.path.exists(finetuned_model_path):
        logging.error(f"Path model tidak ditemukan: {finetuned_model_path}")
        return

    try:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load fine-tuned model
        model = PeftModel.from_pretrained(
            base_model,
            finetuned_model_path,
            is_trainable=False,
        )
        model.config.use_cache = True
        model.eval()
        
        print("✅ Model berhasil dimuat.")

    except Exception as e:
        logging.error(f"Error memuat model: {e}")
        return

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("✅ Tokenizer berhasil dimuat.")
    except Exception as e:
        logging.error(f"Error memuat tokenizer: {e}")
        return

    # Load dataset
    print("\n📚 Memuat dataset...")
    csv_file_path = "fitur_output.csv"
    
    if not os.path.exists(csv_file_path):
        logging.error(f"File CSV tidak ditemukan: {csv_file_path}")
        return

    try:
        df = pd.read_csv(csv_file_path)
        
        # Validasi kolom yang diperlukan
        required_cols = [
            'Video_Bitrate', 'Video_Width(Pixels)', 'Audio_Bitrate', 
            'Video_Height(Pixels)', 'Overall_bitrate', 'Codec_ID', 
            'Format_profile', 'Writing_application', 'Video_Format_profile', 
            'Video_Format_settings', 'Video_Color_range', 
            'Video_Color_primaries', 'Video_Transfer_characteristics', 
            'label_folder'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"Kolom tidak ditemukan: {missing_cols}")
            return
            
        if df.empty:
            logging.error("Dataset kosong.")
            return
            
        print(f"Dataset dimuat: {len(df)} baris, {len(df.columns)} kolom")
        
        # Tampilkan info dataset
        print(f"\n📋 INFORMASI DATASET:")
        print(f"   Jumlah sampel: {len(df)}")
        print(f"   Kolom tersedia: {len(df.columns)}")
        
        # Tampilkan distribusi label
        label_dist = df['label_folder'].value_counts().sort_index()
        print(f"   Distribusi label:")
        for label, count in label_dist.items():
            print(f"      {label}: {count} sampel")
        
    except Exception as e:
        logging.error(f"Error membaca CSV: {e}")
        return

    # Proses dataset
    try:
        full_dataset = Dataset.from_pandas(df)
        
        # Setup ClassLabel
        unique_labels = sorted(df['label_folder'].unique())
        features = full_dataset.features.copy()
        features['label_folder'] = ClassLabel(names=unique_labels)
        full_dataset = full_dataset.cast(features)
        
        # Buat mapping ID ke nama
        label_names = full_dataset.features['label_folder'].names
        id_to_name_map = {i: name for i, name in enumerate(label_names)}
        
        print(f"Label yang tersedia: {label_names}")
        print(f"Total label unik: {len(label_names)}")
        
    except Exception as e:
        logging.error(f"Error memproses dataset: {e}")
        return

    # Split dataset
    try:
        splits = full_dataset.train_test_split(
            test_size=0.2,
            seed=42,
            stratify_by_column="label_folder"
        )
        
        # Format test dataset
        test_dataset = splits['test'].map(
            lambda entry: format_prompt(entry, id_to_name_map),
            remove_columns=splits['test'].column_names,
            desc="Formatting Test Data"
        )
        
        print(f"Data training: {len(splits['train'])} sampel")
        print(f"Data testing: {len(test_dataset)} sampel")
        
        # Verifikasi distribusi label di test set
        test_label_counts = {}
        for example in test_dataset:
            label = example['true_label']
            test_label_counts[label] = test_label_counts.get(label, 0) + 1
        
        print(f"\n📊 Distribusi label di test set:")
        for label in sorted(test_label_counts.keys()):
            count = test_label_counts[label]
            percentage = (count / len(test_dataset)) * 100
            print(f"   {label}: {count} sampel ({percentage:.1f}%)")
        
    except Exception as e:
        logging.error(f"Error dalam pembagian data: {e}")
        return

    # Analisis panjang token
    print("\n🔍 Menganalisis panjang token...")
    sample_size = min(len(test_dataset), 100)  # Kurangi sample untuk efisiensi
    sampled_test = test_dataset.select(range(sample_size))
    
    token_lengths = []
    for example in tqdm(sampled_test, desc="Analyzing token lengths"):
        try:
            tokens = tokenizer(example['text'], add_special_tokens=True)
            token_lengths.append(len(tokens.input_ids))
        except Exception as e:
            logging.warning(f"Error tokenizing: {e}")
            continue
    
    if token_lengths:
        min_len = min(token_lengths)
        max_len = max(token_lengths)
        mean_len = np.mean(token_lengths)
        p95_len = np.percentile(token_lengths, 95)
        p99_len = np.percentile(token_lengths, 99)
        
        print(f"📏 STATISTIK PANJANG TOKEN:")
        print(f"   Min length: {min_len}")
        print(f"   Max length: {max_len}")
        print(f"   Mean length: {mean_len:.2f}")
        print(f"   95th percentile: {p95_len:.2f}")
        print(f"   99th percentile: {p99_len:.2f}")
        
        max_seq_length = min(int(p95_len * 1.1), 1024)
        print(f"   📌 Selected max_seq_length: {max_seq_length}")
    else:
        max_seq_length = 512
        print(f"⚠️ Menggunakan default max_seq_length: {max_seq_length}")

    # Evaluasi model
    print("\n" + "="*60)
    print("📈 MEMULAI EVALUASI MODEL")
    print("="*60)
    model.eval()
    
    true_labels = []
    predicted_labels = []
    possible_labels = [name.lower() for name in label_names]
    
    print(f"🔍 Mengevaluasi {len(test_dataset)} sampel...")
    print("📋 Hasil prediksi per sampel:")
    print("-"*70)
    
    for i, example in enumerate(tqdm(test_dataset, desc="Evaluating", ncols=100)):
        try:
            full_text = example['text']
            true_label = example['true_label']
            
            # Ekstrak prompt
            prompt_marker = "<start_of_turn>model\n"
            if prompt_marker not in full_text:
                logging.warning(f"Prompt marker tidak ditemukan pada sampel {i}")
                continue
                
            prompt_text = full_text.split(prompt_marker)[0] + prompt_marker
            
            # Tokenize dan generate
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                )
            
            # Decode hasil
            generated_text = tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):],
                skip_special_tokens=True
            ).strip()
            
            # Bersihkan prediksi
            predicted_raw = generated_text.split("<end_of_turn>")[0].strip()
            predicted_clean = clean_label_string(predicted_raw, possible_labels)
            true_clean = clean_label_string(true_label, possible_labels)
            
            if not predicted_clean or not true_clean:
                logging.warning(f"Label kosong setelah pembersihan pada sampel {i}")
                continue
                
            true_labels.append(true_clean)
            predicted_labels.append(predicted_clean)
            
            # Tampilkan semua hasil prediksi
            match_status = "✅ MATCH" if true_clean == predicted_clean else "❌ MISMATCH"
            print(f"Sampel {i+1:3d}: True='{true_clean:12s}' | Pred='{predicted_clean:12s}' | {match_status}")
            
            # Tampilkan raw prediction untuk debugging mismatch
            if true_clean != predicted_clean:
                print(f"         Raw pred: '{predicted_raw}'")
                
        except Exception as e:
            logging.error(f"Error pada sampel {i}: {e}")
            continue

    # Tampilkan hasil
    print("\n" + "="*80)
    print("📊 HASIL EVALUASI LENGKAP")
    print("="*80)
    
    if true_labels and predicted_labels:
        try:
            accuracy = accuracy_score(true_labels, predicted_labels)
            print(f"\n🎯 AKURASI KESELURUHAN: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Hitung statistik tambahan
            total_samples = len(true_labels)
            correct_predictions = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
            incorrect_predictions = total_samples - correct_predictions
            
            print(f"📈 STATISTIK PREDIKSI:")
            print(f"   Total sampel: {total_samples}")
            print(f"   Prediksi benar: {correct_predictions}")
            print(f"   Prediksi salah: {incorrect_predictions}")
            
            # Classification report dengan format yang lebih baik
            all_labels = sorted(set(true_labels + predicted_labels))
            report = classification_report(
                true_labels,
                predicted_labels,
                labels=all_labels,
                zero_division=0,
                output_dict=True
            )
            
            print(f"\n📋 CLASSIFICATION REPORT DETAIL:")
            print("-"*80)
            print(f"{'Label':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
            print("-"*80)
            
            for label in all_labels:
                if label in report:
                    metrics = report[label]
                    print(f"{label:<12} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                          f"{metrics['f1-score']:<10.4f} {int(metrics['support']):<8}")
            
            # Macro dan weighted averages
            print("-"*80)
            macro_avg = report['macro avg']
            weighted_avg = report['weighted avg']
            print(f"{'macro avg':<12} {macro_avg['precision']:<10.4f} {macro_avg['recall']:<10.4f} "
                  f"{macro_avg['f1-score']:<10.4f} {int(macro_avg['support']):<8}")
            print(f"{'weighted avg':<12} {weighted_avg['precision']:<10.4f} {weighted_avg['recall']:<10.4f} "
                  f"{weighted_avg['f1-score']:<10.4f} {int(weighted_avg['support']):<8}")
            
            # Analisis error per kelas
            print(f"\n🔍 ANALISIS ERROR PER KELAS:")
            print("-"*60)
            
            # Buat confusion matrix manual untuk analisis
            error_analysis = {}
            for true_label, pred_label in zip(true_labels, predicted_labels):
                if true_label != pred_label:
                    key = f"{true_label} → {pred_label}"
                    error_analysis[key] = error_analysis.get(key, 0) + 1
            
            if error_analysis:
                sorted_errors = sorted(error_analysis.items(), key=lambda x: x[1], reverse=True)
                print("Error paling sering (True → Predicted):")
                for error_type, count in sorted_errors[:10]:  # Top 10 errors
                    print(f"   {error_type:<20}: {count} kali")
            else:
                print("🎉 Tidak ada error! Semua prediksi benar!")
            
            # Distribusi prediksi vs true labels
            pred_counts = pd.Series(predicted_labels).value_counts().sort_index()
            true_counts = pd.Series(true_labels).value_counts().sort_index()
            
            print(f"\n📊 DISTRIBUSI LABEL:")
            print("-"*50)
            print(f"{'Label':<12} {'True Count':<12} {'Pred Count':<12} {'Difference':<12}")
            print("-"*50)
            
            all_unique_labels = sorted(set(list(pred_counts.index) + list(true_counts.index)))
            for label in all_unique_labels:
                true_count = true_counts.get(label, 0)
                pred_count = pred_counts.get(label, 0)
                diff = pred_count - true_count
                diff_str = f"{diff:+d}" if diff != 0 else "0"
                print(f"{label:<12} {true_count:<12} {pred_count:<12} {diff_str:<12}")
            
            # Confidence analysis (jika perlu)
            print(f"\n🎯 RINGKASAN PERFORMA:")
            print("-"*40)
            best_f1_labels = []
            worst_f1_labels = []
            
            for label in all_labels:
                if label in report and report[label]['support'] > 0:
                    f1_score = report[label]['f1-score']
                    if f1_score >= 0.9:
                        best_f1_labels.append((label, f1_score))
                    elif f1_score < 0.5:
                        worst_f1_labels.append((label, f1_score))
            
            if best_f1_labels:
                print("🏆 Label dengan performa terbaik (F1 ≥ 0.9):")
                for label, f1 in sorted(best_f1_labels, key=lambda x: x[1], reverse=True):
                    print(f"   {label}: {f1:.4f}")
            
            if worst_f1_labels:
                print("⚠️  Label dengan performa rendah (F1 < 0.5):")
                for label, f1 in sorted(worst_f1_labels, key=lambda x: x[1]):
                    print(f"   {label}: {f1:.4f}")
            
        except Exception as e:
            logging.error(f"Error dalam perhitungan metrik: {e}")
            print(f"❌ Error dalam perhitungan metrik: {e}")
    else:
        print("❌ Tidak ada data untuk evaluasi")

    print("\n" + "="*80)
    print("✅ PROGRAM SELESAI")
    print("="*80)
    
    if true_labels and predicted_labels:
        final_accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"🏆 AKURASI FINAL: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        print(f"📊 Total sampel dievaluasi: {len(true_labels)}")
        print(f"✅ Prediksi benar: {sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)}")
        print(f"❌ Prediksi salah: {sum(1 for t, p in zip(true_labels, predicted_labels) if t != p)}")
    
    print("\n🎯 Evaluasi selesai! Lihat hasil detail di atas.")
    print("="*80)

if __name__ == '__main__':
    main()