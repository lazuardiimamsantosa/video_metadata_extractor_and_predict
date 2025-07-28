# -*- coding: utf-8 -*-
"""
Skrip Fine-Tuning Gemma 2-2B-IT untuk Klasifikasi Metadata Video menggunakan QLoRA.
Versi ini berisi perbaikan lengkap untuk masalah stratifikasi dan optimasi VRAM.
"""

# --- Impor Pustaka ---
import torch
import pandas as pd
import numpy as np
import logging
import os # Untuk memeriksa keberadaan file

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset, ClassLabel, Features, Value
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm # Untuk progress bar

# Konfigurasi logging untuk Transformers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("transformers").setLevel(logging.WARNING) # Kurangi verbositas Transformers

# --- Fungsi untuk memformat prompt ---
# Penting: Fungsi ini harus didefinisikan di luar atau di atas blok if __name__ == '__main__':
# agar dapat diakses dengan benar oleh proses multiprocessing (jika dataloader_num_workers > 0)
def format_prompt(entry):
    """
    Memformat entri data metadata menjadi format prompt untuk model Gemma.
    """
    # Kolom yang dianggap sangat penting untuk klasifikasi
    metadata_points = {
        "Video Bitrate": f"{entry.get('Video_Bitrate', 'N/A')} kbps",
        "Resolution": f"{entry.get('Video_Width(Pixels)', 'N/A')}x{entry.get('Video_Height(Pixels)', 'N/A')}",
        "Overall Bitrate": f"{entry.get('Overall_bitrate', 'N/A')} kbps",
        "Codec ID": f"'{entry.get('Codec_ID', 'N/A')}'",
        "Format Profile": f"'{entry.get('Format_profile', 'N/A')}'",
        "Video Format Profile": f"'{entry.get('Video_Format_profile', 'N/A')}'",
        "Video Format Settings": f"'{entry.get('Video_Format_settings', 'N/A')}'", # Ditambahkan
        "Video Color Range": f"'{entry.get('Video_Color_range', 'N/A')}'",
        "Video Color Primaries": f"'{entry.get('Video_Color_primaries', 'N/A')}'", # Ditambahkan
        "Video Transfer Characteristics": f"'{entry.get('Video_Transfer_characteristics', 'N/A')}'", # Ditambahkan
        "Audio Bitrate": f"{entry.get('Audio_Bitrate', 'N/A')} kbps", # Ditambahkan
    }
    
    metadata_str = ", ".join([f"{key}: {value}" for key, value in metadata_points.items() if value != "'N/A'" and "N/A" not in value])
    
    # Kumpulan label yang mungkin (sesuaikan jika ada lebih banyak atau lebih sedikit)
    possible_labels = [
        "band", "discord", "facebook", "kakao", "line", "qqmesenger", 
        "session", "signal", "slack", "snap", "teams", "telegram", 
        "viber", "wechat", "whatsapp", "wire"
    ]
    labels_str = ", ".join(possible_labels)

    instruction = (
        f"Analyze the following video metadata and determine the correct classification label. "
        f"Metadata: {metadata_str}. "
        f"What is the corresponding folder label for this video ({labels_str})?"
    )
    
    # Mengambil label dari dataset.features jika sudah dikonversi ke ClassLabel
    # Jika belum, asumsikan 'label_folder' adalah string
    label_value = entry['label_folder']
    if isinstance(label_value, int):
        label_folder_str = format_prompt.label_map.int2str(label_value).lower() # Tambahkan .lower()
    else:
        label_folder_str = str(label_value).lower() # Tambahkan .lower()

    formatted_string = (
        f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n{label_folder_str}<end_of_turn>"
    )
    return {"text": formatted_string}

# --- MULAI BLOK UTAMA ---
if __name__ == '__main__':
    # --- 0. Verifikasi Lingkungan GPU ---
    logging.info("Memulai skrip fine-tuning Gemma 2-2B-IT.")
    if not torch.cuda.is_available():
        logging.error("❌ CUDA tidak tersedia. QLoRA membutuhkan GPU untuk berjalan.")
        raise SystemError("CUDA tidak tersedia. QLoRA membutuhkan GPU untuk berjalan.")

    logging.info(f"✅ CUDA tersedia: {torch.cuda.is_available()}")
    logging.info(f"   Nama Perangkat: {torch.cuda.get_device_name(0)}")
    logging.info(f"   Kemampuan Komputasi: {torch.cuda.get_device_capability(0)}")

    # --- 1. Konfigurasi Kuantisasi (4-bit) ---
    logging.info("\n⚙️ Mengkonfigurasi kuantisasi 4-bit (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # --- 2. Muat Model Dasar dan Tokenizer ---
    model_id = "google/gemma-2b-it"
    logging.info(f"\n📥 Memuat model dasar: {model_id} dengan kuantisasi...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            # local_files_only=True, # Nonaktifkan untuk memungkinkan unduhan jika tidak ada
        )
        logging.info(f"✅ Model {model_id} berhasil dimuat.")
    except Exception as e:
        logging.error(f"❌ Error saat memuat model {model_id}: {e}")
        logging.info("Pastikan Anda memiliki koneksi internet atau model sudah di-cache.")
        exit()
    
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    logging.info("📥 Memuat tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            # local_files_only=True, # Nonaktifkan untuk memungkinkan unduhan jika tidak ada
        )
        logging.info(f"✅ Tokenizer {model_id} berhasil dimuat.")
    except Exception as e:
        logging.error(f"❌ Error saat memuat tokenizer {model_id}: {e}")
        logging.info("Pastikan Anda memiliki koneksi internet atau tokenizer sudah di-cache.")
        exit()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 3. Konfigurasi LoRA ---
    logging.info("🛠️ Mengkonfigurasi adapter LoRA...")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    logging.info("\n📊 Parameter yang dapat dilatih setelah menerapkan LoRA:")
    model.print_trainable_parameters()

    # --- 4. Memuat Dataset dari CSV ---
    logging.info("\n📚 Memuat dataset dari CSV...")
    csv_file_path = "fitur_output.csv" # Pastikan nama file ini sesuai

    if not os.path.exists(csv_file_path):
        logging.error(f"❌ Error: File CSV '{csv_file_path}' tidak ditemukan di direktori saat ini: {os.getcwd()}")
        exit()

    try:
        df = pd.read_csv(csv_file_path)
        # Validasi kolom yang diperlukan
        required_cols = ['Video_Bitrate', 'Video_Width(Pixels)', 'Audio_Bitrate', 'Video_Height(Pixels)', 
                         'Overall_bitrate', 'Codec_ID', 'Format_profile', 'Writing_application', 
                         'Video_Format_profile', 'Video_Format_settings', 'Video_Color_range', 
                         'Video_Color_primaries', 'Video_Transfer_characteristics', 'label_folder']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"❌ Kolom yang dibutuhkan tidak ditemukan dalam file CSV: {', '.join(missing_cols)}")
            exit()

        if df.empty:
            logging.error("❌ File CSV kosong. Tidak ada data untuk pelatihan.")
            exit()

    except pd.errors.EmptyDataError:
        logging.error(f"❌ Error: File CSV '{csv_file_path}' kosong.")
        exit()
    except Exception as e:
        logging.error(f"❌ Error saat membaca CSV: {e}")
        exit()

    full_dataset = Dataset.from_pandas(df)
    
    # Peta label untuk fungsi format_prompt (agar bisa diakses oleh map())
    unique_labels = sorted(list(set(full_dataset['label_folder'])))
    num_classes = len(unique_labels)

    inferred_features = {
        col: Value(dtype='string') if df[col].dtype == 'object' else full_dataset.features[col]
        for col in full_dataset.column_names
    }
    inferred_features['label_folder'] = ClassLabel(names=unique_labels)
    new_features = Features(inferred_features)

    full_dataset = full_dataset.cast(new_features)
    # Penting: Lampirkan label_map ke fungsi format_prompt
    format_prompt.label_map = full_dataset.features['label_folder']

    # --- 5. Pembagian Data dan Pemformatan ---
    logging.info("\n🔪 Membagi dataset sebelum pemformatan untuk stratifikasi...")

    train_test_splits = full_dataset.train_test_split(
        test_size=0.2,
        seed=42,
        stratify_by_column="label_folder"
    )

    logging.info("📜 Menerapkan pemformatan prompt ke set pelatihan dan pengujian...")
    train_dataset = train_test_splits['train'].map(
        format_prompt,
        remove_columns=train_test_splits['train'].column_names,
        desc="Formatting Train Data"
    )
    test_dataset = train_test_splits['test'].map(
        format_prompt,
        remove_columns=train_test_splits['test'].column_names,
        desc="Formatting Test Data"
    )

    logging.info(f"   Jumlah data pelatihan: {len(train_dataset)}")
    logging.info(f"   Jumlah data pengujian: {len(test_dataset)}")
    logging.info(f"\n✅ Proses pembagian dan pemformatan data selesai.")

    # --- Analisis Panjang Token untuk max_seq_length optimal ---
    logging.info("\n🔍 Menganalisis panjang token untuk menentukan max_seq_length optimal...")
    
    # Batasi analisis pada sampel kecil jika dataset terlalu besar
    sample_size = min(len(train_dataset), 1000) # Batasi 1000 sampel untuk kecepatan
    sampled_train = train_dataset.select(range(sample_size)) if len(train_dataset) > sample_size else train_dataset
    sampled_test = test_dataset.select(range(sample_size)) if len(test_dataset) > sample_size else test_dataset

    train_lengths = [len(tokenizer(example['text'], add_special_tokens=True).input_ids) for example in tqdm(sampled_train, desc="Calculating Train Token Lengths")]
    test_lengths = [len(tokenizer(example['text'], add_special_tokens=True).input_ids) for example in tqdm(sampled_test, desc="Calculating Test Token Lengths")]

    all_lengths = train_lengths + test_lengths

    if all_lengths:
        min_len = np.min(all_lengths)
        mean_len = np.mean(all_lengths)
        max_len = np.max(all_lengths)
        p95_len = np.percentile(all_lengths, 95)
        p99_len = np.percentile(all_lengths, 99)
        
        logging.info(f"   Panjang token min: {min_len}")
        logging.info(f"   Panjang token rata-rata: {mean_len:.2f}")
        logging.info(f"   Panjang token maks: {max_len}")
        logging.info(f"   Panjang token persentil ke-95: {p95_len:.2f}")
        logging.info(f"   Panjang token persentil ke-99: {p99_len:.2f}")
        
        # Tambahkan sedikit buffer dan batasi maksimal 1024 atau 2048 tergantung GPU
        optimal_max_seq_length = int(p99_len * 1.1) # Tambah 10% buffer
        final_max_seq_length = min(optimal_max_seq_length, 1024) # Batasi maksimal 1024 atau 2048 jika VRAM cukup
        logging.info(f"   Menggunakan max_seq_length yang dioptimalkan: {final_max_seq_length}")
    else:
        final_max_seq_length = 512
        logging.warning(f"   Dataset kosong setelah sampling, menggunakan max_seq_length default: {final_max_seq_length}")


    # --- 6. Konfigurasi Argumen Pelatihan ---
    logging.info("\n📝 Mengkonfigurasi argumen pelatihan...")
    training_arguments = TrainingArguments(
        output_dir="./gemma-2-2b-finetuned-video-metadata",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=25,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        bf16=True, # Menggunakan bfloat16 untuk performa dan memori
        tf32=False, # Pertimbangkan tf32=True jika GPU Anda mendukungnya (misal: A100, H100)
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=False,
        dataloader_num_workers=0, # Set ke 0 untuk menghindari masalah multiprocessing di Windows
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        disable_tqdm=False, # Aktifkan tqdm untuk melihat progress
    )

    # --- 7. Inisialisasi dan Jalankan SFTTrainer ---
    logging.info("\n🚀 Memulai pelatihan dengan SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=lora_config,
        args=training_arguments,
        tokenizer=tokenizer,
        max_seq_length=final_max_seq_length,
        dataset_text_field="text",
        packing=False, # Set packing=True jika Anda ingin mengemas beberapa contoh ke dalam satu input
    )

    trainer.train()

    logging.info("\n✅ Pelatihan selesai.")
    logging.info("💾 Menyimpan model terakhir...")
    final_model_path = f"{training_arguments.output_dir}/final_model"
    trainer.save_model(final_model_path)
    logging.info(f"   Model disimpan di: {final_model_path}")

    # --- 8. Evaluasi Model pada Test Set ---
    logging.info("\n--- 📈 Melakukan Evaluasi pada Test Set ---")
    eval_model = trainer.model
    eval_model.eval()

    true_labels = []
    predicted_labels = []

    for i, example in enumerate(tqdm(test_dataset, desc="Evaluasi Model")):
        full_text = example['text']
        prompt_end_marker = "<start_of_turn>model\n"
        
        # Mengekstrak prompt dan true_label
        if prompt_end_marker in full_text:
            prompt_text = full_text.split(prompt_end_marker)[0] + prompt_end_marker
            try:
                true_label_raw = full_text.split(prompt_end_marker)[1].split("<end_of_turn>")[0].strip()
            except IndexError:
                logging.warning(f"⚠️ Peringatan: Tidak dapat mengekstrak true_label dari contoh {i}. Melanjutkan.")
                continue
        else:
            logging.warning(f"⚠️ Peringatan: Penanda prompt tidak ditemukan di contoh {i}. Melewati.")
            continue

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=final_max_seq_length).to(model.device)

        with torch.no_grad():
            outputs = eval_model.generate(
                **inputs,
                max_new_tokens=20, # Cukup untuk label singkat
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, # Gunakan greedy decoding untuk prediksi yang deterministik
                temperature=0.0, # Rendah untuk keluaran yang deterministik
                top_p=1.0,
            )

        generated_text = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True).strip()
        predicted_label_raw = generated_text.split("<end_of_turn>")[0].strip() # Pastikan hanya mengambil bagian sebelum <end_of_turn>

        true_labels.append(true_label_raw.lower())
        predicted_labels.append(predicted_label_raw.lower())

        if i < 5 or true_label_raw.lower() != predicted_label_raw.lower(): # Cetak 5 contoh pertama atau jika ada kesalahan
            logging.info(f"   Contoh {i+1}/{len(test_dataset)} | Sebenarnya: '{true_label_raw}' | Prediksi: '{predicted_label_raw}'")

    # --- 9. Tampilkan Hasil Metrik ---
    logging.info("\n\n--- 📊 Hasil Metrik Klasifikasi ---")
    if true_labels and predicted_labels:
        # Menangani kasus di mana prediksi mungkin tidak ada dalam true_labels
        all_possible_labels = sorted(list(set(true_labels + predicted_labels)))
        
        report = classification_report(
            true_labels,
            predicted_labels,
            labels=all_possible_labels, # Pastikan semua label dipertimbangkan
            zero_division=0,
            output_dict=False
        )
        print(report)

        accuracy = accuracy_score(true_labels, predicted_labels)
        logging.info(f"🎯 Akurasi Keseluruhan: {accuracy:.4f}")
    else:
        logging.warning("❌ Tidak dapat menghasilkan metrik, tidak ada label yang dievaluasi.")

    logging.info("Program selesai.")