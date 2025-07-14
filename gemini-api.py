import os
import json
import random
from google import genai
from google.genai import types
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# --- Setup Gemini client ---
GEMINI_API_KEY = "INSERT YOUR API HERE"
if not GEMINI_API_KEY:
    raise ValueError("❌ Environment variable 'GEMINI_API_KEY' is not set.")

client = genai.Client(api_key=GEMINI_API_KEY)
model_name = "gemini-2.5-pro"

# --- Create few-shot examples ---
def create_few_shot_examples(train_data, num_examples=5):
    examples_by_class = {}
    for item in train_data:
        output_class = item['output']
        if output_class not in examples_by_class:
            examples_by_class[output_class] = []
        examples_by_class[output_class].append(item)

    selected_examples = []
    for class_name, class_items in examples_by_class.items():
        if len(selected_examples) < num_examples:
            selected_examples.append(random.choice(class_items))

    while len(selected_examples) < num_examples and len(selected_examples) < len(train_data):
        remaining_items = [item for item in train_data if item not in selected_examples]
        if remaining_items:
            selected_examples.append(random.choice(remaining_items))
        else:
            break

    return selected_examples

# --- Build prompt ---
def build_few_shot_prompt(few_shot_examples, test_input):
    instruction = (
        "Your task is to classify video metadata into one and only of the following applications: "
        "WhatsApp, Telegram, Signal, Snapchat, Facebook Messenger, Line, QQ, WeChat, Viber, Wire, Slack, Teams, KakaoTalk, Band, Session.\n\n"
        "Here are some examples of correct classifications:\n\n"
    )
    for i, example in enumerate(few_shot_examples, 1):
        instruction += f"Example {i}:\n"
        instruction += f"Metadata: {example['input']}\n"
        instruction += f"Application: {example['output']}\n\n"

    instruction += f"Now classify the following metadata:\n"
    instruction += f"Metadata: {test_input}\n"
    instruction += f"Application: "
    return instruction

# --- Gemini prediction ---
def gemini_predict_few_shot(prompt: str, few_shot_examples: list) -> str:
    instruction = build_few_shot_prompt(few_shot_examples, prompt)

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=instruction)],
        )
    ]

    config = types.GenerateContentConfig(
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="text/plain",
    )

    try:
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=config,
        ):
            if chunk.text is not None:
                response_text += chunk.text

        if not response_text.strip():
            print("🔄 Attempting with non-streaming...")
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            response_text = response.text if response.text else ""

        return response_text.strip()
    except Exception as e:
        print(f"⚠️ Error during Gemini prediction: {e}")
        return "UNKNOWN"

# --- Split JSONL ---
def split_jsonl(json_path, train_path, test_path, test_size=0.2):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    with open(train_path, 'w', encoding='utf-8') as f_train:
        for item in train_data:
            f_train.write(json.dumps(item) + "\n")

    with open(test_path, 'w', encoding='utf-8') as f_test:
        for item in test_data:
            f_test.write(json.dumps(item) + "\n")

    print(f"✅ Split complete: {len(train_data)} train, {len(test_data)} test")
    return train_data, test_data

# --- Load JSONL ---
def load_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# --- Evaluate with few-shot learning ---
def evaluate_gemini_few_shot(test_data, train_data, num_examples=5):
    y_true = []
    y_pred = []

    few_shot_examples = create_few_shot_examples(train_data, num_examples)

    print(f"🎯 Menggunakan {len(few_shot_examples)} data untuk pembelajaran (few-shot):")
    for i, example in enumerate(few_shot_examples, 1):
        print(f"   🔹 Contoh {i}: {example['output']} <- {example['input'][:50]}...")
    print()

    for i, item in enumerate(test_data):
        input_text = item['input']
        true_label = item['output']

        print(f"🔎 [{i+1}] Menebak aplikasi dari metadata berikut:")
        print(f"     {input_text[:80]}...")

        prediction = gemini_predict_few_shot(input_text, few_shot_examples)

        print(f"📌 Predicted: {prediction} | True: {true_label}\n")

        y_true.append(true_label.strip().lower())
        y_pred.append(prediction.strip().lower())

    acc = accuracy_score(y_true, y_pred)
    print(f"🎯 Akurasi Gemini (Few-shot): {acc:.4f}")
    return y_true, y_pred

# --- MAIN ---
if __name__ == "__main__":
    jsonl_path = "data_gemini_format.json"
    train_path = "train.json"
    test_path = "test.json"

    print("📂 Sedang membagi data menjadi train.json dan test.json...")
    train_data, test_data = split_jsonl(jsonl_path, train_path, test_path, test_size=0.2)

    print("\n📘 Sedang mempelajari data dari train.json (few-shot learning)...")
    # Sudah otomatis dipelajari di evaluate function

    print("\n🤖 Sekarang melakukan prediksi pada data test.json...")
    y_true, y_pred = evaluate_gemini_few_shot(test_data, train_data, num_examples=5)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT:")
    print("=" * 50)
    print(classification_report(y_true, y_pred, zero_division=0))

    # Simpan hasil prediksi
    results = []
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        results.append({
            'index': i,
            'true_label': true_label,
            'predicted_label': pred_label,
            'correct': true_label == pred_label
        })

    with open('prediction_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Prediction results saved to 'prediction_results.json'")
