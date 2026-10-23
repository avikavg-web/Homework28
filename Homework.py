import base64
import os
import requests
from config import HF_API_KEY

API_URL = "https://router.huggingface.co/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json",
}

MODELS = [
    "zai-org/GLM-4.5V",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "google/gemma-3-27b-it",
]

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def data_url(b: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")


def extract_err(r: requests.Response) -> str:
    try:
        j = r.json()
        return j.get("error", {}).get("message") or str(j)
    except Exception:
        return (r.text or "").strip() or r.reason or "Request failed."


def box(title: str, lines: list[str], icon: str):
    w = max(30, len(title) + 4, *(len(x) for x in lines))
    print("\n" + "┏" + "-" * (w + 2) + "┓")
    print(f"┃ {icon} {title.ljust(w - 2)} ┃")
    print("┣" + "-" * (w + 2) + "┫")
    for x in lines:
        print(f"┃ {x.ljust(w)} ┃")
    print("┗" + "-" * (w + 2) + "┛\n")


def caption_image(image_path: str) -> str | None:
    try:
        with open(image_path, "rb") as f:
            img = f.read()
    except Exception as e:
        box("File Error", [f"Could not load: {image_path}", f"Reason: {e}"], "⚠️")
        return None

    base = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Give a short caption for this image."},
                    {"type": "image_url", "image_url": {"url": data_url(img)}},
                ],
            }
        ],
        "max_tokens": 60,
        "temperature": 0.2,
    }

    last = None
    for model in MODELS:
        payload = dict(base, model=model)
        try:
            r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        except requests.RequestException as e:
            last = f"Request failed: {e}"
            continue

        if r.status_code != 200:
            last = extract_err(r)
            continue

        try:
            d = r.json()
        except Exception:
            last = "Non-JSON response received from the API."
            continue

        cap = (d.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        if cap:
            box(
                "Image Caption Generated",
                [
                    f"🖼️ Image  : {image_path}",
                    "📝 Caption:",
                    f"   {cap}",
                ],
                "🎉",
            )
            return cap
        last = "No caption found."

    box("Caption Failed", [f"🖼️ Image  : {image_path}", f"❌ Error : {last or 'Unknown error'}"], "⚠️")
    return None


def process_image_folder():
    folder_path = input("Enter image folder path (default: .): ").strip() or "."

    if not os.path.isdir(folder_path):
        box("Folder Error", [f"Directory not found: {folder_path}"], "⚠️")
        return

    # Find valid images in the directory
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    if not image_files:
        box("No Images", [f"No valid images found in: {folder_path}"], "⚠️")
        return

    summary_file = os.path.join(folder_path, "captions_summary.txt")
    results = []

    for filename in image_files:
        full_path = os.path.join(folder_path, filename)
        caption = caption_image(full_path)
        results.append((filename, caption if caption else "FAILED"))

    # Write results to summary file
    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("Image Captioning Summary\n")
            f.write("=" * 30 + "\n\n")
            for name, cap in results:
                f.write(f"File   : {name}\n")
                f.write(f"Caption: {cap}\n\n")
        box("Summary Saved", [f"Report: {summary_file}", f"Total processed: {len(results)}"], "📄")
    except Exception as e:
        box("Save Error", [f"Could not save summary: {e}"], "⚠️")


def main():
    process_image_folder()


if __name__ == "__main__":
    main()