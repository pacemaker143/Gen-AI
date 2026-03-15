"""
Assignment 10: Text-to-Image Generation using Transformer-based Models
======================================================================
Exploring Transformer capabilities in multimodal tasks by generating
images from text prompts using Cloudflare Workers AI API.

Model: @cf/black-forest-labs/flux-1-schnell
"""

import requests
import os
import json
import base64
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────────────────────

# Load .env from the same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

API_KEY = os.getenv("API_KEY", "").strip()
ACCOUNT_ID = os.getenv("ACCOUNT_ID", "").strip()

MODEL = "@cf/black-forest-labs/flux-1-schnell"
BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{MODEL}"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Output folder inside Assignment10
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Text Prompts ────────────────────────────────────────────────────────────

PROMPTS = [
    {
        "prompt": "A futuristic city skyline at sunset with flying cars and neon lights",
        "label": "futuristic_city"
    },
    {
        "prompt": "A serene Japanese garden with cherry blossom trees and a koi pond, watercolor style",
        "label": "japanese_garden"
    },
    {
        "prompt": "An astronaut riding a horse on the surface of Mars, digital art",
        "label": "astronaut_mars"
    },
    {
        "prompt": "A steampunk mechanical owl perched on an old book, highly detailed",
        "label": "steampunk_owl"
    },
    {
        "prompt": "A cozy cabin in a snowy mountain landscape during golden hour, photorealistic",
        "label": "snowy_cabin"
    },
]


# ─── Helper Functions ────────────────────────────────────────────────────────

def generate_image(prompt: str) -> bytes | None:
    """Send a text prompt to Cloudflare Workers AI (FLUX.1-schnell) and return image bytes."""
    payload = {
        "prompt": prompt,
        "steps": 8   # max allowed for this model, higher = better quality
    }
    try:
        print(f"  -> Sending request to Cloudflare Workers AI...")
        response = requests.post(BASE_URL, headers=HEADERS, json=payload, timeout=120)

        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")

            # The API returns JSON with base64-encoded image in result.image
            if "application/json" in content_type or "json" in content_type:
                data = response.json()
                if data.get("success") and data.get("result", {}).get("image"):
                    img_b64 = data["result"]["image"]
                    img_bytes = base64.b64decode(img_b64)
                    print(f"  [OK] Image received ({len(img_bytes)} bytes)")
                    return img_bytes
                else:
                    print(f"  [FAIL] Unexpected JSON: {json.dumps(data, indent=2)[:400]}")
            elif "image" in content_type:
                # Direct binary image response
                print(f"  [OK] Image received ({len(response.content)} bytes)")
                return response.content
            else:
                # Try parsing as JSON anyway
                try:
                    data = response.json()
                    if data.get("result", {}).get("image"):
                        img_bytes = base64.b64decode(data["result"]["image"])
                        print(f"  [OK] Image decoded ({len(img_bytes)} bytes)")
                        return img_bytes
                except Exception:
                    pass
                print(f"  [FAIL] Unknown content-type: {content_type}")
                print(f"         Response: {response.text[:300]}")
        else:
            error_msg = response.text[:500]
            print(f"  [FAIL] HTTP {response.status_code}: {error_msg}")
            if response.status_code == 401:
                print("  [!] Authentication failed - API_KEY is invalid or expired")
                print("      Generate new token: https://dash.cloudflare.com/profile/api-tokens")
            elif response.status_code == 403:
                print("  [!] Forbidden - your API token may lack Workers AI permissions")
            elif response.status_code == 404:
                print("  [!] Not found - ACCOUNT_ID is likely incorrect")

    except requests.exceptions.Timeout:
        print("  [FAIL] Request timed out (120s)")
    except requests.exceptions.ConnectionError:
        print("  [FAIL] Connection error - check internet connectivity")
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")

    return None


def save_image(image_bytes: bytes, label: str) -> str:
    """Save raw image bytes to a PNG file."""
    filepath = os.path.join(OUTPUT_DIR, f"{label}.png")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.save(filepath, "PNG")
    except Exception:
        with open(filepath, "wb") as f:
            f.write(image_bytes)
    print(f"  -> Saved: {filepath}")
    return filepath


def create_summary_grid(results: list):
    """Create a grid visualization of all generated images."""
    successful = [r for r in results if r["status"] == "success"]
    if not successful:
        print("\nNo images to display in grid.")
        return

    n = len(successful)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, result in enumerate(successful):
        img = Image.open(result["filepath"])
        axes[i].imshow(img)
        title = result["prompt"][:60] + ("..." if len(result["prompt"]) > 60 else "")
        axes[i].set_title(title, fontsize=10, fontweight="bold", wrap=True)
        axes[i].axis("off")

    for j in range(len(successful), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Text-to-Image Generation Results\n(FLUX.1-schnell via Cloudflare Workers AI)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, "summary_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[OK] Summary grid saved: {grid_path}")


def generate_report(results: list):
    """Generate a comprehensive text report."""
    report_lines = [
        "=" * 70,
        "  TEXT-TO-IMAGE GENERATION REPORT",
        "  Transformer Capabilities in Multimodal Tasks",
        "=" * 70,
        f"\n  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Model      : {MODEL}",
        f"  API        : Cloudflare Workers AI",
        f"  Prompts    : {len(results)}",
        f"  Successful : {sum(1 for r in results if r['status'] == 'success')}",
        f"  Failed     : {sum(1 for r in results if r['status'] == 'failed')}",
        "\n" + "-" * 70,
        "  RESULTS",
        "-" * 70,
    ]

    for i, r in enumerate(results, 1):
        report_lines.append(f"\n  [{i}] {r['label']}")
        report_lines.append(f"      Prompt : {r['prompt']}")
        report_lines.append(f"      Status : {r['status'].upper()}")
        if r["status"] == "success":
            report_lines.append(f"      File   : {r['filepath']}")

    report_lines += [
        "\n" + "-" * 70,
        "  HOW TRANSFORMERS ENABLE TEXT-TO-IMAGE GENERATION",
        "-" * 70,
        """
  FLUX.1-schnell uses a Transformer-based architecture for image generation:

  1. TEXT ENCODING (T5 / CLIP Transformer):
     - The text prompt is tokenized and passed through text encoders
     - Self-attention layers capture semantic relationships between words
     - Output: a rich embedding vector representing the prompt meaning

  2. FLOW MATCHING (Rectified Flow Transformer):
     - Unlike traditional diffusion, FLUX uses rectified flow matching
     - A Transformer predicts the velocity field that maps noise to images
     - Cross-attention layers condition image generation on text embeddings
     - This enables faster generation with fewer steps (4-8 steps)

  3. MULTIMODAL BRIDGE:
     - Cross-attention layers allow image features to attend to text tokens
     - The model learns alignment between visual and textual concepts
     - This is the core mechanism enabling text -> image translation

  4. TRANSFORMER COMPONENTS USED:
     * Multi-Head Self-Attention  - captures relationships within each modality
     * Cross-Attention            - bridges text and image modalities
     * Positional Encoding        - maintains spatial (2D) and sequential structure
     * Layer Normalization        - stabilizes deep network training
     * Feed-Forward Networks      - non-linear transformations between layers

  5. MULTIMODAL CAPABILITIES:
     - Transformers enable cross-modal understanding (text <-> image)
     - Attention allows fine-grained control (style, objects, scene, composition)
     - Pre-trained on large-scale text-image datasets
     - FLUX.1-schnell is optimized for speed while maintaining quality
""",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)
    report_path = os.path.join(OUTPUT_DIR, "generation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n[OK] Report saved: {report_path}")


# ─── Main Execution ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TEXT-TO-IMAGE GENERATION WITH TRANSFORMERS")
    print("  Using Cloudflare Workers AI - FLUX.1-schnell")
    print("=" * 60)

    # Validate config
    if not API_KEY:
        print("\n[ERROR] API_KEY not found in .env file")
        print("  Add to .env:  API_KEY=your_cloudflare_api_token")
        return

    if not ACCOUNT_ID:
        print("\n[ERROR] ACCOUNT_ID not found in .env file")
        print("  Add to .env:  ACCOUNT_ID=your_cloudflare_account_id")
        print("")
        print("  HOW TO FIND YOUR ACCOUNT ID:")
        print("  1. Go to https://dash.cloudflare.com")
        print("  2. Click 'Workers & Pages' in the left sidebar")
        print("  3. Your Account ID is shown on the right side")
        print("     OR in the URL: https://dash.cloudflare.com/<ACCOUNT_ID>/workers")
        return

    print(f"\n  API Key    : {API_KEY[:8]}...{API_KEY[-4:]}")
    print(f"  Account ID : {ACCOUNT_ID[:8]}...{ACCOUNT_ID[-4:]}")
    print(f"  Model      : {MODEL}")
    print(f"  Output     : {OUTPUT_DIR}")

    results = []

    for i, item in enumerate(PROMPTS, 1):
        prompt = item["prompt"]
        label = item["label"]

        print(f"\n[{i}/{len(PROMPTS)}] Generating: \"{prompt}\"")

        image_bytes = generate_image(prompt)

        if image_bytes:
            filepath = save_image(image_bytes, label)
            results.append({
                "prompt": prompt,
                "label": label,
                "status": "success",
                "filepath": filepath
            })
        else:
            results.append({
                "prompt": prompt,
                "label": label,
                "status": "failed",
                "filepath": None
            })

    # Save results JSON
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results JSON saved: {results_path}")

    # Generate report
    generate_report(results)

    # Create image grid if any succeeded
    try:
        create_summary_grid(results)
    except Exception as e:
        print(f"\n[WARNING] Could not create summary grid: {e}")

    print("\n[OK] Assignment 10 complete.")


if __name__ == "__main__":
    main()
