"""
Assignment 2: Prompt Engineering Approaches
============================================
Implement and compare all major Prompt Engineering techniques:
  1. Zero-Shot Prompting
  2. Few-Shot Prompting
  3. Chain of Thought (CoT) Prompting
  4. Tree of Thought (ToT) Prompting
  5. Interview Approach Prompting

Compare and contrast all approaches, analyse their applications.
Uses Google Gemini API (gemini-2.5-flash-lite).
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# --- Configuration -----------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Gemini Helper -----------------------------------------------------------

def call_gemini(prompt, label=""):
    """Send a prompt to Gemini and return the response text."""
    import google.generativeai as genai

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"\nPrompt:\n{prompt}\n")

    # Retry up to 3 times on rate-limit (429) errors
    for attempt in range(3):
        try:
            start = time.time()
            response = model.generate_content(prompt)
            elapsed = round(time.time() - start, 2)
            text = response.text
            print(f"Response ({elapsed}s):\n{text}")
            return {"prompt": prompt, "response": text, "time_sec": elapsed, "label": label}
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise


# --- 1. Zero-Shot Prompting --------------------------------------------------
# No examples provided; the model relies purely on its pre-training.

def zero_shot():
    prompt = (
        "Classify the sentiment of the following sentence as "
        "Positive, Negative, or Neutral.\n\n"
        "Sentence: \"The new update made the app incredibly slow and frustrating to use.\"\n\n"
        "Sentiment:"
    )
    return call_gemini(prompt, "1. ZERO-SHOT PROMPTING")


# --- 2. Few-Shot Prompting ---------------------------------------------------
# A few labelled examples are provided so the model learns the pattern.

def few_shot():
    prompt = (
        "Classify the sentiment of each sentence as Positive, Negative, or Neutral.\n\n"
        "Sentence: \"I love the new design of this website!\"\n"
        "Sentiment: Positive\n\n"
        "Sentence: \"The delivery was okay, nothing special.\"\n"
        "Sentiment: Neutral\n\n"
        "Sentence: \"The food was absolutely terrible and the service was rude.\"\n"
        "Sentiment: Negative\n\n"
        "Sentence: \"The new update made the app incredibly slow and frustrating to use.\"\n"
        "Sentiment:"
    )
    return call_gemini(prompt, "2. FEW-SHOT PROMPTING")


# --- 3. Chain of Thought (CoT) Prompting -------------------------------------
# The model is asked to reason step-by-step before giving the answer.

def chain_of_thought():
    prompt = (
        "Solve the following problem step by step.\n\n"
        "Problem: A store sells notebooks for Rs.45 each. A student buys 3 notebooks "
        "and pays with a Rs.200 note. How much change does the student receive? "
        "The student then buys 2 pens at Rs.15 each from the change. "
        "How much money is left?\n\n"
        "Think through this step by step:\n"
        "Step 1: Calculate the total cost of notebooks.\n"
        "Step 2: Calculate the change from Rs.200.\n"
        "Step 3: Calculate the cost of pens.\n"
        "Step 4: Calculate the remaining money.\n\n"
        "Solve:"
    )
    return call_gemini(prompt, "3. CHAIN OF THOUGHT (CoT) PROMPTING")


def without_cot():
    """Same problem without CoT - for comparison."""
    prompt = (
        "A store sells notebooks for Rs.45 each. A student buys 3 notebooks "
        "and pays with a Rs.200 note. How much change does the student receive? "
        "The student then buys 2 pens at Rs.15 each from the change. "
        "How much money is left?\n\n"
        "Answer:"
    )
    return call_gemini(prompt, "3b. WITHOUT CoT (Direct Answer - Comparison)")


# --- 4. Tree of Thought (ToT) Prompting --------------------------------------
# Multiple reasoning paths are explored, evaluated, and the best is chosen.

def tree_of_thought():
    prompt = (
        "Problem: A small company needs to reduce its carbon footprint by 40% "
        "in 5 years. Current annual emissions are 1000 tonnes of CO2.\n\n"
        "Explore THREE different strategic paths to achieve this goal. "
        "For each path, reason through feasibility, cost, and timeline step by step. "
        "Then evaluate and recommend the best strategy.\n\n"
        "Path 1 - Renewable Energy Transition:\n"
        "Think through this approach step by step.\n\n"
        "Path 2 - Operational Efficiency & Process Optimization:\n"
        "Think through this approach step by step.\n\n"
        "Path 3 - Carbon Offsetting & Green Technology Investment:\n"
        "Think through this approach step by step.\n\n"
        "Evaluation:\n"
        "Compare all three paths and recommend the best strategy or combination. "
        "Justify your choice."
    )
    return call_gemini(prompt, "4. TREE OF THOUGHT (ToT) PROMPTING")


# --- 5. Interview Approach Prompting -----------------------------------------
# The model acts as an expert interviewer, asking clarifying questions
# before providing a recommendation.

def interview_approach():
    prompt = (
        "You are an expert software architect. I want to build a mobile app.\n\n"
        "Instead of giving a solution directly, act as an interviewer. "
        "Ask me 5 important clarifying questions to understand my requirements. "
        "Then, based on the answers I provide below, give a final recommendation.\n\n"
        "My answers (provided upfront):\n"
        "1. The app is for food delivery on a college campus.\n"
        "2. Target users are college students aged 18-25.\n"
        "3. Budget is around Rs.50,000 for an MVP.\n"
        "4. I need it on both Android and iOS.\n"
        "5. Key features: menu browsing, ordering, payment, and order tracking.\n\n"
        "First list your 5 interview questions, then map my answers to them, "
        "and finally provide your architectural recommendation."
    )
    return call_gemini(prompt, "5. INTERVIEW APPROACH PROMPTING")


# --- 6. Comparison & Analysis ------------------------------------------------

def comparison_analysis():
    prompt = (
        "You are an AI/NLP expert. Provide a detailed comparison of these five "
        "prompt engineering techniques:\n\n"
        "1. Zero-Shot Prompting\n"
        "2. Few-Shot Prompting\n"
        "3. Chain of Thought (CoT) Prompting\n"
        "4. Tree of Thought (ToT) Prompting\n"
        "5. Interview Approach Prompting\n\n"
        "Create a comparison table with these columns:\n"
        "- Technique\n"
        "- Definition (1 line)\n"
        "- Examples Needed\n"
        "- Reasoning Depth (Low / Medium / High)\n"
        "- Best Use Cases\n"
        "- Limitations\n"
        "- Accuracy on Complex Tasks (Low / Medium / High)\n\n"
        "After the table, write a paragraph on when to use each technique "
        "in real-world applications."
    )
    return call_gemini(prompt, "6. COMPARISON & ANALYSIS OF ALL APPROACHES")


# --- Report Generation -------------------------------------------------------

def generate_report(results):
    """Write a comprehensive text report from all collected results."""
    lines = [
        "=" * 70,
        "  ASSIGNMENT 2 - PROMPT ENGINEERING APPROACHES",
        "  Comprehensive Report",
        "=" * 70,
        f"\n  Date  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Model : gemini-2.5-flash-lite",
        f"  Runs  : {len(results)}",
        "",
    ]

    for r in results:
        lines.append("-" * 70)
        lines.append(f"  {r['label']}")
        lines.append("-" * 70)
        lines.append(f"\n  [Prompt]\n{r['prompt']}\n")
        lines.append(f"  [Response]  ({r['time_sec']}s)\n{r['response']}\n")

    lines += [
        "=" * 70,
        "  KEY TAKEAWAYS",
        "=" * 70,
        "",
        "  * Zero-Shot is simplest but may lack precision on nuanced tasks.",
        "  * Few-Shot improves accuracy by providing pattern examples.",
        "  * CoT forces step-by-step reasoning - ideal for math/logic.",
        "  * ToT explores multiple paths - best for open-ended/strategic problems.",
        "  * Interview Approach gathers requirements first - great for design tasks.",
        "",
        "  Zero-Shot vs Few-Shot:",
        "    - Few-Shot consistently produces more formatted, accurate answers",
        "      because the examples anchor the model's output style.",
        "    - Zero-Shot is faster to write but relies entirely on training data.",
        "",
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    report_path = os.path.join(OUTPUT_DIR, "comprehensive_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[OK] Report saved: {report_path}")
    return report_text


# --- Main --------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  ASSIGNMENT 2 - PROMPT ENGINEERING APPROACHES")
    print("  Zero-Shot | Few-Shot | CoT | ToT | Interview")
    print("=" * 70)

    if not GOOGLE_API_KEY:
        print("\n[ERROR] GOOGLE_API_KEY not found.")
        print("  Create a .env file in the Assignment 2 folder with:")
        print("  GOOGLE_API_KEY=your_key_here")
        return

    print(f"\n  API Key : {GOOGLE_API_KEY[:8]}...{GOOGLE_API_KEY[-4:]}")
    print(f"  Output  : {OUTPUT_DIR}")

    results = []

    # 1 & 2 - Zero-Shot vs Few-Shot (same task for direct comparison)
    results.append(zero_shot())
    results.append(few_shot())

    # 3 - Chain of Thought (with and without for comparison)
    results.append(chain_of_thought())
    results.append(without_cot())

    # 4 - Tree of Thought
    results.append(tree_of_thought())

    # 5 - Interview Approach
    results.append(interview_approach())

    # 6 - Comparison & Analysis
    results.append(comparison_analysis())

    # Save JSON
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results JSON saved: {results_path}")

    # Generate report
    generate_report(results)

    print("\n" + "=" * 70)
    print("  ASSIGNMENT 2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
