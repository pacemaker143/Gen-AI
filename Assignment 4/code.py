"""
Assignment 4: Question-Answering Chatbot using Pre-trained Language Model
=========================================================================
Build a simple QA chatbot using Google Gemini (pre-trained LLM).
Demonstrates:
  1. Basic question answering
  2. Context-based QA (reading comprehension)
  3. Multi-turn conversation
  4. Domain-specific QA
"""

import os
import sys
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
    model = genai.GenerativeModel("gemini-2.5-flash")

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"\nPrompt:\n{prompt}\n")

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
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


# =============================================================================
# PART 1: Basic Question Answering
# =============================================================================
# The chatbot answers general knowledge questions directly.

def basic_qa():
    questions = [
        "What is machine learning?",
        "Explain the difference between supervised and unsupervised learning.",
        "What are the main components of a neural network?",
    ]

    results = []
    for i, q in enumerate(questions, 1):
        result = call_gemini(
            f"Answer the following question concisely in 2-3 sentences.\n\nQuestion: {q}\n\nAnswer:",
            f"PART 1 - Basic QA [{i}/{len(questions)}]"
        )
        results.append(result)
    return results


# =============================================================================
# PART 2: Context-Based QA (Reading Comprehension)
# =============================================================================
# Given a passage, the chatbot answers questions based on it.

CONTEXT_PASSAGE = """
Artificial Intelligence (AI) has transformed numerous industries since its inception.
In healthcare, AI algorithms can analyse medical images to detect diseases like cancer
at early stages with accuracy rivalling human doctors. In finance, AI-powered trading
systems process millions of data points per second to make investment decisions.
The transportation sector has seen the rise of autonomous vehicles, which use a
combination of computer vision, sensor fusion, and deep learning to navigate roads.
In education, AI tutors provide personalised learning experiences, adapting to each
student's pace and learning style. However, the rapid advancement of AI has also
raised ethical concerns about job displacement, privacy, and algorithmic bias.
Researchers and policymakers are working together to establish guidelines that
ensure AI development benefits humanity while minimising potential harms.
""".strip()


def context_based_qa():
    questions = [
        "How is AI used in healthcare according to the passage?",
        "What ethical concerns are mentioned about AI?",
    ]

    results = []
    for i, q in enumerate(questions, 1):
        prompt = (
            f"Read the following passage and answer the question based ONLY on the information provided.\n\n"
            f"Passage:\n{CONTEXT_PASSAGE}\n\n"
            f"Question: {q}\n\n"
            f"Answer:"
        )
        result = call_gemini(prompt, f"PART 2 - Context-Based QA [{i}/{len(questions)}]")
        results.append(result)
    return results


# =============================================================================
# PART 3: Multi-Turn Conversation
# =============================================================================
# Simulates a conversation where context is maintained across turns.

def multi_turn_conversation():
    conversation_turns = [
        "What is Python programming language?",
        "What are its main advantages?",
        "How is it different from Java?",
    ]

    history = []
    results = []

    for i, user_msg in enumerate(conversation_turns, 1):
        # Build conversation context from history
        conv_context = ""
        for turn in history:
            conv_context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

        prompt = (
            f"You are a helpful QA chatbot. Continue the conversation naturally.\n\n"
            f"{conv_context}"
            f"User: {user_msg}\nAssistant:"
        )

        result = call_gemini(prompt, f"PART 3 - Multi-Turn [{i}/{len(conversation_turns)}]")
        results.append(result)

        history.append({"user": user_msg, "assistant": result["response"][:300]})

    return results


# =============================================================================
# PART 4: Domain-Specific QA (Science)
# =============================================================================
# The chatbot is given a specific role/domain to answer within.

def domain_specific_qa():
    questions = [
        "Why is the sky blue?",
        "How do vaccines work?",
    ]

    results = []
    for i, q in enumerate(questions, 1):
        prompt = (
            "You are a science teacher explaining concepts to a high school student. "
            "Give clear, accurate answers with simple examples.\n\n"
            f"Student's question: {q}\n\n"
            f"Teacher's answer:"
        )
        result = call_gemini(prompt, f"PART 4 - Domain-Specific QA [{i}/{len(questions)}]")
        results.append(result)
    return results


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(all_results):
    lines = [
        "=" * 70,
        "  ASSIGNMENT 4 - QA CHATBOT USING PRE-TRAINED LANGUAGE MODEL",
        "  Comprehensive Report",
        "=" * 70,
        f"\n  Date  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Model : gemini-2.5-flash",
        f"  Total Questions : {len(all_results)}",
        "",
    ]

    for r in all_results:
        lines.append("-" * 70)
        lines.append(f"  {r['label']}")
        lines.append("-" * 70)
        lines.append(f"\n  [Prompt]\n{r['prompt']}\n")
        lines.append(f"  [Response] ({r['time_sec']}s)\n{r['response']}\n")

    lines += [
        "=" * 70,
        "  ANALYSIS",
        "=" * 70,
        "",
        "  Part 1 (Basic QA):",
        "    The model answers general questions accurately and concisely.",
        "    Pre-trained knowledge covers a wide range of topics.",
        "",
        "  Part 2 (Context-Based QA):",
        "    When given a passage, the model extracts relevant information",
        "    and answers questions grounded in the provided text.",
        "",
        "  Part 3 (Multi-Turn Conversation):",
        "    By maintaining conversation history in the prompt, the model",
        "    produces coherent follow-up responses that reference earlier turns.",
        "",
        "  Part 4 (Domain-Specific QA):",
        "    Role prompting (e.g., 'You are a science teacher') effectively",
        "    adjusts the model's tone and explanation level.",
        "",
        "  Key Takeaway:",
        "    Pre-trained LLMs serve as powerful QA engines out of the box.",
        "    Performance improves with context, conversation history, and",
        "    role-based prompting — no fine-tuning required for basic QA.",
        "",
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    report_path = os.path.join(OUTPUT_DIR, "generation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Report saved: {report_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  ASSIGNMENT 4 - QA CHATBOT USING PRE-TRAINED LANGUAGE MODEL")
    print("  Basic QA | Context QA | Multi-Turn | Domain-Specific")
    print("=" * 70)

    if not GOOGLE_API_KEY:
        print("\n[ERROR] GOOGLE_API_KEY not found.")
        print("  Create a .env file in the Assignment 4 folder with:")
        print("  GOOGLE_API_KEY=your_key_here")
        return

    print(f"\n  API Key : {GOOGLE_API_KEY[:8]}...{GOOGLE_API_KEY[-4:]}")
    print(f"  Output  : {OUTPUT_DIR}")

    all_results = []

    # Part 1
    all_results.extend(basic_qa())

    # Part 2
    all_results.extend(context_based_qa())

    # Part 3
    all_results.extend(multi_turn_conversation())

    # Part 4
    all_results.extend(domain_specific_qa())

    # Save JSON
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results JSON saved: {results_path}")

    # Generate report
    generate_report(all_results)

    print("\n" + "=" * 70)
    print("  ASSIGNMENT 4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
