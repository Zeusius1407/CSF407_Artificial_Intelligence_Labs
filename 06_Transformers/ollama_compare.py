"""
CSF407 Lab 6, Part 3: run a GPT-style model locally with Ollama and compare it
with ChatGPT / Claude.

Prerequisites (Linux / Colab):
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve &                   # starts the server on http://localhost:11434
    ollama pull llama3.2:1b          # or qwen2.5:1.5b, gemma3:1b, ...

Usage:
    python3 ollama_compare.py                        # default model llama3.2:1b
    python3 ollama_compare.py --model qwen2.5:1.5b
    python3 ollama_compare.py --model llama3.2:1b --model qwen2.5:1.5b

Writes ollama_responses.md containing every prompt, each local model's answer,
its latency and tokens/second, plus empty columns in which to paste the
ChatGPT / Claude answers and score them.
"""

import argparse
import json
import time
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434"

# Prompts chosen to probe different capabilities (and known small-model weaknesses)
PROMPTS = [
    ("Factual",        "What is the capital of Australia? Answer in one word."),
    ("Explanation",    "Explain self-attention in a Transformer to a second-year CS student in 4 sentences."),
    ("Reasoning",      "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
                       "How much does the ball cost? Show your reasoning."),
    ("Arithmetic",     "What is 17 * 24? Give only the number."),
    ("Code",           "Write a C++ function that returns the length of the longest increasing "
                       "subsequence of a vector<int> in O(n log n)."),
    ("Translation",    "Translate to French: 'I love machine learning.'"),
    ("Summarisation",  "Summarise in one sentence: The Transformer architecture replaced recurrence "
                       "with attention, which allowed training to be parallelised across sequence "
                       "positions and made it easier to model long-range dependencies. It became "
                       "the foundation for BERT, GPT and most modern language models."),
    ("Hallucination",  "Summarise the main findings of the 2019 paper 'Quantum Attention Networks "
                       "for Protein Folding' by J. Smith."),   # the paper does not exist
    ("Instruction",    "List exactly three prime numbers between 20 and 40, comma-separated, nothing else."),
]


def post(path, payload, timeout=600):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def installed_models():
    with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=10) as r:
        return [m["name"] for m in json.loads(r.read())["models"]]


def ask(model, prompt, temperature=0.0, seed=42):
    t0 = time.time()
    r = post("/api/chat", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "seed": seed},
    })
    wall = time.time() - t0
    n_tok = r.get("eval_count", 0)
    gen_s = r.get("eval_duration", 0) / 1e9
    return r["message"]["content"].strip(), wall, (n_tok / gen_s if gen_s else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", help="Ollama model tag (repeatable)")
    ap.add_argument("--out", default="ollama_responses.md")
    args = ap.parse_args()
    models = args.model or ["llama3.2:1b"]

    try:
        have = installed_models()
    except urllib.error.URLError:
        raise SystemExit("Ollama server not reachable on localhost:11434 - run `ollama serve` first.")
    missing = [m for m in models if m not in have and m + ":latest" not in have]
    if missing:
        raise SystemExit(f"Models not pulled: {missing}. Run: " + "; ".join(f"ollama pull {m}" for m in missing))

    lines = ["# Local LLM (Ollama) vs ChatGPT / Claude\n",
             f"Models: {', '.join(models)}. Temperature 0, seed 42.\n"]
    for cat, prompt in PROMPTS:
        lines.append(f"\n## {cat}\n\n**Prompt:** {prompt}\n")
        for m in models:
            print(f"[{m}] {cat} ...", flush=True)
            answer, wall, tps = ask(m, prompt)
            lines.append(f"\n**{m}** ({wall:.1f}s, {tps:.1f} tok/s)\n\n```\n{answer}\n```\n")
        lines.append("\n**ChatGPT:** _paste here_\n\n**Claude:** _paste here_\n")
        lines.append("\n| model | correct? (Y/N/partial) | follows instructions? | notes |\n|---|---|---|---|\n")
        for m in models + ["ChatGPT", "Claude"]:
            lines.append(f"| {m} | | | |\n")

    with open(args.out, "w") as f:
        f.writelines(lines)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
