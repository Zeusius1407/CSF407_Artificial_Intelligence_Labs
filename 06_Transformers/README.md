# Lab 6 — Transformers

| File | What it does | Where it runs |
|---|---|---|
| `01_attention_from_scratch.ipynb` | The 6 concepts from the handout (attention, self-attention, cross-attention, multi-head attention, positional encoding, masked attention), each implemented from its equation and **verified** | anywhere (CPU, a few seconds); **already executed** |
| `02_huggingface_pipelines.ipynb` | Translation (encoder–decoder), sentiment and semantic search (encoder-only), GPT-style generation (decoder-only) | Colab or local with internet (downloads models under 1 GB) |
| `03_ollama_local_llms.ipynb` | Installs Ollama, pulls small models, runs the comparison | Colab (GPU recommended) or Linux |
| `ollama_compare.py` | Sends 9 probe prompts to local models and writes `ollama_responses.md`, with slots for ChatGPT and Claude answers | wherever `ollama serve` is running |

## Why the Transformer was introduced, and its three families

The Transformer was introduced for **machine translation**: one sequence in, another sequence out. The original model is an **encoder–decoder**:

- **Encoder**: bidirectional self-attention over the source sentence.
- **Decoder**: masked self-attention over the target produced so far, plus cross-attention to the encoder output.

Removing one half of the model gives the other two families:

| Family | Attention | Examples | Used for |
|---|---|---|---|
| Encoder–decoder | bidirectional encoder; causal + cross-attention decoder | original Transformer, T5, MarianMT | translation, sequence-to-sequence tasks |
| Encoder-only | bidirectional, no mask | BERT, DistilBERT, MiniLM | classification, semantic search, retrieval, clustering |
| Decoder-only | causal only | GPT, Llama, Qwen | generation, chat, code, summarisation, question answering |

## Part 1 — Concepts, and the checks that passed (executed)

| Concept | Key equation or idea | Verification in the notebook |
|---|---|---|
| Attention | softmax(QKᵀ/√d_k + M)V: the output is a weighted average of the values | matches `F.scaled_dot_product_attention` (max difference 2 × 10⁻⁷); rows of the weights sum to 1 |
| Why √d_k | var(q·k) = d_k: measured 4.0, 63.6 and 502 for d_k = 4, 64, 512, and about 1 after scaling | unscaled scores at d_k = 512 give a one-hot softmax (entropy 0); scaled scores give entropy 2.69 |
| Self-attention | Q, K and V all come from the same sequence | **permutation-equivariant** without positional encoding: shuffling the inputs just shuffles the outputs |
| Cross-attention | Q from the decoder; K and V from the encoder | weights have shape (target length × source length); changing a source token changes the output |
| Multi-head attention | h heads with d_k = d_model/h, concatenated and multiplied by W_O | matches `nn.MultiheadAttention` with copied weights for self, cross and causal attention (9 × 10⁻⁸) |
| Positional encoding | sinusoids at geometrically spaced frequencies | PE[pos+k] = R_k · PE[pos] with the *same* rotation R_k for every pos (error 5 × 10⁻⁶); adding PE breaks the permutation equivariance |
| Masked attention | lower-triangular mask; blocked scores set to −∞, becoming exactly 0 after softmax | changing token 4 leaves outputs 0–3 unchanged; *without* the mask, output 0 changes, i.e. information leaks from the future |
| Full layers | encoder layer = self-attention + feed-forward; decoder layer = masked self-attention + cross-attention + feed-forward, each with Add & Norm | a full source → memory → decoder → vocabulary-probability pass has the correct shapes, and every row sums to 1 |

## Part 2 — Pretrained models (run on Colab)

1. **Translation** with `Helsinki-NLP/opus-mt-en-fr`. The notebook prints the model configuration, shows that each decoder layer contains an `encoder_attn` (cross-attention) module, translates four sentences including an ambiguous one ("bank"), and plots the **cross-attention map**, which acts as a soft word alignment for "I love machine learning."
2. **Encoder-only.** The DistilBERT SST-2 sentiment model is run on deliberately chosen cases: negation, sarcasm, and a neutral sentence it is forced to label. There is also **semantic search** with MiniLM mean-pooled embeddings: the query "I can't sign in" should rank "Troubleshooting login problems" first despite sharing no keywords.
3. **Decoder-only.**
   - The notebook verifies the **causal mask on a real GPT-2**: changing the last token leaves the logits at every earlier position unchanged.
   - It shows the next-token distribution (classification over roughly 50k tokens), and compares greedy decoding with temperature sampling.
   - It then contrasts base GPT-2 with the instruction-tuned **Qwen2.5-0.5B-Instruct**, including the special tokens its chat template adds.

The code was checked for syntax and API compatibility against transformers 5.18 using small randomly initialised Marian and GPT-2 models (generation, `cross_attentions`, the causal-mask test). The real pretrained checkpoints could not be downloaded in the environment where this solution was prepared, so **run the notebook yourself and record the outputs**.

## Part 3 — Ollama vs GPT / Claude

On Colab, run `03_ollama_local_llms.ipynb`. On Linux, run:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2:1b && ollama pull qwen2.5:1.5b
python3 ollama_compare.py --model llama3.2:1b --model qwen2.5:1.5b
```

Then paste the ChatGPT and Claude answers into `ollama_responses.md` and fill in the scoring tables. The 9 prompts each target one capability:

| Prompt | Tests | What to watch for |
|---|---|---|
| Capital of Australia | factual recall | small models sometimes say Sydney |
| Explain self-attention | explanation quality | accuracy vs vagueness |
| Bat and ball | multi-step reasoning | the intuitive but wrong answer "$0.10" (the correct answer is $0.05) |
| 17 × 24 | arithmetic | 408; small models often get this wrong |
| LIS in C++, O(n log n) | code generation | does it actually use `lower_bound`, and does it compile? |
| Translate to French | translation | compare with MarianMT from Part 2 |
| One-sentence summary | instruction following + compression | length constraint respected? |
| Non-existent 2019 paper | **hallucination** | does it invent findings, or say it does not know the paper? |
| Exactly three primes, comma-separated | strict format | 23, 29, 31, 37 are valid; watch for extra text or wrong count |

**Points for the write-up.**
- Local 1–2B models run privately and offline, at zero marginal cost, and fast on a GPU. They are noticeably weaker on reasoning, arithmetic, code correctness and hallucination resistance than frontier hosted models, which are orders of magnitude larger and more heavily post-trained.
- They are all the same basic architecture: decoder-only Transformers. The difference comes from scale, data and post-training.
- Record latency and tokens per second (the script measures both), and judge correctness yourself, not by how fluent an answer sounds.

## Libraries and tools
- PyTorch and matplotlib (Part 1).
- `transformers`, `sentencepiece` and `torch` (Part 2); Google Colab is optional.
- Ollama (Part 3).
