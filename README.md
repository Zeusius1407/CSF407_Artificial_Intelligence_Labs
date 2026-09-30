# CSF407_Artificial_Intelligence_Labs
Repository containing the hands-on exercises and labs for the CSF407 AI course.

| # | Lab | Main files | How to run |
|---|---|---|---|
| 01 | [Goal-based agent](01_Agents_Goal_Based_Agent/) | `warehouse_agent.py`, `test_warehouse_agent.py` | `python3 warehouse_agent.py` |
| 02 | [Search and A*](02_Search_AStar/) | `astar_search.py`, `experiments.py` | `python3 experiments.py` |
| 03 | [Logic for planning (+ Prolog)](03_Logic_Planning/) | `planner.py`, `test_planner.py`, `planner.pl`, `rules.pl` | `python3 test_planner.py`; `swipl planner.pl` |
| 04 | [Bayesian networks with an LLM coding assistant](04_Bayesian_Networks/) | `llm_bn_solution.ipynb` | Jupyter (`pip install -r requirements.txt`) |
| 05 | [Neural models: XOR, backprop, activations, softmax](05_Neural_Models/) | `xor_lab.py` | `python3 xor_lab.py` |
| 06 | [Transformers](06_Transformers/) | `01_attention_from_scratch.ipynb`, `02_huggingface_pipelines.ipynb`, `03_ollama_local_llms.ipynb` | Jupyter / Colab |

Each folder has a `README.md` with the lab report: the problem formulation, the answers to every task and reflection question, the result tables, and the prompts given to the LLM (`PROMPTS.md` or inside the notebook).

**Requirements:** Python ≥ 3.10, `numpy`, `pandas`, `matplotlib`, `torch` (CPU is fine) and `pgmpy>=1.1`. Lab 6 Part 2 also needs `transformers` and `sentencepiece`, Part 3 needs Ollama, and the optional Prolog part of Lab 3 needs SWI-Prolog.
