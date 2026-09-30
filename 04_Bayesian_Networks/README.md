# Lab 4 — Building and Learning a Bayesian Network (LLM as a coding assistant)

| File | Purpose |
|---|---|
| `llm_bn_solution.ipynb` | The course notebook, completed and executed. All original cells are kept; solution cells are added |
| `recorded_llm_responses.json` | LLM responses replayed by the notebook (see below) |
| `requirements.txt` | pgmpy ≥ 1.1, pandas, numpy, matplotlib |

The notebook runs top to bottom with `LLM_BACKEND = "recorded"`, with no GPU and no model download. Setting `LLM_BACKEND = "qwen"` regenerates the v1 responses live with Qwen2.5-Coder-1.5B-Instruct. It was tested with pgmpy 1.1.2.

**Where the responses come from.** The `qwen1.5b/*_v1` entries are the actual outputs Qwen2.5-Coder-1.5B produced for the course prompts. They were taken from the outputs saved in the lab notebook. The `claude/*` entries were produced by Claude for the revised (v2) prompts written in this solution.

## What the solution does

**1. Reviews the v1 LLM code and shows which validation layer catches each defect.**

| Program | Defects in the Qwen 1.5B output | Caught by |
|---|---|---|
| Inference | `BayesianModel` (removed API); Rain rows swapped; Sprinkler and WetGrass columns do not sum to 1; `evidence_card=[2]` for 2 parents; `model.query` does not exist | execution (`ImportError`) → `check_model()` (`ValueError`, Sprinkler) → **semantic posterior test** (swapped-but-valid rows give 0.846 instead of 0.705) |
| MLE estimation | **overwrites `data` with 4 invented rows**; removed API; misused estimator | *only* the numerical comparison with the trusted fit. With just the API repaired, `check_model()` passes but the CPTs are off by up to 0.91 |
| Bayesian (BDeu) | **`pd.read_csv('your_data.csv')`** despite "do not read files"; wrong `BayesianEstimator` signature; no graph | the original AST check **missed it**. A stronger checker (attribute calls plus assignments to `data`) catches it |

**2. Revised prompts (v2)** that remove the ambiguities: pgmpy ≥ 1.0 class names, the TabularCPD row and column convention, "never modify `data`", and the `get_parameters()` route. Every v2 program was inspected, approved and **validated**:

| v2 program | Validation result |
|---|---|
| Inference | structure ✔, `check_model` ✔, P(R=1\|W=1) = 0.7048, equal to VE and to enumeration |
| MLE | max CPT difference to the trusted `DiscreteMLE` fit = 0; `data` unchanged |
| BDeu | max difference to the trusted `DiscreteBayesianEstimator` = 0. It also **differs from MLE** (0.781 vs 0.909), which shows the prior really is applied |

Note on API drift: the course prompt itself asks for `MaximumLikelihoodEstimator` with `fit`. In pgmpy 1.1, `model.fit(data, estimator=MaximumLikelihoodEstimator)` raises `TypeError`, because an instance such as `DiscreteMLE()` is now required. The legacy classes still work through `.get_parameters()`, and give identical CPTs.

**3. HW exercise: changing the query.** The predictions were written *before* running anything, and the results were then checked against the trusted engine and an independent 16-assignment enumeration:

| Query | Prior | Prediction | Result |
|---|---|---|---|
| P(S=1 \| W=1) | 0.30 | increase (diagnostic reasoning), but stays below rain | **0.4278** ✔ |
| P(C=1 \| W=1) | 0.50 | slight increase: the positive path via Rain outweighs the negative path via Sprinkler | **0.5746** ✔ |
| P(R=1 \| W=1, S=0) | 0.50 | sharp increase: the alternative cause is ruled out | **0.9922** ✔ (hand-computed as 0.369 / 0.3719) |

For contrast, P(R=1 \| W=1, S=1) = 0.320. This is *explaining away*.

**4. CPD-explanation exercise.** The LLM's column explanation (Sprinkler varies fastest) was verified programmatically: the matrix entries, name-based `get_value(...)` lookups, and the specification all agree.

**5. Multiple datasets.** Over 20 seeds, the estimate of P̂(R=1\|C=1) is centred on 0.8. Its spread is 0.061 at N=100 and 0.020 at N=1000, which matches the binomial standard error √(0.16 / N(C=1)) (0.057 and 0.018). The differences are **sampling variability**. They are not caused by pgmpy or the LLM.

## Key takeaways
1. The most dangerous defects are the ones that *pass* `check_model()`. Semantic tests are what catch them.
2. Static checks need to cover attribute calls and protected inputs. Even then, they are not a sandbox.
3. Almost every v1 defect traced back to something the prompt left implicit.
4. Pin library versions, because generated code, and even course prompts, drift out of date.
