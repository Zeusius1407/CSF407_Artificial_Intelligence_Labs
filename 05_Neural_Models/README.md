# Lab 5 — Neural Models: Learning, Depth, Activations and Output Layers

| File | Purpose |
|---|---|
| `xor_lab.py` | All tasks: linear baseline, the 2-2-1 XOR network, gradient checks, symmetry, activations, 3-class extension |
| `results.txt` | Full output of `python3 xor_lab.py` (PyTorch, CPU, a few seconds) |
| `PROMPTS.md` | LLM prompts and the corrections made |

## Task 1 — Problem specification

- **Input space** 𝒳 = {0,1}², **output space** 𝒴 = {0,1}.
- **Examples:** (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0 (the XOR truth table).

```
x2
1 | (0,1)=1    (1,1)=0
0 | (0,0)=0    (1,0)=1
  +---------------------- x1
```

**Why one line can't separate them.** Class 1 sits on one diagonal and class 0 on the other. Suppose a line w·x + b had both 1-points on its positive side and both 0-points on its negative side. Adding the two 1-points' inequalities gives w₁ + w₂ + 2b > 0. Adding the two 0-points' inequalities gives w₁ + w₂ + 2b < 0. These contradict each other, so no such line exists.

**Prediction for affine + sigmoid.** It cannot fit XOR. By symmetry, the best it can do is output 0.5 for every input, where the loss is ln 2 ≈ 0.693.

**Observed.** Loss 0.6931, all four probabilities 0.49999..., and weights of about 10⁻⁶. The prediction was confirmed exactly.

**Think about it.** XOR tests the claim that *capability depends on the kind of representation, not the number of parameters*. A stack of affine layers can have any number of parameters and still only draw one linear boundary. A single nonlinear hidden layer changes what the model can represent.

## Task 2 — Model design and validation criteria

The model is 2 → 2 hidden units (tanh, later sigmoid or ReLU) → 1 logit. Training uses `BCEWithLogitsLoss`, which applies the sigmoid and binary cross-entropy in one numerically stable step, with full-batch Adam at lr = 0.1 for 3000 steps and a fixed seed.

1. **Why the hidden nonlinearity is necessary.** Without it, W₂(W₁x + b₁) + b₂ is still a single affine map (Task 1). The nonlinearity lets the hidden layer re-map the four points into a space where they become linearly separable. The network learns hidden features that act like OR and AND, and the output combines them as "OR and not AND".
2. **Why sigmoid + BCE.** The target is a single yes/no answer, i.e. a Bernoulli variable. Sigmoid produces a valid probability, and BCE is exactly its negative log-likelihood. Together, the gradient with respect to the logit is simply p − y. There is no σ′ factor, so the output layer does not saturate the gradient.
3. **Evidence that learning succeeded:**
   - the final loss is far below ln 2;
   - all 4 thresholded labels are correct;
   - first-layer gradients are non-zero early on and match finite differences;
   - the hidden units become different from each other (the representation changed);
   - the result holds across several seeds, or the failure rate is reported.

**Think about it.** No target values are given for the hidden units. What each hidden unit computes is determined entirely by the output loss. Backpropagation sends ∂L/∂h back through W₂ (the *credit assignment*), so each hidden unit is pushed toward whatever feature most reduces the final error. Different random initial weights send the units toward different features.

## Task 3 — LLM implementation

The prompt is in `PROMPTS.md`. Where each step of training happens in `xor_lab.py`, in `train()`:

| Step | Code |
|---|---|
| forward pass | `logits = model(X)` |
| scalar loss | `loss = loss_fn(logits, y)` (mean over the 4 examples) |
| reverse-mode AD | `loss.backward()` fills `.grad` |
| parameter update | `optimiser.step()` (after `zero_grad()`) |

**Checked by inspection before running:**
- `forward` returns raw logits. Applying `sigmoid` there *and* using `BCEWithLogitsLoss` would apply the sigmoid twice, which is a classic LLM slip.
- `zero_grad()` is inside the loop. Without it, gradients accumulate from step to step.

**Changes made:**
- `train()` also records the early gradient norm (for Task 4D).
- The linear-baseline call crashed because a plain `nn.Linear` has no `.hidden` attribute. I guarded that line with `hasattr`.

**Think about it.** Some things can be verified just by reading the code: the architecture shapes, the loss/activation pairing, the target labels, where `zero_grad`, `backward` and `step` sit, and whether the seed is set. Other things need running and measuring: whether the optimisation actually converges, whether the gradient values are correct (finite differences), whether a given seed hits a local minimum, and the numerical stability of the softmax.

## Task 4 — Results

### A. Basic learning check (tanh, seed 0)

| | value |
|---|---|
| initial loss | 0.7152 |
| final loss | 0.000038 |
| p(y=1) for (0,0), (0,1), (1,0), (1,1) | 0.0000, 1.0000, 0.9999, 0.0000 |
| correct | **4/4** |

### B. Backpropagation check

`model.hidden.weight.grad` holds ∂L/∂W⁽¹⁾. It has the same shape as W⁽¹⁾ (2×2), and entry (i,j) is the rate of change of the mean loss with respect to the weight from input j to hidden unit i.

| comparison | max abs. difference |
|---|---|
| full-batch gradient vs. average of the 4 per-example gradients | 0.0 |
| autograd vs. chain rule written out by hand, δ₂ = (σ(z)−y)/4 and δ₁ = δ₂W₂ ⊙ (1−h²) | 0.0 |
| autograd vs. central finite differences | 3.7 × 10⁻⁸ |

**Why the gradient is the average of per-example gradients.** The loss is L = ¼ Σᵢ ℓᵢ, and differentiation is linear, so ∇L = ¼ Σᵢ ∇ℓᵢ.

### C. Symmetry experiment

| initialisation | W⁽¹⁾ rows identical during training? | outcome |
|---|---|---|
| all zeros, tanh | yes, and they never move from 0 | loss stays at 0.6931, p = 0.5 everywhere |
| all zeros, sigmoid | yes, and they never move from 0 | loss stays at 0.6931 |
| all 0.5, tanh | yes: both rows change, but stay equal at every step (0.4967 → 0.5087 → 3.976) | stuck at loss 0.48, 3/4 correct |

**Explanation.** If the two hidden units start identical, they compute the same activation for every input. They also receive the same backpropagated error, since their outgoing weights in W₂ are equal. So they get identical gradients and identical updates, and stay identical forever. The network then behaves like a network with **one** hidden unit, which cannot represent XOR.

With all-zero initialisation it is even worse. The W₁ gradient is δ₂·W₂·f′(a)·x, and W₂ = 0, so W₁ gets no gradient at all. W₂'s own gradient is δ₂·h, and for tanh h = tanh(0) = 0, so W₂ is also stuck (only the output bias learns). That is why the 0.5 initialisation is the clearer demonstration of the symmetry itself: learning happens, but the two units never diverge. Random initialisation exists to break this symmetry.

### D. Activation experiment

Same seed and identical initial weights for all three runs:

| Hidden activation | Final loss | 4/4 correct? | ‖∇W⁽¹⁾L‖₂ at step 0 | at step 10 |
|---|---|---|---|---|
| Sigmoid | 0.4774 | ✘ (3/4) | 0.00089 | 0.00090 |
| Tanh | 0.00004 | ✔ | 0.06178 | 0.03976 |
| ReLU | 0.6931 | ✘ (2/4) | 0.00170 | **0.00000** |

Robustness across 20 seeds (number of seeds that reach 4/4):

| hidden units | sigmoid | tanh | ReLU |
|---|---|---|---|
| 2 | 9/20 | 7/20 | 5/20 |
| 4 | 18/20 | 19/20 | 11/20 |

**What was observed (engineering).**
- For this seed, tanh had a first-layer gradient about 70× larger than sigmoid and solved the task. Sigmoid converged to a local minimum. ReLU's gradient became exactly zero within 10 steps, and the network was stuck at ln 2.
- Across seeds, the 2-2-1 network with any of the three activations often fails. The typical failure loss is 0.3466 = ½ ln 2, meaning two of the four examples are fit and the other two are left at p = 0.5.
- Doubling the hidden layer to 4 units makes success far more reliable.

**Why (science).**
- **Sigmoid.** σ′(a) ≤ 0.25, and σ(a) is centred at 0.5 rather than 0. The backpropagated signal is therefore scaled down at every layer, which makes gradients small.
- **Tanh.** tanh′(0) = 1 and tanh is zero-centred, so there is a stronger gradient near initialisation.
- **ReLU.** A unit whose pre-activation is negative for **all four** inputs has derivative 0 everywhere, so it is "dead" and never recovers. With only two units, losing even one is fatal.
- **Local minima.** Two hidden units is the minimum capacity for XOR, so the loss landscape has poor local minima. Extra width creates more routes to a solution.

From four data points, no activation can be called "best". These results describe this architecture, optimiser and seed.

**Think about it (distinguishing the mechanisms).** Inspect the pre-activations a⁽¹⁾ (returned by `model(X, return_hidden=True)`) across the whole batch.
- A **saturated sigmoid** has |a| large, and σ(a) is close to 0 or 1 with σ′(a) ≈ 0 but *not* exactly 0. Moving the weights slightly still changes the output a little.
- A **dead ReLU** has a ≤ 0 for *every* input. Its activation is exactly 0 and its derivative is exactly 0, so the gradient for that unit is identically zero.

A histogram of a⁽¹⁾ plus a count of units with max(a) ≤ 0 tells the two apart.

## Task 5 — Three-class extension

Predictions made before running:

1. The final weight matrix W⁽²⁾ has shape **(3, 2)**, one row per class. The bias has shape (3,).
2. There are **3 logits** per example, so the batch output is (4, 3).
3. **Why softmax sums to 1.** pₖ = e^{zₖ} / Σⱼ e^{zⱼ}, so Σₖ pₖ = Σₖ e^{zₖ} / Σⱼ e^{zⱼ} = 1, and every pₖ > 0.
4. **Why the logit gradient is p − y.** L = −log p_y = −z_y + log Σⱼ e^{zⱼ}. Therefore ∂L/∂zₖ = −[k = y] + e^{zₖ} / Σⱼ e^{zⱼ} = pₖ − yₖ, where y is the one-hot target.

All four predictions were confirmed:

| input | p(class 0, 1, 2) | predicted | target |
|---|---|---|---|
| (0,0) | (1.00, 0.00, 0.00) | 0 | 0 |
| (0,1) | (0.00, 1.00, 0.00) | 1 | 1 |
| (1,0) | (0.00, 1.00, 0.00) | 1 | 1 |
| (1,1) | (0.00, 0.00, 1.00) | 2 | 2 |

- The final loss was 1 × 10⁻⁵. For x = (0,1), the probabilities sum to 1.0.
- The autograd logit gradient matched p − y to within 5.5 × 10⁻¹².
- **Shift invariance.** softmax(z) and softmax(z + 100) differ by at most 2.5 × 10⁻¹¹. The reason is e^{zₖ+c} / Σ e^{zⱼ+c} = e^{c}e^{zₖ} / (e^{c} Σ e^{zⱼ}); the factor e^{c} cancels.
- **Stability.** A naive softmax of z + 1000 returns `[nan, nan, nan]`, because e^{1000} overflows to inf, and inf/inf = nan. Subtracting the maximum logit first gives exactly the same result mathematically (by shift invariance), but the largest exponent is then e⁰ = 1, so nothing overflows. That is what stable implementations do.

**Think about it (next-token prediction).** Several things stay mathematically the same with a 50k-token vocabulary: a linear map to one logit per class, softmax, cross-entropy as the negative log-likelihood, the p − y logit gradient, shift invariance and max-subtraction. What changes dramatically is everything around them:
- the output matrix grows to (vocabulary × d_model), and is often tied to the input embedding;
- the hidden representation becomes a deep stack of Transformer blocks (attention, residual connections, layer norm) instead of 2 tanh units;
- inputs are learned token embeddings plus positional information;
- training runs on huge datasets with mini-batches, mixed precision, and memory-saving tricks for the large softmax.

## Reflection questions

1. **Depth vs. nonlinearity.** Depth alone does not help. Affine layers composed together are still affine, so the linear model is stuck at ln 2. What made XOR learnable was one *nonlinear* hidden layer. Depth only adds expressive power when there are nonlinearities between the layers.
2. **Evidence of a useful learning signal, not just a nonzero gradient.**
   - The loss fell from 0.715 to 4 × 10⁻⁵ and all labels became correct.
   - The gradient matched finite differences, so it is the *true* descent direction.
   - The hidden units became different, specialised features: h for (0,1) was (+0.99, −1.00) and for (1,0) was (−1.00, +0.99).
   - For contrast, the sigmoid seed-0 run also had non-zero gradients but stalled in a local minimum. A non-zero gradient alone is not proof of learning.
3. **Why identical or zero initialisation fails.** Identical units receive identical gradients, so they remain copies of each other and the effective width is 1. With zeros, the gradient to W⁽¹⁾ is additionally zero because W⁽²⁾ = 0.
4. **How the activation affected the gradient.**
   - *Observation:* at initialisation, tanh gave ‖∇W⁽¹⁾‖ ≈ 0.062, sigmoid ≈ 0.0009, and ReLU dropped to exactly 0 within 10 steps.
   - *Explanation:* the size of the derivative (σ′ ≤ 0.25, tanh′(0) = 1), whether the activation is zero-centred, and ReLU's exactly-zero derivative for negative inputs.
5. **Why output layer and loss go together.** The output layer defines a probability model for the target, and the loss should be that model's negative log-likelihood. Sigmoid pairs with BCE for one binary label; softmax pairs with categorical CE for one-of-K. These pairings give correctly normalised probabilities and the simple p − y gradient. Mismatches such as sigmoid + MSE, or softmax on multi-label targets, give wrong probabilities or saturating gradients. Applying the sigmoid twice (logits → sigmoid → `BCEWithLogitsLoss`) silently squashes the outputs into (0.5, 0.73).
6. **LLM productivity vs. human verification.**
   - *Productivity:* the LLM produced the training loop, the per-example gradient code and the finite-difference harness in seconds.
   - *Verification was essential:*
     - The first "symmetry" run used all-zero weights. That froze W⁽¹⁾ completely, so the output looked like a symmetry demonstration while actually showing a zero gradient. Recognising this and adding the 0.5 initialisation was a human judgement.
     - The single-seed activation table alone suggested "tanh is best". The 20-seed runs showed the 2-2-1 network fails often with every activation.
     - The code also had to be checked so that the sigmoid is not applied twice alongside `BCEWithLogitsLoss`.
7. **Which tests scale.**
   - *Keep:* loss curves; train and validation accuracy; gradient-norm monitoring per layer (to detect vanishing gradients or dead ReLUs); checking that a model can overfit a tiny batch; multiple seeds; softmax sum and stability checks; shape assertions.
   - *Too expensive at scale:* an exhaustive finite-difference check, which needs 2 forward passes *per parameter*. At scale you instead spot-check a random few parameters or a tiny model. Exact per-example gradient comparisons for the whole dataset also become too expensive.
