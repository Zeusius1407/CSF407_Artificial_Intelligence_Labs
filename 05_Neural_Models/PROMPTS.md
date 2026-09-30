# LLM prompts used (Lab 5)

LLM used: Claude (Anthropic).

## Prompt 1: binary XOR (Task 3)
> Generate minimal PyTorch code for the following model and dataset. Do not change the architecture or the task.
> Data: X = (0,0),(0,1),(1,0),(1,1) with y = 0,1,1,0.
> Model: 2 inputs → 2 hidden units (tanh, but make the activation selectable between sigmoid, tanh and ReLU) → 1 output logit. Use BCEWithLogitsLoss, random initialisation, and full-batch training for 3000 CPU steps.
> After training, report the final loss, all four probabilities, the thresholded labels, and the gradient of the first-layer weights after backward(). Set a random seed for reproducibility and explain each test in one sentence.

## Prompt 2: gradient checks (Task 4B)
> Add a check that the full-batch gradient of W1 equals the mean of the four per-example gradients. Also compare it with a hand-written chain-rule computation and with central finite differences in float64.

## Prompt 3: symmetry and activations (Task 4C/4D)
> Add a copy of the experiment with all weights set to zero, printing both rows of W1 at several steps. Then train with sigmoid, tanh and ReLU from the same initial weights, recording the final loss, whether 4/4 are correct, and the norm of the W1 gradient early in training.

## Prompt 4: three-class output (Task 5)
> Modify only the output and loss: 3 logits and CrossEntropyLoss, with classes (0,0)→0, (0,1)/(1,0)→1, (1,1)→2. Print the softmax probabilities, verify that dL/dlogits = p − y, check shift invariance under +100, and show naive vs max-subtracted softmax at +1000.

## Verification done and changes made
- The linear baseline crashed (`Linear` has no `.hidden`). I fixed it with a `hasattr` guard.
- All-zero initialisation gives zero W1 gradients, not just symmetric ones. I added a constant-0.5 initialisation that shows the symmetry more clearly.
- I added multi-seed runs with 2 and 4 hidden units, so that conclusions are not drawn from a single seed.
