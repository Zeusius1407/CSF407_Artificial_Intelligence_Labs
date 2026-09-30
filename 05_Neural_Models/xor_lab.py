"""
CSF407 - Neural Models Lab: learning, depth, activations and output layers.

Run:  python3 xor_lab.py            (CPU, a few seconds)

Sections
  Task 1  : linear baseline (affine + sigmoid) cannot learn XOR
  Task 3/4A: 2-2-1 network, BCEWithLogitsLoss, full-batch training
  Task 4B : what .grad holds; mean loss => averaged per-example gradients;
            autograd vs. hand-derived backprop vs. finite differences
  Task 4C : symmetry problem with zero / identical initialisation
  Task 4D : sigmoid vs tanh vs ReLU hidden activation (+ multi-seed robustness)
  Task 5  : three-class extension with softmax + cross-entropy, p - y check,
            shift invariance and numerical stability
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(precision=4, sci_mode=False)

# ------------------------------------------------------------------ data ---
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y_XOR = torch.tensor([[0.], [1.], [1.], [0.]])
Y_3CLASS = torch.tensor([0, 1, 1, 2])          # both off / disagree / both on

ACTS = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu}


# ----------------------------------------------------------------- model ---
class XORNet(nn.Module):
    """2 -> 2 -> n_out. Returns LOGITS (the loss applies sigmoid/softmax)."""

    def __init__(self, act="tanh", n_hidden=2, n_out=1):
        super().__init__()
        self.hidden = nn.Linear(2, n_hidden)    # W1: (n_hidden, 2), b1: (n_hidden,)
        self.out = nn.Linear(n_hidden, n_out)   # W2: (n_out, n_hidden)
        self.act_name = act
        self.act = ACTS[act]

    def forward(self, x, return_hidden=False):
        a1 = self.hidden(x)                     # pre-activation
        h1 = self.act(a1)                       # hidden representation
        logits = self.out(h1)
        return (logits, a1, h1) if return_hidden else logits


def train(model, y, loss_fn, steps=3000, lr=0.1, opt="adam", log_every=None, early_step=10):
    optimiser = (torch.optim.Adam if opt == "adam" else torch.optim.SGD)(model.parameters(), lr=lr)
    hist, early_norm = [], None
    for step in range(steps):
        optimiser.zero_grad()
        logits = model(X)                       # forward pass
        loss = loss_fn(logits, y)               # scalar loss (mean over 4 examples)
        loss.backward()                         # reverse-mode AD fills .grad
        if step == early_step and hasattr(model, "hidden"):
            early_norm = model.hidden.weight.grad.norm().item()
        optimiser.step()                        # parameter update
        hist.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"    step {step:5d}  loss {loss.item():.4f}")
    return hist, early_norm


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ------------------------------------------------------------- Task 1 ------
def task1_linear_baseline():
    section("Task 1 - Linear baseline: a single affine map + sigmoid")
    torch.manual_seed(0)
    lin = nn.Linear(2, 1)
    hist, _ = train(lin, Y_XOR, nn.BCEWithLogitsLoss(), steps=3000, lr=0.1)
    p = torch.sigmoid(lin(X)).detach()
    print(f"  final loss {hist[-1]:.4f}   (ln 2 = {math.log(2):.4f})")
    print("  probabilities:", p.squeeze().tolist())
    print("  weights:", lin.weight.data.tolist(), " bias:", lin.bias.data.tolist())
    print("  -> the best a linear boundary can do is output ~0.5 everywhere.")


# --------------------------------------------------------- Task 3 / 4A -----
def task3_4a_binary(seed=0, act="tanh"):
    section(f"Task 3/4A - 2-2-1 network, hidden={act}, BCEWithLogitsLoss, seed={seed}")
    torch.manual_seed(seed)
    model = XORNet(act)
    loss_fn = nn.BCEWithLogitsLoss()
    init_loss = loss_fn(model(X), Y_XOR).item()
    hist, _ = train(model, Y_XOR, loss_fn, steps=3000, lr=0.1, log_every=500)
    with torch.no_grad():
        logits, a1, h1 = model(X, return_hidden=True)
        prob = torch.sigmoid(logits)
    labels = (prob > 0.5).float()
    print(f"  initial loss {init_loss:.4f}  ->  final loss {hist[-1]:.6f}")
    for x, p, l, y in zip(X.tolist(), prob.squeeze().tolist(), labels.squeeze().tolist(), Y_XOR.squeeze().tolist()):
        print(f"  x={x}  p(y=1)={p:.4f}  label={int(l)}  target={int(y)}")
    correct = int((labels == Y_XOR).sum())
    print(f"  {correct}/4 correct")
    print("  learned hidden representation h1 (rows = inputs):")
    print("  ", h1.tolist())
    return model


# -------------------------------------------------------------- Task 4B ----
def task4b_gradients(seed=0):
    section("Task 4B - Backpropagation check")
    torch.manual_seed(seed)
    model = XORNet("tanh")
    loss_fn = nn.BCEWithLogitsLoss()           # reduction='mean'

    model.zero_grad()
    loss = loss_fn(model(X), Y_XOR)
    loss.backward()
    g_batch = model.hidden.weight.grad.clone()
    print("  dL/dW1 from autograd (full batch, mean loss):\n  ", g_batch.tolist())

    # (i) mean loss => gradient is the average of per-example gradients
    per_ex = []
    for i in range(4):
        model.zero_grad()
        loss_fn(model(X[i:i + 1]), Y_XOR[i:i + 1]).backward()
        per_ex.append(model.hidden.weight.grad.clone())
    g_avg = torch.stack(per_ex).mean(0)
    print("  average of the 4 per-example gradients:\n  ", g_avg.tolist())
    print("  max |difference| =", (g_batch - g_avg).abs().max().item())

    # (ii) hand-derived backprop (chain rule) for BCE-with-logits + tanh
    W1, b1 = model.hidden.weight.detach(), model.hidden.bias.detach()
    W2, b2 = model.out.weight.detach(), model.out.bias.detach()
    a1 = X @ W1.T + b1
    h1 = torch.tanh(a1)
    z = h1 @ W2.T + b2
    delta2 = (torch.sigmoid(z) - Y_XOR) / 4          # dL/dz  (sigmoid+BCE => p - y, /N for mean)
    delta1 = (delta2 @ W2) * (1 - h1 ** 2)           # dL/da1 = delta2 W2 * tanh'(a1)
    g_manual = delta1.T @ X                          # dL/dW1
    print("  hand-derived chain rule dL/dW1:\n  ", g_manual.tolist())
    print("  max |autograd - manual| =", (g_batch - g_manual).abs().max().item())

    # (iii) central finite differences (feasible only because there are 4 weights)
    eps = 1e-3
    g_fd = torch.zeros_like(W1)
    m64 = copy.deepcopy(model).double()
    for i in range(2):
        for j in range(2):
            with torch.no_grad():
                m64.hidden.weight[i, j] += eps
                lp = loss_fn(m64(X.double()), Y_XOR.double()).item()
                m64.hidden.weight[i, j] -= 2 * eps
                lm = loss_fn(m64(X.double()), Y_XOR.double()).item()
                m64.hidden.weight[i, j] += eps
            g_fd[i, j] = (lp - lm) / (2 * eps)
    print("  finite-difference dL/dW1:\n  ", g_fd.tolist())
    print("  max |autograd - finite diff| =", (g_batch - g_fd).abs().max().item())


# -------------------------------------------------------------- Task 4C ----
def task4c_symmetry():
    section("Task 4C - Symmetry: identical initial weights")
    for act, init in [("tanh", 0.0), ("sigmoid", 0.0), ("tanh", 0.5)]:
        torch.manual_seed(0)
        model = XORNet(act)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(init)
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        loss_fn = nn.BCEWithLogitsLoss()
        print(f"\n  hidden={act}, all weights and biases = {init}")
        for step in range(2001):
            opt.zero_grad()
            loss = loss_fn(model(X), Y_XOR)
            loss.backward()
            if step in (0, 1, 2, 10, 2000):
                W1 = model.hidden.weight.data
                print(f"    step {step:4d}  W1 row0={W1[0].tolist()}  row1={W1[1].tolist()}  "
                      f"identical={torch.equal(W1[0], W1[1])}  loss={loss.item():.4f}")
            opt.step()
        p = torch.sigmoid(model(X)).detach().squeeze().tolist()
        print("    final probabilities:", [round(v, 3) for v in p])


# -------------------------------------------------------------- Task 4D ----
def task4d_activations(seed=0):
    section(f"Task 4D - Activation experiment (same seed={seed}, same initial weights)")
    torch.manual_seed(seed)
    base = XORNet("tanh")
    init_state = copy.deepcopy(base.state_dict())
    rows = []
    for act in ["sigmoid", "tanh", "relu"]:
        model = XORNet(act)
        model.load_state_dict(init_state)
        # gradient at step 0, before any update, is a clean "early" comparison
        model.zero_grad()
        nn.BCEWithLogitsLoss()(model(X), Y_XOR).backward()
        g0 = model.hidden.weight.grad.norm().item()
        with torch.no_grad():
            a1 = model.hidden(X)
            dead = int((a1 <= 0).all(0).sum()) if act == "relu" else 0
            sat = (torch.sigmoid(a1) * (1 - torch.sigmoid(a1))).mean().item() if act == "sigmoid" else None
        hist, g10 = train(model, Y_XOR, nn.BCEWithLogitsLoss(), steps=3000, lr=0.1)
        pred = (torch.sigmoid(model(X)) > 0.5).float()
        ok = int((pred == Y_XOR).sum())
        rows.append((act, hist[-1], ok, g0, g10, dead))
    print(f"  {'activation':<10}{'final loss':>12}{'correct':>9}{'||grad W1|| step0':>20}{'step10':>10}{'dead ReLUs@init':>17}")
    for act, L, ok, g0, g10, dead in rows:
        print(f"  {act:<10}{L:>12.5f}{ok:>7}/4{g0:>20.5f}{g10:>10.5f}{dead:>17}")

    for n_hidden in (2, 4):
      print(f"\n  Robustness over 20 seeds, {n_hidden} hidden units (3000 Adam steps, lr=0.1):")
      for act in ["sigmoid", "tanh", "relu"]:
        succ, finals = 0, []
        for s in range(20):
            torch.manual_seed(s)
            m = XORNet(act, n_hidden=n_hidden)
            h, _ = train(m, Y_XOR, nn.BCEWithLogitsLoss(), steps=3000, lr=0.1)
            succ += int(((torch.sigmoid(m(X)) > 0.5).float() == Y_XOR).all())
            finals.append(h[-1])
        finals.sort()
        print(f"    {act:<8} solved {succ:2d}/20   median final loss {finals[10]:.4f}   worst {finals[-1]:.4f}")


# ---------------------------------------------------------------- Task 5 ---
def task5_three_class(seed=0):
    section("Task 5 - Three-class extension: 3 logits + softmax cross-entropy")
    torch.manual_seed(seed)
    model = XORNet("tanh", n_hidden=2, n_out=3)
    print("  final weight matrix W2 shape:", tuple(model.out.weight.shape), " bias:", tuple(model.out.bias.shape))
    print("  logits per batch:", tuple(model(X).shape), "-> 3 logits per example")
    loss_fn = nn.CrossEntropyLoss()             # log-softmax + NLL, on raw logits
    hist, _ = train(model, Y_3CLASS, loss_fn, steps=3000, lr=0.1)
    with torch.no_grad():
        probs = F.softmax(model(X), dim=1)
    print(f"  final loss {hist[-1]:.5f}")
    for x, p, y in zip(X.tolist(), probs.tolist(), Y_3CLASS.tolist()):
        print(f"  x={x}  p={[round(v, 4) for v in p]}  argmax={max(range(3), key=lambda k: p[k])}  target={y}")
    print("  sum of probabilities for x=[0,1]:", probs[1].sum().item())

    # p - y check on the logits
    logits = model(X).detach().requires_grad_(True)
    loss = F.cross_entropy(logits, Y_3CLASS, reduction="sum")
    loss.backward()
    p_minus_y = F.softmax(logits, 1).detach() - F.one_hot(Y_3CLASS, 3).float()
    print("  dL/dlogits (autograd):", logits.grad[0].tolist())
    print("  p - y (formula)      :", p_minus_y[0].tolist())
    print("  max |difference| over all examples:", (logits.grad - p_minus_y).abs().max().item())

    # shift invariance and stability
    z = model(X[1:2]).detach().squeeze()
    print("\n  softmax(z)       =", F.softmax(z, 0).tolist())
    print("  softmax(z + 100) =", F.softmax(z + 100, 0).tolist())
    print("  max diff:", (F.softmax(z, 0) - F.softmax(z + 100, 0)).abs().max().item())

    def naive_softmax(v):
        e = torch.exp(v)
        return e / e.sum()

    def stable_softmax(v):
        e = torch.exp(v - v.max())
        return e / e.sum()

    big = z + 1000
    print("  naive  softmax(z + 1000) =", naive_softmax(big).tolist(), " <- exp overflows to inf, inf/inf = nan")
    print("  stable softmax(z + 1000) =", stable_softmax(big).tolist())


if __name__ == "__main__":
    task1_linear_baseline()
    task3_4a_binary()
    task4b_gradients()
    task4c_symmetry()
    task4d_activations()
    task5_three_class()
