# Lab 3 — Logical Reasoning for Planning

| File | Purpose |
|---|---|
| `planner.py` | STRIPS-style planner: `applicable` / `apply` (logic), `bfs_plan` (search), `validate_plan` (independent checker) |
| `test_planner.py` | Task 0 applicability, Task 1 hand-plan check, Tests A, B, C and C2, and an extra edge case |
| `results.txt` | Output of `python3 test_planner.py` |
| `planner.pl`, `rules.pl` | Optional Prolog extension (Tasks 6–8), plus a whole-plan verifier |
| `prolog_results.txt` | Query results from SWI-Prolog 9.0.4 |
| `PROMPTS.md` | LLM prompts and the changes made by hand |

## Task 0 — The planning problem

- **I** = {At(Robot,A), At(Package,A)}
- **G** = {At(Package,C)}
- **Actions.** In every action below, X, Y and L range over the locations, and Move(X,Y) requires X and Y to be connected (A–B and B–C, in both directions).

| Action | Preconditions | Add effects | Delete effects |
|---|---|---|---|
| Move(X,Y) | At(Robot,X) | At(Robot,Y) | At(Robot,X) |
| PickUp(Package,L) | At(Robot,L), At(Package,L), ¬Holding(Package)\* | Holding(Package) | At(Package,L) |
| Drop(Package,L) | At(Robot,L), Holding(Package) | At(Package,L) | Holding(Package) |

\* The negative precondition is not in the handout. I added it to exercise the negative-precondition feature the prompt asks for. It does not change any result, because there is only one package.

**Applicability in I.**
- **PickUp(Package,A) is applicable.** Its preconditions At(Robot,A) and At(Package,A) are both in I, and Holding(Package) is not.
- **Drop(Package,C) is not applicable.** At(Robot,C) and Holding(Package) are both false in I.
- The only applicable actions in I are Move(A,B) and PickUp(Package,A). This was checked by the program (see `results.txt`).

## Task 1 — Plan constructed by hand

The example sequence suggested in the handout, Move(A,B), PickUp(Package,B), …, is **not valid**. The package is at A, so At(Package,B) is false when PickUp(Package,B) is attempted. `validate_plan` reports: *step 2 PickUp(Package,B): preconditions not satisfied, missing {At(Package,B)}*. This is a good example of a plan that "looks reasonable" but fails.

A correct plan is PickUp(Package,A) → Move(A,B) → Move(B,C) → Drop(Package,C):

| State | Facts |
|---|---|
| S0 | At(Robot,A), At(Package,A) |
| S1 (PickUp(Package,A)) | At(Robot,A), Holding(Package) |
| S2 (Move(A,B)) | At(Robot,B), Holding(Package) |
| S3 (Move(B,C)) | At(Robot,C), Holding(Package) |
| S4 (Drop(Package,C)) | At(Robot,C), At(Package,C)  ⊨ G ✔ |

## Task 2 — The LLM implementation

The prompt from the handout was used (see `PROMPTS.md`). Where the specification appears in `planner.py`:

| Specification | Code |
|---|---|
| Preconditions (when is an action applicable?) | `applicable()`: `pos_pre <= state and not (neg_pre & state)` |
| Effects (how does the state change?) | `apply()`: `(state - delete) | add` |
| Goal (when does planning stop?) | `satisfies(s, goal)`: `goal <= s`, checked when a state is dequeued |
| BFS (how are alternatives explored?) | `bfs_plan()`: a FIFO `deque`, with a `parent` dict as the visited set |

Assumptions: the closed-world assumption (a fact that is absent is false); states are `frozenset`s so they can be hashed; delete effects are applied before add effects; and the plan returned is the shortest in number of actions (the BFS property).

## Task 3 — Tests

| Test | Initial state | Goal | Plan found? | Plan | Valid? |
|---|---|---|---|---|---|
| A: original | I | At(Package,C) | yes | PickUp(P,A), Move(A,B), Move(B,C), Drop(P,C) | ✔ (the validator re-executes it) |
| B: PickUp removed | I | At(Package,C) | **No plan found** | – | ✔ correct refusal; no action was invented |
| C: added robot-only actions Move(A,C), Recharge(C) | I | At(Package,C) | yes | PickUp(P,A), Move(A,C), Drop(P,C) | ✔. With goal At(Robot,C) instead, the plan is just Move(A,C), so the planner does **not** confuse the robot being at C with the package being at C |
| C2: Drop removed | I | At(Package,C) | **No plan found** | – | ✔. The robot *can* reach C holding the package, but that does not satisfy At(Package,C) |
| Extra: goal already true | {At(Robot,A), At(Package,C)} | At(Package,C) | yes | empty plan | ✔ |

## Task 4 — Logic and search

```
Current state
     ↓
Check action preconditions         (logic:  S ⊨ Pre(a)?)
     ↓
Select an applicable action a      (keep only actions that pass the check)
     ↓
Generate successor state           (logic:  S' = (S − Del(a)) ∪ Add(a))
     ↓
Search over alternatives           (search: BFS queue, visited set)
     ↓
Goal?  (S' ⊨ G)  — yes: return the plan / no: continue the search
```

Logical reasoning answers local questions. Given a state, which actions are legal, and what is true afterwards? Search answers the global question: in what order should the possible action sequences be explored so that one reaching the goal is found (the shortest, in the case of BFS)? In short, logic defines the edges of the state graph, and search walks that graph.

## Task 5 — Can the LLM verify its own plan?

The LLM (Claude) was asked to justify each step of its plan. Its explanation was: *PickUp(Package,A) needs At(Robot,A) and At(Package,A), both true in S0. Move(A,B) needs At(Robot,A), still true in S1. Move(B,C) needs At(Robot,B), true in S2. Drop(Package,C) needs At(Robot,C) and Holding(Package), both true in S3. The result S4 contains At(Package,C).* This agrees with the state table computed by `validate_plan`. Even so, **the independently executed state transitions are what should be trusted.** The explanation is generated text. It is produced by the same process that produced the plan, so any mistake in the model's understanding (for example the handout's PickUp-at-B sequence) can be reproduced *and* "justified". The executed checker mechanically applies the precondition and effect definitions to explicit states, and it gives the same answer every time. A generated explanation is not an independent verification.

## Reflection questions

1. **Why specify preconditions and effects first?** They *are* the semantics of the problem. With them written down, the generated code can be checked against a precise standard, and the prompt leaves the LLM no room to invent its own rules (for example, letting the package move with the robot without a Holding fact).
2. **An error from skipping precondition checks.** The planner could return Move(A,B), PickUp(Package,B), …, picking up a package that is not there. A "teleporting" plan such as Drop(Package,C) as the first action would also be possible. Either one could be reported as a valid 1–4 step plan.
3. **Why "looks reasonable" ≠ valid.** Validity is a precise property: every action's preconditions hold in the state where it is executed, and the final state satisfies G. A sequence can have the right *shape* (move, pick up, move, drop) and still violate a precondition. The handout's own example shows this.
4. **What the LLM contributed.** Boilerplate: the `Action` dataclass, the BFS loop with parent pointers, and formatted output.
5. **What was verified independently.** Applicability of every action in I; the returned plan (re-executed from scratch by `validate_plan`); the no-plan cases (B and C2); that a robot-only goal is distinguished from the package goal (C); and that the Prolog verifier agrees with the Python result.
6. **Where logic is used.** Entailment of preconditions (S ⊨ Pre(a)), the effect/successor-state computation, and the goal test (S ⊨ G). The Prolog part adds inference from facts and rules.
7. **Relation to search.** Planning *is* search in a state space whose states are sets of propositions and whose transitions are applicable actions. BFS here is the same algorithm used on the grid in the search lab. Only the representation of states and successors has changed, and A* would apply just as well given a heuristic (for example, the number of unsatisfied goal facts).

## Optional extension — Prolog (SWI-Prolog 9.0.4)

| Query | Result |
|---|---|
| `can_move(a,b)` | true |
| `can_move(a,c)` | false |
| `valid_move(a,b)` | true |
| `valid_move(b,c)` | true |
| `valid_move(a,c)` | false |
| `reduce_speed` (rules.pl) | true |

**Task 6.**
(a) `can_move(a,b)` succeeds because it unifies with the rule head `can_move(X,Y)` with X=a, Y=b, and the body `connected(a,b)` matches a fact.
(b) `can_move(a,c)` fails because there is no fact `connected(a,c)`, and the rule is not transitive. Prolog uses negation as failure, so "not provable" is reported as false.
(c) The rule `can_move(X,Y) :- connected(X,Y).` is the Horn clause ∀X,Y. Connected(X,Y) → CanMove(X,Y). The body is the antecedent and the head is the consequent.

**Task 7 challenge.** A proposed `Move(a,c)` is **not** supported by the warehouse knowledge: `valid_move(a,c)` is false. (Test C in the Python part *adds* such an action on purpose. Prolog would flag it as inconsistent with the map.)

**Extension.** `planner.pl` also defines `step/3` and `valid_plan/3`, which simulate the robot and package state, so Prolog can verify a *whole* plan. It accepts `[pickup(a),move(a,b),move(b,c),drop(c)]`. It rejects the handout's sequence, the plan using the non-existent A→C edge, and a plan without the final drop. Run backwards, it also *finds* the unique 4-step plan by itself.

**Task 8.** `wet_road` is a fact, so `slippery` holds, and therefore `reduce_speed` holds:

WetRoad (fact) ⇒ [WetRoad → Slippery] ⇒ Slippery ⇒ [Slippery → ReduceSpeed] ⇒ ReduceSpeed.

Prolog proves this by backward chaining: to prove `reduce_speed`, prove `slippery`; to prove that, prove `wet_road`, which is a fact.

**Prolog reflection.**
1. A **fact** is unconditionally true (`connected(a,b).`). A **rule** is true *if* its body holds (`can_move(X,Y) :- connected(X,Y).`).
2. A query asks whether the goal can be derived from the knowledge base (KB ⊢ query) by resolution. Under Prolog's closed-world, negation-as-failure semantics, this approximates asking whether the KB entails it.
3. **Why verify a Python plan with Prolog?** The verifier is written in a different language, from the declarative domain description, and does not share code or bugs with the generator. Agreement between two independent implementations is much stronger evidence than one implementation agreeing with itself.
4. **Advantage when an LLM helped write the plan or planner.** LLM output can be fluent and wrong. A separate formal checker converts "looks right" into "was derived from the stated rules". This is the generate → independently verify architecture: the LLM or planner proposes, and logic disposes.
