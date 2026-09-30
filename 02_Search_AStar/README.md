# Lab 2 — Search and A* (using an LLM as an engineering assistant)

| File | Purpose |
|---|---|
| `astar_search.py` | Problem formulation (`GridProblem`), heuristics, A*, BFS, path rendering |
| `experiments.py` | Task 3 tests (with assertions), Task 5 BFS vs A*, Task 6 heuristic study |
| `results.txt` | Output of `python3 experiments.py` |
| `PROMPTS.md` | Prompts given to the LLM and the changes made by hand |

## Task 0 — Search problem formulation

| Component | Specification |
|---|---|
| State S | a free cell `(row, col)`, i.e. any cell that is not `#` |
| Actions A | {Up, Down, Left, Right} |
| Transition T | T((r,c),a) = (r+dr, c+dc); defined only if the target cell is inside the grid and free |
| Initial state s₀ | cell of `S` = (1, 1) |
| Goal G | {cell of `G`} = {(7, 15)} |
| Cost c | 1 per move |

(a) **What specifies a state?** The robot's position alone. The map is static, so it is part of the problem, not part of the state.
(b) **What makes an action invalid?** The target cell is `#` or outside the grid.
(c) **Deterministic?** Yes. Each (state, action) pair has exactly one outcome, and the environment is fully observable and static.
(d) **What is a solution?** A sequence of actions whose transitions lead from s₀ to G. An *optimal* solution is one of minimum total cost, which here means the fewest moves.

## Task 1 — Agent design

1. **State representation:** a tuple `(row, col)`. Tuples are hashable, so they can be used in sets and dicts.
2. **Warehouse representation:** a list of lists of characters, padded with `#` if rows are ragged.
3. **Valid actions:** `actions(s)` returns every move whose target cell passes `free()`.
4. **Goal recognition:** `is_goal(s)` checks `s == goal`. The test is applied when a node is **popped** from the frontier, not when it is generated. This is what guarantees optimality for A*.
5. **Frontier contents:** entries `(f, tie_breaker, state)` in a binary heap, plus a `g` dict holding the best known cost to each state.
6. **Path reconstruction:** a `parent` dict, followed backwards from the goal.

Reported at termination: whether a solution was found, the path, its length, and the number of states expanded.

## Task 3 — Tests

| Test | Map | Expected | Result |
|---|---|---|---|
| 1 | original warehouse | a valid shortest path | found, length **40**, 64 expanded |
| 2 | `#SG##` | 1 step | length 1 ✔ |
| 3 | goal walled off | failure, no infinite loop | `found=False` after 9 expansions ✔ |
| 4 | a 6-step top route and a 10-step bottom route | 6 | length 6 ✔ (equals BFS) |

Each path is also checked **independently** by `check_path()`. It starts at S, ends at G, every step is a unit move onto a free cell, and the reported length equals the number of steps.

```
#################
#S****#*********#
#.###*#*#######*#
#...#*#*******#*#
###.#*#######*#*#
#...#*********#*#
#.###########.#*#
#.............#G#
#################
```

## Task 4 — Where each concept lives in the code

| Concept | Location in `astar_search.py` |
|---|---|
| State | `(row, col)` tuples; `GridProblem.start` |
| Action | `ACTIONS` dict; `GridProblem.actions(s)` |
| Transition | `GridProblem.result(s, a)` |
| Goal test | `GridProblem.is_goal(s)`, called right after popping in `astar` |
| g(n) | the `g` dict; `g2 = g[s] + problem.cost(...)` |
| h(n) | `manhattan()` (passed in as `h`) |
| f(n) | `f2 = g2 + h(s2, goal)`, which is the heap key |
| Frontier | `frontier`, a `heapq` min-heap |
| Visited states | `closed` set (expanded) plus the `g` dict (best cost seen) |
| Path reconstruction | `reconstruct(parent, s)` |

(a) The frontier is a binary min-heap (`heapq`) of `(f, counter, state)`.
(b) `heappop` returns the entry with the smallest f. Ties are broken FIFO by the counter.
(c) The heuristic is computed in `h(s2, goal)` when a successor is pushed, and once for the start node.
(d) Yes: `f2 = g2 + h(s2, goal)` is computed explicitly.
(e) There are two mechanisms. A successor is pushed only if it gives a strictly lower g than any seen so far. Stale heap entries for states already in `closed` are skipped. So every state is expanded at most once.

## Task 5 — A* vs BFS (original warehouse)

| Measure | BFS | A* (Manhattan) |
|---|---|---|
| Solution found | yes | yes |
| Path length | 40 | 40 |
| States expanded | 64 | 64 |

(a) Both found a solution. (b) The paths have the same length, because both algorithms are optimal for unit costs (BFS by level order, A* because Manhattan distance is admissible).
(c) and (d) On **this** map, A* expanded **no fewer** states. The map has exactly 64 free cells, and both algorithms expanded all of them. The reason is structural. The map is a narrow maze whose only route to G is 40 steps long, while G is only 20 steps away in Manhattan distance. The heuristic is therefore weak everywhere. Worse, it actively points the search toward the dead-end bottom corridor (row 7), which lies close to G in Manhattan terms but does not connect to it. Every cell has f ≤ 40, so an optimal A* must expand them all before it can confirm the goal.

A* saves work when h is **informative**. On the open-floor map below it expanded 47 states, against 72 for uninformed search. That is the right way to state the conclusion: A* is not automatically better. Its advantage depends on how closely h tracks the true remaining cost h\*.

## Task 6 — Heuristic investigation

**Why Manhattan distance is appropriate.** With 4-connected unit-cost moves, every path must contain at least |Δrow| vertical moves and |Δcol| horizontal moves. Obstacles can only add steps. So Manhattan distance is a lower bound on h\*, which means it is admissible. It is also consistent, because one move changes it by exactly ±1 while costing 1.

Original warehouse map (optimal = 40):

| h(n) | found | length | expanded |
|---|---|---|---|
| 0 | ✔ | 40 | 64 |
| Manhattan | ✔ | 40 | 64 |
| Euclidean | ✔ | 40 | 64 |
| 2 × Manhattan | ✔ | 40 | 64 |

All of them give the same result on this map, because there is only one route to G and every state must be expanded (see Task 5).

Open-floor map (in `experiments.py`, optimal = 18). This map has many alternative routes, so the differences become visible:

| h(n) | found | length | expanded | optimal? |
|---|---|---|---|---|
| 0 (uniform-cost / Dijkstra ≈ BFS) | ✔ | 18 | 72 | ✔ |
| Euclidean | ✔ | 18 | 55 | ✔ |
| Manhattan | ✔ | 18 | 47 | ✔ |
| 2 × Manhattan | ✔ | **20** | **33** | ✘ |

Interpretation, based on these experiments:

- **h = 0** turns A* into uniform-cost search. It is still optimal, but it has no guidance, so it expands the most states.
- **Euclidean** is admissible (the straight line is never longer than any grid path) but is **smaller** than Manhattan whenever both coordinates differ. It is less informed, so it expands more states than Manhattan while remaining optimal.
- **Manhattan** is the most informed of the admissible heuristics tested, and it is the best here.
- **2 × Manhattan** **overestimates** (h > h\* near the goal), so it is not admissible. The search becomes greedier: it expanded 30% fewer states, but it returned a **suboptimal** path of 20 instead of 18. When h is too aggressive, A* can pop G with a non-minimal g before a cheaper route has been explored. (For weight w, weighted A* is still bounded: the cost is at most w × the optimum.)

Summary: the more closely an admissible h approaches h\*, the fewer nodes A* expands. Once h exceeds h\*, the optimality guarantee is lost in exchange for speed.

## Task 7 — Evaluating the LLM-generated agent

1. **Correct immediately:** the grid parsing, successor generation, heap-based frontier, and path reconstruction.
2. **Bugs or design problems to watch for (and checked):** a goal test placed at *generation* time instead of *expansion* time. This is the most common LLM mistake and breaks optimality for weighted or inadmissible cases. Another is a `visited` set that blocks re-opening a node when a cheaper g is found later. My version pushes whenever the g is strictly better and skips stale entries. The first experiment also had a bug on my side: the first "open room" map I tried had no solution at all. The `check_path` assertion caught it immediately.
3. **How they were found:** by writing tests with *known* answers (the trivial map, the unreachable map, the two-route map), and by checking every returned path independently of the search code.
4. **Unfamiliar terminology:** the `itertools.count()` tie-breaker. It is needed because Python cannot compare two tuples whose earlier elements are equal and whose later elements are not orderable. It also makes the order in which equal-f nodes are expanded deterministic.
5. **Modifications:** a pluggable `h` argument, `expanded` counting, `scaled()` for the weighted experiments, and `check_path()`.
6. **Most useful tests:** the no-solution map (termination) and the two-route map (optimality). Beyond those, the heuristic experiments on the open-floor map were the only ones that showed a *difference* between variants.
7. **Could it be trusted untested?** No. The original map alone would not have revealed an optimality bug, because there is only one route to G.
8. **What I learned about A\*:** its advantage depends entirely on map structure and heuristic quality. On a tight maze, A* with an admissible heuristic can do exactly as much work as BFS.

## Final reflection

1. **Formulate first.** The formulation fixes what a state is, which actions are legal, what counts as success, and what is being optimised. Without it, generated code can quietly solve a different problem, for example allowing diagonal moves or treating any path as a solution. With it, there is a concrete standard against which to test the code.
2. **Why A\* is "informed".** It uses problem-specific knowledge, h(n), an estimate of the remaining cost, to decide which frontier node to expand next. BFS uses only depth.
3. **Why the heuristic matters.** It determines both **efficiency** (how many nodes are expanded; a closer admissible h means fewer) and **correctness of optimality** (an admissible or consistent h guarantees an optimal path, while an overestimating h can return a worse one, as the 2× experiment showed).
4. **What the LLM contributed.** A fast first implementation of the standard parts (heap frontier, parent pointers, rendering) and an explanation of the Manhattan heuristic. The problem formulation, the choice of tests, and the interpretation of the results were not delegated.
5. **Risks of accepting untested code.** Subtle, plausible-looking errors: a misplaced goal test, a visited set that blocks improvements, infinite loops on unsolvable maps, off-by-one path lengths, or a heuristic with swapped coordinates. None of these prevents the program from printing *a* path, and all of them make it wrong.
