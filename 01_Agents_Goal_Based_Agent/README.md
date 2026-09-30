# Lab 1 — Constructing a Goal-Based Agent using an LLM

**Files**

| File | Purpose |
|---|---|
| `warehouse_agent.py` | Environment, goal-based agent, BFS planner, execution loop |
| `test_warehouse_agent.py` | Tests: original map, adjacent goal, unreachable goal, two routes, collision |

Run with `python3 warehouse_agent.py` and `python3 test_warehouse_agent.py`.

## Result on the given map

```
Start (1, 1)  Goal (1, 19)
Path length: 20 moves   Nodes expanded: 59
Actions: Right Right Right Down Right Right Right Up Right ×12

#####################
#S***.#************G#
#.##****##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

The Manhattan distance from S to G is 18. The shelf at row 1, column 6 blocks the straight line, and stepping around it costs one move down and one move back up. So 20 is a provable lower bound, and the path found is optimal.

## Task 1 — Understanding the problem

1. **Environment.** A 7×21 grid of cells. Each cell is either free or a shelf. The grid is fully observable, deterministic, static, discrete, and has a single agent.
2. **Goal.** Reach cell G at (row 1, column 19) from S at (1, 1) without entering a `#` cell.
3. **Actions.** Up, Down, Left and Right. Each moves the vehicle one cell. A move is legal only if the target cell is inside the grid and is not an obstacle.
4. **Information the agent maintains.** It keeps its current position, the goal position, and a model of the map (which cells are free and how the actions change position). While planning it also keeps the search frontier, the set of visited cells, and parent pointers for path reconstruction. While executing it keeps the remaining plan.
5. **Why goal-based rather than simple reflex.** A reflex agent maps the current percept straight to an action, for example "if the cell to the right is free, go right". On this map that rule walks into the dead end at (1,5). Choosing the right move requires asking "which action sequence leads to G?", which means reasoning about future states using an explicit goal and a model of the environment. That is exactly what a goal-based agent does.

**Think about it (warehouse twice as large).** BFS is still correct and optimal for unit costs. Its time and memory grow with the number of reachable cells, O(rows × cols). Doubling each dimension gives about 4× as many states. Beyond that, some new difficulties appear. Uninformed search expands many irrelevant cells, and A* with a Manhattan heuristic would focus the search. A real warehouse also has moving obstacles, other vehicles, and non-uniform costs (turns, congestion). These call for replanning, Dijkstra or A*, and multi-agent coordination.

## Task 2 — Agent design

```mermaid
flowchart LR
    E[Environment<br/>warehouse grid] -- percept: position --> S[Current state<br/>(row, col)]
    S --> D[Decision component<br/>BFS planner]
    G[Goal: reach G] --> D
    M[Model: map + legal moves] --> D
    D -- plan / next action --> A[Actuator<br/>Up/Down/Left/Right]
    A -- action --> E
```

| Component | Implementation |
|---|---|
| Environment | `WarehouseEnvironment` holds the true grid and the vehicle's position, and rejects collisions |
| Current state | `(row, col)` tuple, obtained from `env.percept()` |
| Goal | `env.goal`, stored in the agent |
| Actions | the `ACTIONS` dictionary; `successors()` filters out illegal moves |
| Decision-making | `GoalBasedAgent.search()` (BFS), then `act()` pops the next action of the plan |

## Task 3 — Prompt engineering and questions

The prompt used was the suggested specification from the lab sheet, with these constraints added: "use four-connected moves only", "report path length and nodes expanded", and "include a test for an unreachable goal". See [`PROMPTS.md`](PROMPTS.md).

1. **Did the first attempt work?** Yes, the program ran. But "runs" is not the same as "correct". The path was checked by the tests (every step is a unit move onto a free cell, the length equals the lower bound of 20, and the unreachable case reports failure).
2. **How to improve the prompt.** State the cost model ("every move costs 1; return a shortest path"). Ask for a specific output format. Require the plan to be *executed* against the environment, not just printed. Ask for tests with known answers (an adjacent goal, an unreachable goal).
3. **Algorithm chosen.** Breadth-first search.
4. **Why the LLM chose it.** The problem is an unweighted grid with uniform step cost, and in that setting BFS is the textbook choice that is complete and optimal. It is also the most common algorithm in training data for "shortest path in a grid maze". A* would also be optimal with Manhattan distance and would expand fewer nodes. The prompt did not ask for a heuristic, so the simpler algorithm was the natural choice.
