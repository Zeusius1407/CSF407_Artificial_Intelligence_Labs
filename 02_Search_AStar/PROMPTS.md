# LLM prompts used (Lab 2)

LLM used: Claude (Anthropic).

## Prompt 1: A* generation (based on my Task 0/1 design)
> I am implementing a simple goal-based search agent in Python. The environment is a grid given as an ASCII map. The agent starts at S and must reach G. `#` is an obstacle and `.` is a free cell. Moves are Up, Down, Left, Right, each with cost 1.
> Implement A* with the Manhattan heuristic h(n) = |x − x_G| + |y − y_G|. Represent states as (row, col) tuples. Use a `GridProblem` class exposing actions(s), result(s,a), is_goal(s), cost(s,a,s2). Use a heapq frontier keyed on f = g + h, apply the goal test when a node is popped (not when it is generated), never expand a state twice, reconstruct the path with parent pointers, and report found / path / length / states expanded. Make the heuristic a parameter.

## Prompt 2: BFS version
> Add a BFS function for the same GridProblem that returns the same dictionary (found, path, length, expanded).

## Prompt 3: experiments
> Write an experiments script that (1) runs the four lab tests with assertions, including an independent path-validity check, (2) prints a BFS vs A* table, and (3) runs A* with h=0, Manhattan, Euclidean and 2×Manhattan, reporting found / length / expanded / optimal (compared with BFS).

## Changes and verification done by hand
- I noticed that on the given warehouse map all heuristics expanded all 64 cells. I then added a second, open-floor map with several alternative routes so that the heuristic differences can actually be observed.
- The first open-floor map I tried was unsolvable. The `check_path` assertion caught this, and I replaced the map.
- I verified the 2×Manhattan suboptimal path by rendering both paths and counting the steps.
