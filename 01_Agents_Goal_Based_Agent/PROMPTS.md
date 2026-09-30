# LLM prompts used (Lab 1)

LLM used: Claude (Anthropic).

## Prompt 1: generation

> Write a well-documented Python program implementing a goal-based agent for the warehouse navigation problem below.
>
> ```
> #####################
> #S....#............G#
> #.##....##########..#
> #....##.............#
> #.######.###.#.###..#
> #........#..........#
> #####################
> ```
> `S` = start, `G` = goal, `#` = obstacle, `.` = free. The vehicle moves Up, Down, Left, Right, one cell per move, and every move costs 1.
>
> The program should:
> - represent the warehouse as a two-dimensional grid;
> - separate the environment from the agent (the agent has a state, a goal, a model of the map, and a decision component);
> - determine a collision-free shortest path from S to G and then execute it step by step in the environment;
> - avoid all obstacles; the environment must reject illegal moves;
> - print the path, its length and the number of nodes expanded, or a message if no path exists;
> - explain the search algorithm chosen and why it is appropriate.

## Prompt 2: tests

> Write tests for this agent with known expected answers: the original map (check every step is a legal unit move and the length is optimal), a map where G is adjacent to S, a map where G is unreachable, a map with two routes of different lengths, and a check that the environment rejects a move into a wall.

## What was checked or changed by hand
- I verified that the optimal length is 20 by hand: the Manhattan distance is 18, and the wall at (1,6) forces a 2-move detour.
- I added the `assert env.vehicle == env.goal` check after execution, so that the plan is validated against the environment rather than just printed.
