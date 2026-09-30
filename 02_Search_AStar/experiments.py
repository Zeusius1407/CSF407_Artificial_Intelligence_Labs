"""Tests (Task 3), BFS vs A* (Task 5) and the heuristic investigation (Task 6).
Run:  python3 experiments.py
"""

from astar_search import (GridProblem, WAREHOUSE_MAP, astar, bfs, manhattan,
                          euclidean, zero, scaled, render, report)

TRIVIAL = "#####\n#SG##\n#####"

NO_SOLUTION = ("#######\n"
               "#S....#\n"
               "###.###\n"
               "#...#G#\n"
               "#######")

# Two routes: a short one along the top (6 moves) and a long detour below.
ALTERNATIVE = ("#########\n"
               "#S.....G#\n"
               "#.#####.#\n"
               "#.......#\n"
               "#########")

# An open floor with scattered shelves (many routes of different length),
# used to show how heuristic quality/admissibility changes A*'s behaviour.
OPEN_ROOM = ("###############\n"
             "#S#.#.....#...#\n"
             "#.#..#........#\n"
             "#.........###.#\n"
             "##.##.........#\n"
             "#..........#..#\n"
             "#.#....#..#..##\n"
             "#...#..#...#.G#\n"
             "###############")


def check_path(problem, res):
    """Independent validity check for a returned path."""
    p = res["path"]
    assert p[0] == problem.start and p[-1] == problem.goal
    for a, b in zip(p, p[1:]):
        assert problem.free(b)
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    assert res["length"] == len(p) - 1


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    # ---------------- Task 3: tests ----------------
    section("Task 3 - Tests")
    P = GridProblem(WAREHOUSE_MAP)
    r = astar(P); check_path(P, r)
    print("Test 1 (original warehouse)"); report("A*", r)
    print(render(P, r["path"]))

    P = GridProblem(TRIVIAL); r = astar(P); check_path(P, r)
    assert r["length"] == 1
    print("\nTest 2 (trivial)"); report("A*", r)

    P = GridProblem(NO_SOLUTION); r = astar(P)
    assert not r["found"]
    print("\nTest 3 (no solution)"); report("A*", r)

    P = GridProblem(ALTERNATIVE); r = astar(P); check_path(P, r); rb = bfs(P)
    assert r["length"] == rb["length"] == 6
    print("\nTest 4 (alternative paths) - shortest is 6"); report("A*", r)
    print(render(P, r["path"]))

    # ---------------- Task 5: BFS vs A* ----------------
    section("Task 5 - BFS vs A* on the warehouse map")
    P = GridProblem(WAREHOUSE_MAP)
    rb, ra = bfs(P), astar(P)
    print(f"{'Measure':<18}{'BFS':>8}{'A*':>8}")
    print(f"{'Solution found':<18}{rb['found']!s:>8}{ra['found']!s:>8}")
    print(f"{'Path length':<18}{rb['length']:>8}{ra['length']:>8}")
    print(f"{'States expanded':<18}{rb['expanded']:>8}{ra['expanded']:>8}")
    print(f"Free cells in map: {sum(ch != '#' for row in P.grid for ch in row)}")

    # ---------------- Task 6: heuristics ----------------
    heuristics = [zero, manhattan, euclidean, scaled(manhattan, 2)]
    for name, m in [("warehouse map", WAREHOUSE_MAP), ("open-room map", OPEN_ROOM)]:
        section(f"Task 6 - Heuristic investigation ({name})")
        P = GridProblem(m)
        optimal = bfs(P)["length"]
        print(f"(BFS optimal length = {optimal})")
        print(f"{'heuristic':<14}{'found':>7}{'length':>8}{'expanded':>10}{'optimal?':>10}")
        for h in heuristics:
            r = astar(P, h)
            check_path(P, r)
            print(f"{h.__name__:<14}{r['found']!s:>7}{r['length']:>8}{r['expanded']:>10}"
                  f"{str(r['length'] == optimal):>10}")
        if name == "open-room map":
            print("\nPath with 2x Manhattan:")
            print(render(P, astar(P, scaled(manhattan, 2))["path"]))
            print("\nPath with Manhattan:")
            print(render(P, astar(P, manhattan)["path"]))

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
