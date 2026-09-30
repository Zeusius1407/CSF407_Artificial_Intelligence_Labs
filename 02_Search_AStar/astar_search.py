"""
CSF407 - Search Lab: A* (and BFS) for warehouse robot navigation.

Search problem P = (S, A, T, s0, G, c):
    S  : free grid cells (row, col)
    A  : {Up, Down, Left, Right}
    T  : T((r,c), a) = (r+dr, c+dc) if that cell is free, else a is not applicable
    s0 : cell containing 'S'
    G  : {cell containing 'G'}
    c  : 1 per move

A* expands the frontier node with the smallest f(n) = g(n) + h(n).
"""

import heapq
import itertools
import math
from collections import deque

WAREHOUSE_MAP = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""

ACTIONS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


# --------------------------------------------------------------------------
# Problem
# --------------------------------------------------------------------------
class GridProblem:
    def __init__(self, text_map):
        self.grid = [list(r) for r in text_map.strip("\n").splitlines()]
        self.rows, self.cols = len(self.grid), max(map(len, self.grid))
        for r in self.grid:
            r.extend("#" * (self.cols - len(r)))
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, ch):
        for r, row in enumerate(self.grid):
            if ch in row:
                return (r, row.index(ch))
        raise ValueError(f"no '{ch}' in map")

    def free(self, s):
        r, c = s
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != "#"

    def actions(self, s):                       # A(s): applicable actions
        return [a for a, (dr, dc) in ACTIONS.items() if self.free((s[0] + dr, s[1] + dc))]

    def result(self, s, a):                     # transition T(s, a)
        dr, dc = ACTIONS[a]
        return (s[0] + dr, s[1] + dc)

    def is_goal(self, s):                       # goal test
        return s == self.goal

    def cost(self, s, a, s2):                   # step cost c
        return 1


# --------------------------------------------------------------------------
# Heuristics  (x = column, y = row)
# --------------------------------------------------------------------------
def manhattan(s, g):
    return abs(s[0] - g[0]) + abs(s[1] - g[1])


def euclidean(s, g):
    return math.hypot(s[0] - g[0], s[1] - g[1])


def zero(s, g):
    return 0


def scaled(h, w):
    fn = lambda s, g: w * h(s, g)
    fn.__name__ = f"{w}x{h.__name__}"
    return fn


# --------------------------------------------------------------------------
# Search algorithms
# --------------------------------------------------------------------------
def reconstruct(parent, s):
    path = [s]
    while parent[s] is not None:
        s = parent[s]
        path.append(s)
    return path[::-1]


def astar(problem, h=manhattan):
    """Graph-search A*. Returns dict(found, path, length, expanded)."""
    s0, goal = problem.start, problem.goal
    tie = itertools.count()                     # FIFO tie-break among equal f
    g = {s0: 0}                                 # g(n): best known cost from start
    parent = {s0: None}
    frontier = [(h(s0, goal), next(tie), s0)]   # priority queue keyed by f(n)
    closed = set()                              # states already expanded
    expanded = 0

    while frontier:
        f, _, s = heapq.heappop(frontier)       # node with the smallest f
        if s in closed:                         # stale queue entry
            continue
        closed.add(s)
        expanded += 1
        if problem.is_goal(s):
            path = reconstruct(parent, s)
            return dict(found=True, path=path, length=g[s], expanded=expanded)
        for a in problem.actions(s):
            s2 = problem.result(s, a)
            g2 = g[s] + problem.cost(s, a, s2)
            if s2 not in closed and g2 < g.get(s2, math.inf):
                g[s2] = g2
                parent[s2] = s
                f2 = g2 + h(s2, goal)           # f(n) = g(n) + h(n)
                heapq.heappush(frontier, (f2, next(tie), s2))
    return dict(found=False, path=None, length=None, expanded=expanded)


def bfs(problem):
    s0 = problem.start
    parent = {s0: None}
    frontier = deque([s0])
    expanded = 0
    while frontier:
        s = frontier.popleft()
        expanded += 1
        if problem.is_goal(s):
            path = reconstruct(parent, s)
            return dict(found=True, path=path, length=len(path) - 1, expanded=expanded)
        for a in problem.actions(s):
            s2 = problem.result(s, a)
            if s2 not in parent:
                parent[s2] = s
                frontier.append(s2)
    return dict(found=False, path=None, length=None, expanded=expanded)


def render(problem, path):
    g = [row[:] for row in problem.grid]
    for r, c in path or []:
        if g[r][c] == ".":
            g[r][c] = "*"
    return "\n".join("".join(r) for r in g)


def report(name, res):
    print(f"{name:<22} found={res['found']!s:<5}  length={res['length']!s:<5}  expanded={res['expanded']}")


if __name__ == "__main__":
    P = GridProblem(WAREHOUSE_MAP)
    res = astar(P)
    print("A* (Manhattan) on the warehouse map")
    report("A* Manhattan", res)
    print("Path:", " -> ".join(map(str, res["path"])))
    print(render(P, res["path"]))
