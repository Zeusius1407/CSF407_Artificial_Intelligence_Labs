"""
CSF407 - Agents Lab: A goal-based agent for warehouse navigation.

Architecture (goal-based agent):

    Environment --percept--> [ State (x, y) ] ----+
                                                  |
                              [ Goal  G  ] -------+--> Decision component
                                                  |    (BFS planner: "what
                              [ Model: grid,  ]---+     sequence of actions
                              [ legal moves   ]         reaches the goal?")
                                                  |
    Environment <--action---  [ Actuator ] <------+

The agent keeps an internal model of the warehouse (the grid), its current
position, and an explicit goal. It *plans* by searching over future states
(breadth-first search) and then executes the plan one action at a time,
checking every step against the environment.

Why BFS?  Every move costs exactly 1, so BFS explores states in order of
path length and the first time it reaches G it has a shortest path.
It is complete (finds a path if one exists) on a finite grid, and it is
trivial to implement correctly. On this 7x21 map there are < 100 free cells,
so its memory cost is irrelevant.
"""

from collections import deque

WAREHOUSE_MAP = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""

# Action name -> (d_row, d_col)
ACTIONS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class WarehouseEnvironment:
    """The world. Knows the true grid and where the vehicle actually is."""

    def __init__(self, text_map):
        self.grid = [list(row) for row in text_map.strip("\n").splitlines()]
        self.rows = len(self.grid)
        self.cols = max(len(r) for r in self.grid)
        # pad ragged rows with walls so indexing is always safe
        for r in self.grid:
            r.extend("#" * (self.cols - len(r)))
        self.start = self._find("S")
        self.goal = self._find("G")
        self.vehicle = self.start

    def _find(self, ch):
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == ch:
                    return (r, c)
        raise ValueError(f"Map has no '{ch}'")

    def is_free(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != "#"

    def percept(self):
        """What the agent senses: its own position."""
        return self.vehicle

    def execute(self, action):
        """Apply an action. Illegal moves (into a wall) are rejected."""
        dr, dc = ACTIONS[action]
        nxt = (self.vehicle[0] + dr, self.vehicle[1] + dc)
        if not self.is_free(nxt):
            raise RuntimeError(f"Collision: {action} from {self.vehicle} hits an obstacle")
        self.vehicle = nxt
        return self.vehicle


# --------------------------------------------------------------------------
# Goal-based agent
# --------------------------------------------------------------------------
class GoalBasedAgent:
    def __init__(self, world_model: WarehouseEnvironment, goal):
        self.model = world_model      # internal model: map + legal moves
        self.goal = goal              # explicit goal
        self.state = None             # current believed position
        self.plan = []                # remaining actions
        self.nodes_expanded = 0

    # --- model of how actions change the state -------------------------
    def successors(self, pos):
        for name, (dr, dc) in ACTIONS.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if self.model.is_free(nxt):
                yield name, nxt

    # --- decision-making component: BFS planner ------------------------
    def search(self, start):
        frontier = deque([start])
        parent = {start: None}          # also acts as the visited set
        self.nodes_expanded = 0
        while frontier:
            pos = frontier.popleft()
            self.nodes_expanded += 1
            if pos == self.goal:        # goal test
                return self._reconstruct(parent, pos)
            for action, nxt in self.successors(pos):
                if nxt not in parent:
                    parent[nxt] = (pos, action)
                    frontier.append(nxt)
        return None                     # no path exists

    @staticmethod
    def _reconstruct(parent, pos):
        actions, cells = [], [pos]
        while parent[pos] is not None:
            prev, action = parent[pos]
            actions.append(action)
            cells.append(prev)
            pos = prev
        return list(reversed(actions)), list(reversed(cells))

    # --- agent function: percept -> action ------------------------------
    def act(self, percept):
        self.state = percept
        if self.state == self.goal:
            return None                 # goal achieved: stop
        if not self.plan:
            result = self.search(self.state)
            if result is None:
                return "NoOp"
            self.plan = result[0]
        return self.plan.pop(0)


def render(env, path_cells):
    grid = [row[:] for row in env.grid]
    for r, c in path_cells:
        if grid[r][c] == ".":
            grid[r][c] = "*"
    return "\n".join("".join(row) for row in grid)


def run(text_map, verbose=True):
    env = WarehouseEnvironment(text_map)
    agent = GoalBasedAgent(env, env.goal)

    plan = agent.search(env.start)
    if plan is None:
        if verbose:
            print("No collision-free path exists from S to G.")
        return None
    actions, cells = plan

    # Execute the plan in the environment (sense -> decide -> act loop)
    trace = [env.percept()]
    while True:
        action = agent.act(env.percept())
        if action is None or action == "NoOp":
            break
        trace.append(env.execute(action))

    assert env.vehicle == env.goal, "Agent did not reach the goal"
    if verbose:
        print(f"Start {env.start}  Goal {env.goal}")
        print(f"Path length: {len(actions)} moves   Nodes expanded: {agent.nodes_expanded}")
        print("Actions:", " ".join(actions))
        print("Cells  :", " -> ".join(map(str, cells)))
        print()
        print(render(env, cells))
    return actions, cells, agent.nodes_expanded


if __name__ == "__main__":
    run(WAREHOUSE_MAP)
