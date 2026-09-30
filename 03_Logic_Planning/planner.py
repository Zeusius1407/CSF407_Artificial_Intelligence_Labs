"""
CSF407 - Logic Lab: a propositional (STRIPS-style) planner with BFS.

    Logic + Search = Planning

* LOGIC  - `applicable(state, action)` decides S |= Preconditions(a)
           `apply(state, action)`      computes S' = (S - Del(a)) ∪ Add(a)
* SEARCH - `bfs_plan` decides which applicable action sequences to try.

A state is a frozenset of ground propositions, e.g. "At(Robot,A)".
Closed-world assumption: any proposition not in the set is false.
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Action:
    name: str
    pos_pre: frozenset = field(default_factory=frozenset)   # must be true
    neg_pre: frozenset = field(default_factory=frozenset)   # must be false
    add: frozenset = field(default_factory=frozenset)       # positive effects
    delete: frozenset = field(default_factory=frozenset)    # negative effects


def A(name, pos_pre=(), neg_pre=(), add=(), delete=()):
    return Action(name, frozenset(pos_pre), frozenset(neg_pre), frozenset(add), frozenset(delete))


# ----------------------------------------------------------------- logic ---
def applicable(state, action):
    """S |= Pre(a): every positive precondition holds and no negative one does."""
    return action.pos_pre <= state and not (action.neg_pre & state)


def apply(state, action):
    """Remove negative effects, then add positive effects."""
    return frozenset((state - action.delete) | action.add)


def satisfies(state, goal):
    return goal <= state


# ---------------------------------------------------------------- search ---
def bfs_plan(initial, actions, goal):
    """Breadth-first search over states. Returns (plan, states) or None."""
    initial = frozenset(initial)
    goal = frozenset(goal)
    frontier = deque([initial])
    parent = {initial: None}            # state -> (previous state, action)
    while frontier:
        s = frontier.popleft()
        if satisfies(s, goal):
            plan, states = [], [s]
            while parent[s] is not None:
                prev, a = parent[s]
                plan.append(a)
                states.append(prev)
                s = prev
            return plan[::-1], states[::-1]
        for a in actions:
            if applicable(s, a):
                s2 = apply(s, a)
                if s2 not in parent:
                    parent[s2] = (s, a)
                    frontier.append(s2)
    return None


def validate_plan(initial, plan, goal):
    """Independent checker: re-executes the plan from scratch."""
    s = frozenset(initial)
    for i, a in enumerate(plan):
        if not applicable(s, a):
            missing = a.pos_pre - s
            return False, f"step {i+1} {a.name}: preconditions not satisfied, missing {set(missing)}"
        s = apply(s, a)
    if not satisfies(s, goal):
        return False, f"goal {set(goal - s)} not reached"
    return True, "valid"


# -------------------------------------------------- warehouse domain -------
LOCATIONS = ["A", "B", "C"]
CONNECTIONS = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]


def warehouse_actions(include_pickup=True, extra=()):
    acts = []
    for x, y in CONNECTIONS:
        acts.append(A(f"Move({x},{y})",
                      pos_pre=[f"At(Robot,{x})"],
                      add=[f"At(Robot,{y})"],
                      delete=[f"At(Robot,{x})"]))
    for l in LOCATIONS:
        if include_pickup:
            acts.append(A(f"PickUp(Package,{l})",
                          pos_pre=[f"At(Robot,{l})", f"At(Package,{l})"],
                          neg_pre=["Holding(Package)"],
                          add=["Holding(Package)"],
                          delete=[f"At(Package,{l})"]))
        acts.append(A(f"Drop(Package,{l})",
                      pos_pre=[f"At(Robot,{l})", "Holding(Package)"],
                      add=[f"At(Package,{l})"],
                      delete=["Holding(Package)"]))
    return acts + list(extra)


INITIAL = {"At(Robot,A)", "At(Package,A)"}
GOAL = {"At(Package,C)"}


def fmt(state):
    return "{" + ", ".join(sorted(state)) + "}"


def solve_and_print(title, initial, actions, goal):
    print(f"\n=== {title} ===")
    print("Initial:", fmt(initial))
    print("Goal   :", fmt(goal))
    result = bfs_plan(initial, actions, goal)
    if result is None:
        print("No plan found")
        return None
    plan, states = result
    print("Plan   :", ", ".join(a.name for a in plan))
    for i, s in enumerate(states):
        label = "S0" if i == 0 else f"S{i} (after {plan[i-1].name})"
        print(f"  {label:<32} {fmt(s)}")
    ok, msg = validate_plan(initial, plan, goal)
    print("Independent validation:", msg)
    return plan, states


if __name__ == "__main__":
    solve_and_print("Warehouse delivery", INITIAL, warehouse_actions(), GOAL)
