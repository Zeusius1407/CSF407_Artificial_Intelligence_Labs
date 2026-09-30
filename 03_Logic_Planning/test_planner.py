"""Task 0 and Task 3 tests for the planner.  Run:  python3 test_planner.py"""

from planner import (A, INITIAL, GOAL, warehouse_actions, applicable, apply,
                     bfs_plan, validate_plan, solve_and_print, fmt)


def by_name(actions, name):
    return next(a for a in actions if a.name == name)


def task0_applicability():
    print("=== Task 0: applicability in the initial state ===")
    acts = warehouse_actions()
    s0 = frozenset(INITIAL)
    for a in acts:
        print(f"  {a.name:<20} applicable={applicable(s0, a)}")
    assert applicable(s0, by_name(acts, "PickUp(Package,A)"))
    assert not applicable(s0, by_name(acts, "Drop(Package,C)"))


def task1_handout_plan_is_invalid():
    """The example action sequence in the handout picks the package up at B,
    but the package is at A. The validator must reject it."""
    print("\n=== Task 1: checking the example sequence from the handout ===")
    acts = warehouse_actions()
    plan = [by_name(acts, n) for n in
            ["Move(A,B)", "PickUp(Package,B)", "Move(B,C)", "Drop(Package,C)"]]
    ok, msg = validate_plan(INITIAL, plan, GOAL)
    print("  Move(A,B), PickUp(Package,B), Move(B,C), Drop(Package,C) ->", msg)
    assert not ok

    plan = [by_name(acts, n) for n in
            ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)"]]
    ok, msg = validate_plan(INITIAL, plan, GOAL)
    print("  PickUp(Package,A), Move(A,B), Move(B,C), Drop(Package,C) ->", msg)
    assert ok


def test_a_solvable():
    plan, _ = solve_and_print("Test A: solvable", INITIAL, warehouse_actions(), GOAL)
    assert [a.name for a in plan] == ["PickUp(Package,A)", "Move(A,B)",
                                      "Move(B,C)", "Drop(Package,C)"]
    assert validate_plan(INITIAL, plan, GOAL)[0]


def test_b_impossible():
    res = solve_and_print("Test B: PickUp removed", INITIAL,
                          warehouse_actions(include_pickup=False), GOAL)
    assert res is None


def test_c_irrelevant_actions():
    # Extra robot-only actions: a direct shortcut A->C and a Recharge action.
    extra = [
        A("Move(A,C)", pos_pre=["At(Robot,A)"], add=["At(Robot,C)"], delete=["At(Robot,A)"]),
        A("Recharge(C)", pos_pre=["At(Robot,C)"], add=["Charged(Robot)"]),
    ]
    acts = warehouse_actions(extra=extra)
    plan, states = solve_and_print("Test C: irrelevant/robot-only actions", INITIAL, acts, GOAL)
    # Robot at C is NOT the goal; the package must be at C.
    robot_only = bfs_plan(INITIAL, acts, {"At(Robot,C)"})[0]
    print("  (For contrast, goal At(Robot,C) gives:", [a.name for a in robot_only], ")")
    assert "At(Package,C)" in states[-1]
    assert [a.name for a in plan] == ["PickUp(Package,A)", "Move(A,C)", "Drop(Package,C)"]
    assert [a.name for a in robot_only] == ["Move(A,C)"]


def test_c2_robot_at_c_is_not_package_at_c():
    # Without Drop the robot can reach C (even holding the package), but the
    # package is never At(Package,C), so no plan may be returned.
    acts = [a for a in warehouse_actions() if not a.name.startswith("Drop")]
    res = solve_and_print("Test C2: Drop removed (robot can reach C, package cannot)",
                          INITIAL, acts, GOAL)
    assert res is None


def test_goal_already_true():
    res = solve_and_print("Extra: goal already satisfied", {"At(Robot,A)", "At(Package,C)"},
                          warehouse_actions(), GOAL)
    assert res[0] == []


if __name__ == "__main__":
    task0_applicability()
    task1_handout_plan_is_invalid()
    test_a_solvable()
    test_b_impossible()
    test_c_irrelevant_actions()
    test_c2_robot_at_c_is_not_package_at_c()
    test_goal_already_true()
    print("\nAll tests passed.")
