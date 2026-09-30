"""Tests for the goal-based warehouse agent.  Run:  python3 test_warehouse_agent.py"""

from warehouse_agent import WAREHOUSE_MAP, WarehouseEnvironment, ACTIONS, run


def check_path_is_legal(text_map, cells):
    env = WarehouseEnvironment(text_map)
    assert cells[0] == env.start and cells[-1] == env.goal
    for a, b in zip(cells, cells[1:]):
        assert env.is_free(b), f"{b} is an obstacle"
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, f"{a}->{b} is not a single move"


def test_original_map():
    actions, cells, _ = run(WAREHOUSE_MAP, verbose=False)
    check_path_is_legal(WAREHOUSE_MAP, cells)
    # Manhattan distance S->G is 18; the wall at (1,6) forces a 2-step detour.
    assert len(actions) == 20


def test_adjacent_goal():
    m = "####\n#SG#\n####"
    actions, _, _ = run(m, verbose=False)
    assert actions == ["Right"]


def test_unreachable_goal():
    m = "#######\n#S.#.G#\n#######"
    assert run(m, verbose=False) is None


def test_shortest_of_two_routes():
    m = ("#######\n"
         "#S...G#\n"
         "#.###.#\n"
         "#.....#\n"
         "#######")
    actions, cells, _ = run(m, verbose=False)
    check_path_is_legal(m, cells)
    assert len(actions) == 4          # top corridor, not the 8-step bottom loop


def test_collision_is_rejected():
    env = WarehouseEnvironment(WAREHOUSE_MAP)
    try:
        env.execute("Up")             # S is directly under the outer wall
    except RuntimeError:
        return
    raise AssertionError("Environment allowed a move into a wall")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
