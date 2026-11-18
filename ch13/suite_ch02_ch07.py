import diagonal
import split_photos
import weighted


def test_split_photos(tmp_path):
    for cls in ("a", "b", "c"):
        (tmp_path / "raw" / cls).mkdir(parents=True)
        for i in range(10):
            (tmp_path / "raw" / cls / f"{i}.jpg").write_bytes(b"x")
    counts = split_photos.split(tmp_path / "raw", tmp_path / "out")
    assert counts == {"a": (8, 2), "b": (8, 2), "c": (8, 2)}
    assert len(list((tmp_path / "out" / "val" / "a").iterdir())) == 2


def test_chebyshev_is_exact_on_open_grid():
    grid = ["...."] * 4
    assert diagonal.bfs8(grid, (0, 0), (3, 3))[0] == 3
    assert diagonal.chebyshev((0, 0), (3, 3)) == 3
    assert diagonal.manhattan((0, 0), (3, 3)) == 6  # overestimates


def test_weighted_prefers_cheap_detour():
    start, goal = (0, 0), (0, 4)
    came, cost = weighted.astar_weighted(weighted.COSTLY, start, goal)
    assert cost == 8
    assert len(weighted.path_of(came, goal)) - 1 == 8
