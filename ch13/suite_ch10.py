import persist


def test_history_round_trip_and_bad_files(tmp_path):
    h = [{"role": "system", "content": "old"},
         {"role": "user", "content": "My cat is Biscuit."},
         {"role": "assistant", "content": "Nice name."}]
    path = tmp_path / "h.json"
    persist.save_history(h, path)
    got = persist.load_history(path, system="new")
    assert got[1:] == h[1:] and got[0]["content"] == "new"
    missing = persist.load_history(tmp_path / "none.json")
    assert len(missing) == 1 and missing[0]["role"] == "system"
    path.write_text("not json")
    assert len(persist.load_history(path)) == 1
    path.write_text('[{"role": "robot"}]')
    assert len(persist.load_history(path)) == 1
