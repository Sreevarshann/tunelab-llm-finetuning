import json

from tunelab.tasks import ticket_triage as tt


def test_generation_is_deterministic_and_sized():
    a, b = tt.generate(seed=3), tt.generate(seed=3)
    assert a == b
    assert {k: len(v) for k, v in a.items()} == {"train": 600, "valid": 80, "test": 120, "test_heldout": 120}
    assert tt.generate(seed=4)["train"] != a["train"]


def test_no_duplicate_inputs_across_splits():
    d = tt.generate()
    inputs = [r["messages"][1]["content"] for rows in d.values() for r in rows]
    assert len(inputs) == len(set(inputs))


def test_labels_are_consistent_with_rendered_text():
    for rows in tt.generate().values():
        for r in rows:
            text, label = r["messages"][1]["content"], json.loads(r["messages"][2]["content"])
            assert list(label) == tt.FIELDS
            assert label["product"] in text
            assert (label["order_id"] is not None) == ("ORD-" in text)
            if label["order_id"]:
                assert label["order_id"] in text
            assert label["refund_requested"] == any(p in text for p in tt.REFUND)
            if label["category"] == "feature_request":
                assert label["priority"] in ("low", "medium")


def test_heldout_split_uses_only_unseen_phrasings():
    d = tt.generate()
    seen_bodies = {b.split("{")[0] for c in tt.BODIES.values() for b in c[: tt.SEEN_PER_CATEGORY]}
    heldout_urgency = {u for v in tt.URGENCY.values() for u in v[tt.SEEN_URGENCY :]}
    for r in d["test_heldout"]:
        text = r["messages"][1]["content"]
        assert any(u in text for u in heldout_urgency)
    for r in d["train"]:
        text = r["messages"][1]["content"]
        assert not any(u in text for u in heldout_urgency)
    assert seen_bodies  # sanity
