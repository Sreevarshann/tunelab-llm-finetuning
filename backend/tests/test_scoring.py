from tunelab.scoring import aggregate, aggregate_text, parse_json_object, score, score_text, token_f1

F = ["a", "b"]
REF = '{"a": 1, "b": "x"}'


def test_parse_accepts_plain_and_fenced_objects_only():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object("Sure! {\"a\": 1}") is None  # no repair of chatty output
    assert parse_json_object("[1, 2]") is None
    assert parse_json_object("not json") is None


def test_score_fields_and_schema():
    s = score('{"a": 1, "b": "x"}', REF, F)
    assert s.parsed and s.exact_keys and s.all_correct
    s = score('{"a": 1, "b": "y", "c": 0}', REF, F)
    assert s.parsed and not s.exact_keys and s.fields == {"a": True, "b": False} and not s.all_correct
    s = score("oops", REF, F)
    assert not s.parsed and s.fields == {"a": False, "b": False}


def test_aggregate_percentages():
    m = aggregate([score(REF, REF, F), score('{"a": 1, "b": "y"}', REF, F), score("x", REF, F), score(REF, REF, F)], F)
    assert m == {"n": 4, "json_valid_pct": 75.0, "schema_exact_pct": 75.0, "all_fields_correct_pct": 50.0, "field_accuracy_pct": {"a": 75.0, "b": 50.0}}
    assert aggregate([], F) == {"n": 0}


def test_text_metrics():
    assert score_text("The Answer.", "the answer") == {"exact": True, "f1": 1.0}
    assert 0 < token_f1("red blue", "blue green") < 1
    assert token_f1("", "") == 1.0
    assert aggregate_text([{"exact": True, "f1": 1.0}, {"exact": False, "f1": 0.5}]) == {"n": 2, "exact_match_pct": 50.0, "token_f1_pct": 75.0}
