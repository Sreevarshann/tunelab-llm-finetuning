import pytest

from tunelab import datasets as ds


def chat(i, reply=None):
    return {"messages": [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": reply or f'{{"id": {i}}}'}]}


def test_validation_reports_line_numbers():
    with pytest.raises(ds.DatasetError) as e:
        ds.validate_rows([chat(1), {"messages": [{"role": "user", "content": "x"}]}, {"messages": [{"role": "robot", "content": "x"}, {"role": "assistant", "content": "y"}]}, {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": " "}]}])
    assert [m.split(":")[0] for m in e.value.errors] == ["line 2", "line 3", "line 4"]


def test_parse_jsonl_reports_bad_json():
    with pytest.raises(ds.DatasetError, match="line 2: invalid JSON"):
        ds.parse_jsonl('{"messages": []}\n{oops\n')


def test_split_is_deterministic_and_complete():
    rows = [chat(i) for i in range(300)]
    a, b = ds.split_rows(rows), ds.split_rows(list(reversed(rows)))
    assert sum(len(v) for v in a.values()) == 300
    assert {k: sorted(map(str, v)) for k, v in a.items()} == {k: sorted(map(str, v)) for k, v in b.items()}
    assert 15 < len(a["test"]) < 45 and 15 < len(a["valid"]) < 45


def test_scoring_detection():
    assert ds.detect_scoring([chat(1), chat(2)]) == {"mode": "json_fields", "fields": ["id"]}
    assert ds.detect_scoring([chat(1), chat(2, "plain text")]) == {"mode": "text"}
    assert ds.detect_scoring([chat(1), chat(2, '{"other": 1}')]) == {"mode": "text"}
