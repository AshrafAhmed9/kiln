from tools.prepare_massive_intent import make_record


def test_massive_record_has_a_separable_prompt_and_exact_label():
    record = make_record("Where is my card?", "card_arrival")

    assert record["prompt"].endswith("Intent:")
    assert record["label"] == "card_arrival"
    assert record["text"] == record["prompt"] + " card_arrival\n"
