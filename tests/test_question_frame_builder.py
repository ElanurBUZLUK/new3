from src.tools import _build_question_frame


def _frame(router_input: dict, app_mode: str = "", flags: list[str] | None = None) -> dict:
    question_frame, _ = _build_question_frame(
        router_input=router_input,
        app_mode=app_mode,
        merged_flags=flags or [],
    )
    return question_frame


def test_decision_love_new():
    frame = _frame(
        {
            "question_text": "Yeni bir iliski baslatmali miyim?",
            "domain": "love",
            "sub_intent": "yeni_iliski",
            "time_horizon": "near_future",
        }
    )
    assert frame["question_type"] == "decision"
    assert frame["subject"] == "self"
    assert frame["target"] == "unknown_or_new_person"
    assert frame["uncertainty"]["level"] == "low"


def test_reconciliation_target_ex():
    frame = _frame(
        {
            "question_text": "Eski sevgilim geri doner mi?",
            "domain": "love",
            "sub_intent": "barisma",
            "time_horizon": "near_future",
        }
    )
    assert frame["question_type"] == "reconciliation"
    assert frame["target"] == "ex"


def test_timing_time_window_is_derived():
    frame = _frame(
        {
            "question_text": "Ne zaman is bulurum?",
            "domain": "career",
            "sub_intent": "timing",
            "time_horizon": "near_future",
        }
    )
    assert frame["question_type"] == "timing"
    assert frame["time_window_hint"] in {"weeks_to_3_months", "weeks"}


def test_mind_reading_sets_privacy_risk():
    frame = _frame(
        {
            "question_text": "X beni seviyor mu?",
            "domain": "love",
            "sub_intent": "feelings",
            "time_horizon": "near_future",
        }
    )
    assert frame["question_type"] == "feelings_intentions"
    assert frame["privacy_risk"] is True
    assert "avoid definitive mind-reading; frame as possibilities" in frame["safety_notes"]


def test_generic_ambiguous_triggers_single_clarification():
    frame = _frame(
        {
            "question_text": "Ne yapmaliyim?",
            "domain": "general",
            "sub_intent": "unknown",
            "time_horizon": "open_ended",
        }
    )
    assert frame["question_type"] == "advice"
    assert frame["uncertainty"]["level"] == "high"
    assert frame["clarification"]["needed"] is True
    assert isinstance(frame["clarification"]["question"], str)
    assert len(frame["clarification"]["choices"]) <= 3


def test_third_party_sets_privacy_risk():
    frame = _frame(
        {
            "question_text": "Onun hayatinda baska biri var mi?",
            "domain": "love",
            "sub_intent": "third_party",
            "time_horizon": "near_future",
        }
    )
    assert frame["question_type"] == "third_party"
    assert frame["privacy_risk"] is True

