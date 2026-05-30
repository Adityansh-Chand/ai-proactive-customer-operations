from agents.sentiment_agent import sentiment_agent
from orchestration.agent_dag import run_dag


def test_negative_complaint_escalates_to_ticket():
    result = run_dag("I am angry about this complaint", customer_id="cust_1")

    assert result["policy"] == "escalate"
    assert result["action"]["type"] == "create_ticket"


def test_delay_risk_offers_credit():
    result = run_dag("My order is delayed", customer_id="cust_2")

    assert result["route"] == "delivery_agent"
    assert result["policy"] == "offer_credit"
    assert result["action"]["type"] == "apply_credit"


def test_tracking_request_sends_tracking_update():
    result = run_dag("Where is my order? Please track it", customer_id="cust_3")

    assert result["route"] == "tracking_agent"
    assert result["action"]["type"] == "send_tracking_update"


def test_sentiment_agent_detects_positive_feedback():
    assert sentiment_agent("Thanks, this was great") == "positive"
