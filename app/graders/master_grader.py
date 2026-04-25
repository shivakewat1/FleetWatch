import json


def calculate_master_reward(agent_action: dict, ground_truth: dict) -> dict:
    """
    Multi-component reward function for FleetWatch evaluation.

    Components
    ----------
    +0.30  Valid JSON format
    +1.00  Correct anomaly detection  (or -0.50 for false positive)
    +0.50  Agent identification match
    +0.30  Severity accuracy match
    +0.50  Explanation quality  (keyword coverage)
    -1.00  Anti-cheat penalty  (blind True guess with no supporting content)

    Final score is clamped to (0.001, 0.999).
    """
    raw_score = 0.0
    breakdown = {}

    # ------------------------------------------------------------------ #
    # 1. Valid JSON format                                                 #
    # ------------------------------------------------------------------ #
    try:
        if isinstance(agent_action, str):
            agent_action = json.loads(agent_action)
        json.dumps(agent_action)          # confirm it is serialisable
        breakdown["valid_json"] = 0.3
        raw_score += 0.3
    except (TypeError, ValueError):
        breakdown["valid_json"] = 0.0

    # ------------------------------------------------------------------ #
    # 2. Anomaly detection                                                 #
    # ------------------------------------------------------------------ #
    predicted = agent_action.get("anomaly_detected")
    expected  = ground_truth.get("anomaly_detected")

    if predicted is None:
        breakdown["anomaly_detection"] = 0.0
    elif predicted == expected:
        breakdown["anomaly_detection"] = 1.0
        raw_score += 1.0
    else:
        # False positive or false negative
        breakdown["anomaly_detection"] = -0.5
        raw_score -= 0.5

    # ------------------------------------------------------------------ #
    # 3. Agent identification                                              #
    # ------------------------------------------------------------------ #
    predicted_agent  = str(agent_action.get("agent", "")).strip().lower()
    expected_agents  = [
        a.strip().lower()
        for a in str(ground_truth.get("agent", "")).split(",")
    ]

    if predicted_agent and any(ea in predicted_agent for ea in expected_agents):
        breakdown["agent_identification"] = 0.5
        raw_score += 0.5
    else:
        breakdown["agent_identification"] = 0.0

    # ------------------------------------------------------------------ #
    # 4. Severity accuracy                                                 #
    # ------------------------------------------------------------------ #
    predicted_severity = str(agent_action.get("severity", "")).strip().lower()
    expected_severity  = str(ground_truth.get("severity", "")).strip().lower()

    if predicted_severity and predicted_severity == expected_severity:
        breakdown["severity_accuracy"] = 0.3
        raw_score += 0.3
    else:
        breakdown["severity_accuracy"] = 0.0

    # ------------------------------------------------------------------ #
    # 5. Explanation quality  (keyword coverage)                          #
    # ------------------------------------------------------------------ #
    explanation   = str(agent_action.get("explanation", "")).lower()
    keywords      = [kw.lower() for kw in ground_truth.get("keywords", [])]

    if keywords:
        matched = sum(1 for kw in keywords if kw in explanation)
        coverage = matched / len(keywords)
        kw_score = round(coverage * 0.5, 4)
    else:
        kw_score = 0.0

    breakdown["explanation_quality"] = kw_score
    raw_score += kw_score

    # ------------------------------------------------------------------ #
    # 6. Anti-cheat penalty                                               #
    #    Fires when anomaly_detected=True but the action carries no       #
    #    meaningful supporting content (agent, severity, explanation).    #
    # ------------------------------------------------------------------ #
    is_blind_guess = (
        predicted is True
        and not predicted_agent
        and not predicted_severity
        and not explanation
    )

    if is_blind_guess:
        breakdown["anti_cheat_penalty"] = -1.0
        raw_score -= 1.0
    else:
        breakdown["anti_cheat_penalty"] = 0.0

    # ------------------------------------------------------------------ #
    # 7. Normalise & clamp to (0.001, 0.999)                              #
    # ------------------------------------------------------------------ #
    final_score = max(0.001, min(0.999, raw_score))

    return {
        "score": round(final_score, 4),
        "breakdown": breakdown,
    }
