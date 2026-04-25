import json


def calculate_master_reward(agent_action: dict, ground_truth: dict) -> dict:
    """
    Multi-Dimensional Reward System for FleetWatch.
    
    FIXED: Proper multi-agent matching, difficulty-adjusted scoring, realistic reward distribution.

    Scoring breakdown:
      +0.3   Valid JSON format (base score)
      +1.0   Correct anomaly detection
      -0.5   False positive (detected anomaly when there is none)
      -1.0   Missed anomaly (failed to detect a real anomaly)
      +0.5   Agent ID match (fuzzy match for multi-agent cases)
      +0.3   Severity match (case-insensitive)
      +0.5   Explanation quality — summary contains issue keywords
      -0.3   Anti-cheat: anomaly_detected=True but no agent_id provided (reduced from -1.0)

    Raw score is normalised against a max theoretical of 2.6 and clamped
    strictly to (0.001, 0.999).
    """
    MAX_THEORETICAL = 2.6

    # ------------------------------------------------------------------
    # Guard: non-dict input
    # ------------------------------------------------------------------
    if not isinstance(agent_action, dict):
        return {
            "score": 0.001,
            "breakdown": {"format_penalty": -1.0},
            "feedback": "Invalid action: expected a dict. Format penalty applied.",
        }

    raw_score = 0.0
    breakdown: dict = {}
    feedback_parts: list[str] = []

    # ------------------------------------------------------------------
    # 1. Base score — valid JSON format
    # ------------------------------------------------------------------
    breakdown["valid_json"] = 0.3
    raw_score += 0.3
    feedback_parts.append("Valid JSON format (+0.3).")

    # ------------------------------------------------------------------
    # 2. Anomaly detection
    # ------------------------------------------------------------------
    predicted = agent_action.get("anomaly_detected")
    expected  = ground_truth.get("anomaly_detected")

    correctly_detected = False

    if predicted == expected:
        breakdown["anomaly_detection"] = 1.0
        raw_score += 1.0
        correctly_detected = True
        feedback_parts.append("Correct anomaly detection (+1.0).")
    elif predicted is True and expected is False:
        breakdown["anomaly_detection"] = -0.5
        raw_score -= 0.5
        feedback_parts.append("False positive: anomaly flagged when none exists (-0.5).")
    elif predicted is False and expected is True:
        breakdown["anomaly_detection"] = -1.0
        raw_score -= 1.0
        feedback_parts.append("Missed anomaly: failed to detect a real anomaly (-1.0).")
    else:
        breakdown["anomaly_detection"] = 0.0
        feedback_parts.append("Anomaly detection result unclear (0.0).")

    # ------------------------------------------------------------------
    # 3. Deep checks — only when anomaly was correctly detected
    # ------------------------------------------------------------------
    if correctly_detected and expected is True:

        # 3a. Agent identification (FIXED: fuzzy matching for multi-agent)
        predicted_agent_id = str(agent_action.get("agent_id", "")).strip()
        expected_agent_id  = str(ground_truth.get("agent_id", "")).strip()

        # Parse multi-agent IDs (comma-separated)
        expected_agents = [a.strip() for a in expected_agent_id.split(",")]
        predicted_agents = [a.strip() for a in predicted_agent_id.split(",")]
        
        # Calculate match score
        if predicted_agent_id and expected_agent_id:
            # Check if any predicted agent matches any expected agent
            matches = sum(1 for pred in predicted_agents if any(pred == exp for exp in expected_agents))
            total_expected = len(expected_agents)
            
            if matches == total_expected and len(predicted_agents) == total_expected:
                # Perfect match: all agents identified correctly
                breakdown["agent_identification"] = 0.5
                raw_score += 0.5
                feedback_parts.append("All agents correctly identified (+0.5).")
            elif matches > 0:
                # Partial match: some agents correct
                partial_score = 0.5 * (matches / total_expected)
                breakdown["agent_identification"] = partial_score
                raw_score += partial_score
                feedback_parts.append(f"Partial agent match: {matches}/{total_expected} correct (+{partial_score:.2f}).")
            else:
                breakdown["agent_identification"] = 0.0
                feedback_parts.append("Agent ID mismatch (0.0).")
        else:
            breakdown["agent_identification"] = 0.0
            feedback_parts.append("Agent ID missing (0.0).")

        # 3b. Severity accuracy (case-insensitive)
        predicted_severity = str(agent_action.get("severity", "")).strip().lower()
        expected_severity  = str(ground_truth.get("severity", "")).strip().lower()

        if predicted_severity and predicted_severity == expected_severity:
            breakdown["severity_accuracy"] = 0.3
            raw_score += 0.3
            feedback_parts.append("Severity level correct (+0.3).")
        else:
            breakdown["severity_accuracy"] = 0.0
            feedback_parts.append("Severity level incorrect or missing (0.0).")

        # 3c. Explanation quality — summary contains issue keywords (FIXED: partial credit)
        summary   = str(agent_action.get("summary", "")).lower()
        keywords  = [kw.lower() for kw in ground_truth.get("issue_keywords", [])]
        
        if keywords:
            # Count how many keywords are present
            keyword_matches = sum(1 for kw in keywords if kw in summary)
            keyword_ratio = keyword_matches / len(keywords)
            
            if keyword_ratio >= 0.5:
                # At least half the keywords present
                explanation_score = 0.5 * keyword_ratio
                breakdown["explanation_quality"] = explanation_score
                raw_score += explanation_score
                feedback_parts.append(f"Summary contains {keyword_matches}/{len(keywords)} keywords (+{explanation_score:.2f}).")
            else:
                breakdown["explanation_quality"] = 0.0
                feedback_parts.append(f"Summary missing key keywords: {keyword_matches}/{len(keywords)} (0.0).")
        else:
            breakdown["explanation_quality"] = 0.0

    else:
        breakdown["agent_identification"] = 0.0
        breakdown["severity_accuracy"]    = 0.0
        breakdown["explanation_quality"]  = 0.0

    # ------------------------------------------------------------------
    # 4. Anti-cheat penalty (REDUCED: -0.3 instead of -1.0)
    #    Fires when anomaly_detected=True but no agent_id is provided.
    # ------------------------------------------------------------------
    if agent_action.get("anomaly_detected") is True and not str(agent_action.get("agent_id", "")).strip():
        breakdown["anti_cheat_penalty"] = -0.3
        raw_score -= 0.3
        feedback_parts.append("Anti-cheat: anomaly flagged without agent_id (-0.3).")
    else:
        breakdown["anti_cheat_penalty"] = 0.0

    # ------------------------------------------------------------------
    # 5. Normalise and clamp to (0.001, 0.999)
    # ------------------------------------------------------------------
    normalized_score = raw_score / MAX_THEORETICAL
    final_score = max(0.001, min(0.999, normalized_score))

    return {
        "score": round(final_score, 4),
        "breakdown": breakdown,
        "feedback": " ".join(feedback_parts),
        "raw_score": round(raw_score, 2),  # Added for debugging
    }
