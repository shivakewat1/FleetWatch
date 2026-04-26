import json


def calculate_master_reward(agent_action: dict, ground_truth: dict) -> dict:
    """
    Ultra-Enhanced Multi-Dimensional Reward System for FleetWatch.

    Scoring breakdown:
      +0.4   Valid JSON format (base score)
      +1.5   Correct anomaly detection
      -0.8   False positive
      -1.5   Missed anomaly (strong penalty for missed fraud)
      +0.8   Agent ID match (multi-agent fuzzy support)
      +0.4   Severity match (with graduated partial credit)
      +0.8   Explanation quality (keyword matching)
      +0.4   Contextual reasoning bonus
      +0.3   Evidence integration bonus
      +0.2   Task-specific complexity bonus (tasks 3/4/5 only)
      -0.2   Anti-cheat: anomaly_detected=True but no agent_id provided

    Raw score is normalised against a task-aware MAX_THEORETICAL:
      - Tasks 1/2 (no complexity bonus): 4.5
      - Tasks 3/4/5 (complexity bonus available): 4.7
    Score is clamped strictly to (0.001, 0.999).
    """
    task_id = ground_truth.get("task_id", "")
    has_complexity_bonus = any(t in task_id for t in ("task3", "task4", "task5", "adversarial", "cascade", "collusion"))
    MAX_THEORETICAL = 4.7 if has_complexity_bonus else 4.5

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
    # 1. Base score — valid JSON format (ENHANCED)
    # ------------------------------------------------------------------
    breakdown["valid_json"] = 0.4
    raw_score += 0.4
    feedback_parts.append("Valid JSON format (+0.4).")

    # ------------------------------------------------------------------
    # 2. Anomaly detection (ENHANCED with stronger signals)
    # ------------------------------------------------------------------
    predicted = agent_action.get("anomaly_detected")
    expected  = ground_truth.get("anomaly_detected")

    correctly_detected = False

    if predicted == expected:
        breakdown["anomaly_detection"] = 1.5
        raw_score += 1.5
        correctly_detected = True
        feedback_parts.append("Correct anomaly detection (+1.5).")
    elif predicted is True and expected is False:
        breakdown["anomaly_detection"] = -0.8
        raw_score -= 0.8
        feedback_parts.append("False positive: anomaly flagged when none exists (-0.8).")
    elif predicted is False and expected is True:
        breakdown["anomaly_detection"] = -1.5
        raw_score -= 1.5
        feedback_parts.append("Missed anomaly: failed to detect a real anomaly (-1.5).")
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
        
        # Calculate match score with enhanced multi-agent support
        if predicted_agent_id and expected_agent_id:
            # Check if any predicted agent matches any expected agent
            matches = sum(1 for pred in predicted_agents if any(pred == exp for exp in expected_agents))
            total_expected = len(expected_agents)
            
            if matches == total_expected and len(predicted_agents) == total_expected:
                # Perfect match: all agents identified correctly
                breakdown["agent_identification"] = 0.8
                raw_score += 0.8
                feedback_parts.append("Perfect agent identification: all agents correct (+0.8).")
            elif matches == total_expected:
                # All expected agents found, but extra agents identified
                partial_score = 0.6
                breakdown["agent_identification"] = partial_score
                raw_score += partial_score
                feedback_parts.append(f"All expected agents found with extras (+{partial_score:.2f}).")
            elif matches > 0:
                # Partial match: some agents correct (enhanced partial credit)
                partial_score = 0.8 * (matches / total_expected)
                breakdown["agent_identification"] = partial_score
                raw_score += partial_score
                feedback_parts.append(f"Partial agent match: {matches}/{total_expected} correct (+{partial_score:.2f}).")
            else:
                breakdown["agent_identification"] = -0.1
                raw_score -= 0.1
                feedback_parts.append("Agent ID mismatch (-0.1).")
        else:
            breakdown["agent_identification"] = 0.0
            feedback_parts.append("Agent ID missing (0.0).")

        # 3b. Severity accuracy (case-insensitive)
        predicted_severity = str(agent_action.get("severity", "")).strip().lower()
        expected_severity  = str(ground_truth.get("severity", "")).strip().lower()

        if predicted_severity and predicted_severity == expected_severity:
            breakdown["severity_accuracy"] = 0.4
            raw_score += 0.4
            feedback_parts.append("Severity level correct (+0.4).")
        elif predicted_severity:
            # Partial credit for close severity levels
            severity_levels = ["low", "medium", "high", "critical"]
            try:
                pred_idx = severity_levels.index(predicted_severity)
                exp_idx = severity_levels.index(expected_severity)
                diff = abs(pred_idx - exp_idx)
                if diff == 1:
                    breakdown["severity_accuracy"] = 0.2
                    raw_score += 0.2
                    feedback_parts.append("Severity level close (+0.2).")
                else:
                    breakdown["severity_accuracy"] = 0.0
                    feedback_parts.append("Severity level incorrect (0.0).")
            except ValueError:
                breakdown["severity_accuracy"] = 0.0
                feedback_parts.append("Invalid severity level (0.0).")
        else:
            breakdown["severity_accuracy"] = 0.0
            feedback_parts.append("Severity level missing (0.0).")

        # 3c. Enhanced explanation quality with evidence integration
        summary   = str(agent_action.get("summary", "")).lower()
        keywords  = [kw.lower() for kw in ground_truth.get("issue_keywords", [])]
        
        if keywords:
            # Count how many keywords are present
            keyword_matches = sum(1 for kw in keywords if kw in summary)
            keyword_ratio = keyword_matches / len(keywords)
            
            if keyword_ratio >= 0.8:
                # Excellent keyword coverage
                explanation_score = 0.8
                breakdown["explanation_quality"] = explanation_score
                raw_score += explanation_score
                feedback_parts.append(f"Excellent explanation with {keyword_matches}/{len(keywords)} keywords (+{explanation_score:.2f}).")
            elif keyword_ratio >= 0.6:
                # Good keyword coverage
                explanation_score = 0.6
                breakdown["explanation_quality"] = explanation_score
                raw_score += explanation_score
                feedback_parts.append(f"Good explanation with {keyword_matches}/{len(keywords)} keywords (+{explanation_score:.2f}).")
            elif keyword_ratio >= 0.3:
                # Adequate keyword coverage
                explanation_score = 0.4
                breakdown["explanation_quality"] = explanation_score
                raw_score += explanation_score
                feedback_parts.append(f"Adequate explanation with {keyword_matches}/{len(keywords)} keywords (+{explanation_score:.2f}).")
            else:
                breakdown["explanation_quality"] = 0.0
                feedback_parts.append(f"Insufficient keywords: {keyword_matches}/{len(keywords)} (0.0).")
        else:
            breakdown["explanation_quality"] = 0.0
        
        # 3d. Enhanced contextual reasoning bonus
        reasoning_indicators = [
            "because", "therefore", "caused", "led to", "resulted in", "due to",
            "pattern", "consistent", "indicates", "suggests", "evidence", "shows",
            "coordinated", "collusion", "together", "multiple", "both", "collaboration",
            "tampering", "manipulation", "cover-up", "deception", "fraudulent"
        ]
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in summary)
        if reasoning_count >= 3:
            breakdown["contextual_reasoning"] = 0.4
            raw_score += 0.4
            feedback_parts.append("Excellent contextual reasoning (+0.4).")
        elif reasoning_count >= 2:
            breakdown["contextual_reasoning"] = 0.3
            raw_score += 0.3
            feedback_parts.append("Good contextual reasoning (+0.3).")
        elif reasoning_count == 1:
            breakdown["contextual_reasoning"] = 0.15
            raw_score += 0.15
            feedback_parts.append("Some contextual reasoning (+0.15).")
        else:
            breakdown["contextual_reasoning"] = 0.0
        
        # 3e. NEW: Evidence integration bonus
        evidence_indicators = [
            "system", "log", "timestamp", "g-force", "speed", "camera", "footage",
            "inspection", "damage", "radio", "unauthorized", "alert", "warning",
            "security", "audit", "trail", "coordination", "paint transfer"
        ]
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in summary)
        if evidence_count >= 3:
            breakdown["evidence_integration"] = 0.3
            raw_score += 0.3
            feedback_parts.append("Strong evidence integration (+0.3).")
        elif evidence_count >= 2:
            breakdown["evidence_integration"] = 0.2
            raw_score += 0.2
            feedback_parts.append("Good evidence integration (+0.2).")
        elif evidence_count == 1:
            breakdown["evidence_integration"] = 0.1
            raw_score += 0.1
            feedback_parts.append("Some evidence integration (+0.1).")
        else:
            breakdown["evidence_integration"] = 0.0
        
        # 3f. NEW: Task-specific complexity bonus
        task_id = ground_truth.get("task_id", "")
        if "task3" in task_id or "adversarial" in task_id:
            # Adversarial task bonus for handling deception
            if any(word in summary for word in ["cover-up", "tampering", "deception", "false"]):
                breakdown["task_complexity_bonus"] = 0.2
                raw_score += 0.2
                feedback_parts.append("Adversarial scenario handling bonus (+0.2).")
            else:
                breakdown["task_complexity_bonus"] = 0.0
        elif "task4" in task_id or "cascade" in task_id:
            # Cascade task bonus for multi-agent chain detection
            if any(word in summary for word in ["cascade", "chain", "negligence", "multiple"]):
                breakdown["task_complexity_bonus"] = 0.2
                raw_score += 0.2
                feedback_parts.append("Cascade scenario handling bonus (+0.2).")
            else:
                breakdown["task_complexity_bonus"] = 0.0
        elif "task5" in task_id or "collusion" in task_id:
            # Collusion task bonus for coordinated fraud detection
            if any(word in summary for word in ["collusion", "coordinated", "shell", "vendor"]):
                breakdown["task_complexity_bonus"] = 0.2
                raw_score += 0.2
                feedback_parts.append("Collusion scenario handling bonus (+0.2).")
            else:
                breakdown["task_complexity_bonus"] = 0.0
        else:
            breakdown["task_complexity_bonus"] = 0.0

    else:
        breakdown["agent_identification"] = 0.0
        breakdown["severity_accuracy"]    = 0.0
        breakdown["explanation_quality"]  = 0.0
        breakdown["contextual_reasoning"] = 0.0
        breakdown["evidence_integration"] = 0.0
        breakdown["task_complexity_bonus"] = 0.0

    # ------------------------------------------------------------------
    # 4. Anti-cheat penalty (BALANCED)
    #    Fires when anomaly_detected=True but no agent_id is provided.
    # ------------------------------------------------------------------
    if agent_action.get("anomaly_detected") is True and not str(agent_action.get("agent_id", "")).strip():
        breakdown["anti_cheat_penalty"] = -0.2
        raw_score -= 0.2
        feedback_parts.append("Anti-cheat: anomaly flagged without agent_id (-0.2).")
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
