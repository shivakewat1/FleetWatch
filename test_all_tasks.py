#!/usr/bin/env python3
"""
Comprehensive test of all 5 tasks with different response qualities.
Tests reward distribution across difficulty levels.
"""

import requests
import json

BASE_URL = "https://shiva0999-fleet-watch.hf.space"

def test_task(task_num, action, description):
    """Test a specific task and print results"""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"Task: {task_num}")
    print(f"{'='*70}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/test/{task_num}",
            json=action,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        reward = result.get("reward", {})
        breakdown = reward.get("breakdown", {})
        
        print(f"\nInput Action:")
        print(f"  anomaly_detected: {action.get('anomaly_detected')}")
        print(f"  agent_id: {action.get('agent_id')}")
        print(f"  severity: {action.get('severity')}")
        print(f"  summary: {action.get('summary')[:60]}...")
        
        print(f"\nReward Breakdown:")
        for key, value in breakdown.items():
            print(f"  {key:25s}: {value:+.2f}")
        
        print(f"\nScores:")
        print(f"  Raw Score:   {reward.get('raw_score', 'N/A')}")
        print(f"  Final Score: {reward.get('score', 'N/A'):.4f}")
        
        # Verdict
        score = reward.get('score', 0)
        if score >= 0.95:
            verdict = "✅ EXCELLENT"
        elif score >= 0.80:
            verdict = "✅ GOOD"
        elif score >= 0.60:
            verdict = "⚠️  PARTIAL"
        elif score >= 0.40:
            verdict = "⚠️  WEAK"
        else:
            verdict = "❌ POOR"
        
        print(f"\nVerdict: {verdict}")
        
        return score
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return 0.0


def main():
    print("="*70)
    print("FLEETWATCH - COMPREHENSIVE TASK TESTING")
    print("Testing all 5 tasks with varying response qualities")
    print("="*70)
    
    scores = []
    
    # ================================================================
    # TASK 1: Obvious (Easy - Single Agent)
    # ================================================================
    
    # Perfect response
    score = test_task(1, {
        "anomaly_detected": True,
        "agent_id": "DRIVER-04",
        "severity": "high",
        "summary": "DRIVER-04 performed unauthorized route deviation and disabled GPS tracker mid-trip"
    }, "Task1 - Perfect Response")
    scores.append(("Task1-Perfect", score))
    
    # Good response (missing some keywords)
    score = test_task(1, {
        "anomaly_detected": True,
        "agent_id": "DRIVER-04",
        "severity": "high",
        "summary": "Driver deviated from route"
    }, "Task1 - Good Response")
    scores.append(("Task1-Good", score))
    
    # Partial response (wrong severity)
    score = test_task(1, {
        "anomaly_detected": True,
        "agent_id": "DRIVER-04",
        "severity": "medium",
        "summary": "Route deviation detected"
    }, "Task1 - Partial Response")
    scores.append(("Task1-Partial", score))
    
    # ================================================================
    # TASK 2: Pattern (Medium - Repeated Violations)
    # ================================================================
    
    # Perfect response
    score = test_task(2, {
        "anomaly_detected": True,
        "agent_id": "DRIVER-11",
        "severity": "medium",
        "summary": "DRIVER-11 repeatedly clocked out early with falsified timesheet and odometer discrepancy"
    }, "Task2 - Perfect Response")
    scores.append(("Task2-Perfect", score))
    
    # Good response
    score = test_task(2, {
        "anomaly_detected": True,
        "agent_id": "DRIVER-11",
        "severity": "medium",
        "summary": "Early clock-out pattern with timesheet fraud"
    }, "Task2 - Good Response")
    scores.append(("Task2-Good", score))
    
    # ================================================================
    # TASK 3: Adversarial (Hard - Cover-up)
    # ================================================================
    
    # Perfect response
    score = test_task(3, {
        "anomaly_detected": True,
        "agent_id": "DRIVER-22",
        "severity": "critical",
        "summary": "DRIVER-22 covered up collision by tampering logs and coercing witness DRIVER-08 to file false report"
    }, "Task3 - Perfect Response")
    scores.append(("Task3-Perfect", score))
    
    # Partial response (missing witness coercion)
    score = test_task(3, {
        "anomaly_detected": True,
        "agent_id": "DRIVER-22",
        "severity": "critical",
        "summary": "Driver tampered with collision logs"
    }, "Task3 - Partial Response")
    scores.append(("Task3-Partial", score))
    
    # ================================================================
    # TASK 4: Cascade (Hard - Multi-Agent)
    # ================================================================
    
    # Perfect response (all 3 agents)
    score = test_task(4, {
        "anomaly_detected": True,
        "agent_id": "DRIVER-33, MECHANIC-05, DISPATCHER-07",
        "severity": "critical",
        "summary": "Cascade negligence: DRIVER-33 skipped inspection, MECHANIC-05 fraudulent countersignature, DISPATCHER-07 ignored brake alert"
    }, "Task4 - Perfect Response (All 3 Agents)")
    scores.append(("Task4-Perfect", score))
    
    # Good response (2/3 agents)
    score = test_task(4, {
        "anomaly_detected": True,
        "agent_id": "DRIVER-33, MECHANIC-05",
        "severity": "critical",
        "summary": "Skipped inspection with fraudulent countersignature"
    }, "Task4 - Good Response (2/3 Agents)")
    scores.append(("Task4-Good", score))
    
    # Partial response (1/3 agents)
    score = test_task(4, {
        "anomaly_detected": True,
        "agent_id": "DRIVER-33",
        "severity": "critical",
        "summary": "Inspection was skipped"
    }, "Task4 - Partial Response (1/3 Agents)")
    scores.append(("Task4-Partial", score))
    
    # ================================================================
    # TASK 5: Collusion (Hardest - Multi-Agent Fraud)
    # ================================================================
    
    # Perfect response (all 3 agents)
    score = test_task(5, {
        "anomaly_detected": True,
        "agent_id": "DRIVER-41, DRIVER-42, FUEL-MANAGER-02",
        "severity": "critical",
        "summary": "Multi-agent fuel siphoning collusion: DRIVER-41 and DRIVER-42 inflated purchase records with phantom mileage using shell vendor controlled by FUEL-MANAGER-02"
    }, "Task5 - Perfect Response (All 3 Agents)")
    scores.append(("Task5-Perfect", score))
    
    # Good response (2/3 agents)
    score = test_task(5, {
        "anomaly_detected": True,
        "agent_id": "DRIVER-41, DRIVER-42",
        "severity": "critical",
        "summary": "Fuel siphoning with inflated purchases and phantom mileage"
    }, "Task5 - Good Response (2/3 Agents)")
    scores.append(("Task5-Good", score))
    
    # Partial response (1/3 agents)
    score = test_task(5, {
        "anomaly_detected": True,
        "agent_id": "DRIVER-41",
        "severity": "critical",
        "summary": "Fuel fraud detected"
    }, "Task5 - Partial Response (1/3 Agents)")
    scores.append(("Task5-Partial", score))
    
    # ================================================================
    # SUMMARY
    # ================================================================
    
    print("\n" + "="*70)
    print("SUMMARY - ALL TASK SCORES")
    print("="*70)
    
    for name, score in scores:
        bar = "█" * int(score * 50)
        print(f"{name:25s}: {score:.4f} {bar}")
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Group by task
    task_groups = {}
    for name, score in scores:
        task = name.split("-")[0]
        if task not in task_groups:
            task_groups[task] = []
        task_groups[task].append(score)
    
    print("\nAverage Scores by Task:")
    for task in sorted(task_groups.keys()):
        avg = sum(task_groups[task]) / len(task_groups[task])
        print(f"  {task}: {avg:.4f}")
    
    print("\nScore Distribution:")
    perfect_scores = [s for _, s in scores if s >= 0.95]
    good_scores = [s for _, s in scores if 0.80 <= s < 0.95]
    partial_scores = [s for _, s in scores if 0.60 <= s < 0.80]
    weak_scores = [s for _, s in scores if 0.40 <= s < 0.60]
    poor_scores = [s for _, s in scores if s < 0.40]
    
    print(f"  Excellent (≥0.95): {len(perfect_scores)}")
    print(f"  Good (0.80-0.95):  {len(good_scores)}")
    print(f"  Partial (0.60-0.80): {len(partial_scores)}")
    print(f"  Weak (0.40-0.60):  {len(weak_scores)}")
    print(f"  Poor (<0.40):      {len(poor_scores)}")
    
    print("\n" + "="*70)
    print("✅ TESTING COMPLETE")
    print("="*70)
    
    # Overall verdict
    avg_all = sum(s for _, s in scores) / len(scores)
    print(f"\nOverall Average Score: {avg_all:.4f}")
    
    if avg_all >= 0.75:
        print("✅ System is working EXCELLENTLY - Realistic reward distribution!")
    elif avg_all >= 0.60:
        print("✅ System is working WELL - Good reward distribution")
    else:
        print("⚠️  System needs tuning - Rewards may be too harsh")


if __name__ == "__main__":
    main()
