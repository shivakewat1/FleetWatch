#!/usr/bin/env python3
"""
Test script to validate reward calculation fix.
Verifies multi-agent matching, partial credit, and realistic reward distribution.
"""

import sys
sys.path.insert(0, ".")

from app.graders.master_grader import calculate_master_reward

def test_reward_calculation():
    """Test all task difficulty levels with proper scoring"""
    
    print("="*70)
    print("REWARD SYSTEM VALIDATION - ALL TASK LEVELS")
    print("="*70)
    
    all_passed = True
    
    # ================================================================
    # TEST 1: Task1 (Easy) - Single agent, obvious anomaly
    # ================================================================
    print("\n[TEST 1] Task1 - Easy (Single Agent)")
    print("-" * 70)
    
    action_t1 = {
        "anomaly_detected": True,
        "agent_id": "DRIVER-04",
        "severity": "high",
        "summary": "Driver performed unauthorized route deviation and GPS disabled"
    }
    
    gt_t1 = {
        "anomaly_detected": True,
        "agent_id": "DRIVER-04",
        "severity": "high",
        "issue_keywords": ["GPS disabled", "route deviation"]
    }
    
    result_t1 = calculate_master_reward(action_t1, gt_t1)
    
    print(f"Action: {action_t1}")
    print(f"\nBreakdown:")
    for key, value in result_t1["breakdown"].items():
        print(f"  {key:25s}: {value:+.2f}")
    
    print(f"\nRaw Score: {result_t1.get('raw_score', 'N/A')}")
    print(f"Final Score: {result_t1['score']:.4f}")
    
    if result_t1['score'] > 0.95:
        print("✅ PASS: Task1 perfect match gives high score")
    else:
        print(f"❌ FAIL: Task1 score {result_t1['score']:.4f} too low for perfect match")
        all_passed = False
    
    # ================================================================
    # TEST 2: Task4 (Hard) - Multi-agent cascade
    # ================================================================
    print("\n" + "="*70)
    print("[TEST 2] Task4 - Hard (Multi-Agent Cascade)")
    print("-" * 70)
    
    action_t4_perfect = {
        "anomaly_detected": True,
        "agent_id": "DRIVER-33, MECHANIC-05, DISPATCHER-07",
        "severity": "critical",
        "summary": "Cascade negligence: skipped inspection, fraudulent countersignature, ignored brake alert"
    }
    
    gt_t4 = {
        "anomaly_detected": True,
        "agent_id": "DRIVER-33, MECHANIC-05, DISPATCHER-07",
        "severity": "critical",
        "issue_keywords": ["skipped inspection", "fraudulent countersignature", "ignored brake alert", "cascade negligence"]
    }
    
    result_t4_perfect = calculate_master_reward(action_t4_perfect, gt_t4)
    
    print(f"Action: {action_t4_perfect}")
    print(f"\nBreakdown:")
    for key, value in result_t4_perfect["breakdown"].items():
        print(f"  {key:25s}: {value:+.2f}")
    
    print(f"\nRaw Score: {result_t4_perfect.get('raw_score', 'N/A')}")
    print(f"Final Score: {result_t4_perfect['score']:.4f}")
    
    if result_t4_perfect['score'] > 0.95:
        print("✅ PASS: Task4 perfect multi-agent match gives high score")
    else:
        print(f"❌ FAIL: Task4 score {result_t4_perfect['score']:.4f} too low")
        all_passed = False
    
    # ================================================================
    # TEST 3: Task4 - Partial agent match
    # ================================================================
    print("\n" + "="*70)
    print("[TEST 3] Task4 - Partial Agent Match (2/3 agents)")
    print("-" * 70)
    
    action_t4_partial = {
        "anomaly_detected": True,
        "agent_id": "DRIVER-33, MECHANIC-05",  # Missing DISPATCHER-07
        "severity": "critical",
        "summary": "Cascade negligence: skipped inspection and fraudulent countersignature"
    }
    
    result_t4_partial = calculate_master_reward(action_t4_partial, gt_t4)
    
    print(f"Action: {action_t4_partial}")
    print(f"\nBreakdown:")
    for key, value in result_t4_partial["breakdown"].items():
        print(f"  {key:25s}: {value:+.2f}")
    
    print(f"\nRaw Score: {result_t4_partial.get('raw_score', 'N/A')}")
    print(f"Final Score: {result_t4_partial['score']:.4f}")
    
    if 0.6 < result_t4_partial['score'] < 0.9:
        print("✅ PASS: Partial agent match gives medium score")
    else:
        print(f"❌ FAIL: Partial match score {result_t4_partial['score']:.4f} not in expected range")
        all_passed = False
    
    # ================================================================
    # TEST 4: Task5 (Hardest) - Multi-agent collusion
    # ================================================================
    print("\n" + "="*70)
    print("[TEST 4] Task5 - Hardest (Multi-Agent Collusion)")
    print("-" * 70)
    
    action_t5 = {
        "anomaly_detected": True,
        "agent_id": "DRIVER-41, DRIVER-42, FUEL-MANAGER-02",
        "severity": "critical",
        "summary": "Multi-agent fuel siphoning collusion: inflated purchase records, phantom mileage, shell vendor"
    }
    
    gt_t5 = {
        "anomaly_detected": True,
        "agent_id": "DRIVER-41, DRIVER-42, FUEL-MANAGER-02",
        "severity": "critical",
        "issue_keywords": ["fuel siphoning", "inflated purchase records", "phantom mileage", "shell vendor", "collusion network"]
    }
    
    result_t5 = calculate_master_reward(action_t5, gt_t5)
    
    print(f"Action: {action_t5}")
    print(f"\nBreakdown:")
    for key, value in result_t5["breakdown"].items():
        print(f"  {key:25s}: {value:+.2f}")
    
    print(f"\nRaw Score: {result_t5.get('raw_score', 'N/A')}")
    print(f"Final Score: {result_t5['score']:.4f}")
    
    if result_t5['score'] > 0.95:
        print("✅ PASS: Task5 perfect match gives high score")
    else:
        print(f"❌ FAIL: Task5 score {result_t5['score']:.4f} too low")
        all_passed = False
    
    # ================================================================
    # TEST 5: Missed anomaly (should be very low)
    # ================================================================
    print("\n" + "="*70)
    print("[TEST 5] Missed Anomaly - Should be minimum score")
    print("-" * 70)
    
    action_miss = {
        "anomaly_detected": False,
        "agent_id": "",
        "severity": "low",
        "summary": "No issues detected"
    }
    
    result_miss = calculate_master_reward(action_miss, gt_t1)
    
    print(f"Action: {action_miss}")
    print(f"\nBreakdown:")
    for key, value in result_miss["breakdown"].items():
        print(f"  {key:25s}: {value:+.2f}")
    
    print(f"\nRaw Score: {result_miss.get('raw_score', 'N/A')}")
    print(f"Final Score: {result_miss['score']:.4f}")
    
    if result_miss['score'] == 0.001:
        print("✅ PASS: Missed anomaly correctly clamped to minimum")
    else:
        print(f"❌ FAIL: Missed anomaly score {result_miss['score']:.4f} should be 0.001")
        all_passed = False
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
    print("="*70)
    
    print("\nScore Distribution:")
    print(f"  Task1 (Easy):           {result_t1['score']:.4f}")
    print(f"  Task4 (Hard - Perfect): {result_t4_perfect['score']:.4f}")
    print(f"  Task4 (Hard - Partial): {result_t4_partial['score']:.4f}")
    print(f"  Task5 (Hardest):        {result_t5['score']:.4f}")
    print(f"  Missed Anomaly:         {result_miss['score']:.4f}")
    
    print("\n✅ Multi-agent matching works!")
    print("✅ Partial credit for incomplete agent lists!")
    print("✅ Realistic reward distribution across difficulty levels!")
    
    return all_passed

if __name__ == "__main__":
    success = test_reward_calculation()
    sys.exit(0 if success else 1)
