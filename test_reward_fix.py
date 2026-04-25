#!/usr/bin/env python3
"""
Test script to validate reward calculation fix.
Verifies that the server correctly normalizes scores and client doesn't double-clamp.
"""

import sys
sys.path.insert(0, ".")

from app.graders.master_grader import calculate_master_reward

def test_reward_calculation():
    """Test case from bug report: should return ~0.615, not 0.999"""
    
    print("="*70)
    print("REWARD CALCULATION FIX VALIDATION TEST")
    print("="*70)
    
    # Test case 1: Partial match (from bug report)
    print("\n[TEST 1] Partial Match - Expected ~0.615 (no anti-cheat)")
    print("-" * 70)
    
    action = {
        "anomaly_detected": True,
        "agent_id": "DRIVER-04",  # Correct agent_id to avoid anti-cheat
        "severity": "medium",      # Wrong severity
        "summary": "Driver performed unauthorized route"  # Missing keywords
    }
    
    ground_truth = {
        "anomaly_detected": True,
        "agent_id": "DRIVER-04",
        "severity": "high",
        "issue_keywords": ["GPS disabled", "route deviation"]
    }
    
    result = calculate_master_reward(action, ground_truth)
    
    print(f"Action: {action}")
    print(f"\nBreakdown:")
    for key, value in result["breakdown"].items():
        print(f"  {key:25s}: {value:+.1f}")
    
    # Calculate expected raw score
    raw_score = sum(result["breakdown"].values())
    expected_normalized = raw_score / 2.6
    
    print(f"\nRaw Score Sum: {raw_score:.1f}")
    print(f"Expected Normalized (raw/2.6): {expected_normalized:.4f}")
    print(f"Actual Score: {result['score']:.4f}")
    
    # Validation
    if abs(result['score'] - expected_normalized) < 0.001:
        print("✅ PASS: Score matches expected normalized value")
    else:
        print(f"❌ FAIL: Score {result['score']:.4f} != Expected {expected_normalized:.4f}")
        return False
    
    if result['score'] < 0.7:  # Should be ~0.615, definitely not 0.999
        print("✅ PASS: Score is realistic (not stuck at 0.999)")
    else:
        print(f"❌ FAIL: Score {result['score']:.4f} is too high (stuck at 0.999?)")
        return False
    
    # Test case 2: Perfect match
    print("\n" + "="*70)
    print("[TEST 2] Perfect Match - Expected ~0.999")
    print("-" * 70)
    
    action_perfect = {
        "anomaly_detected": True,
        "agent_id": "DRIVER-04",
        "severity": "high",
        "summary": "Driver performed unauthorized route deviation and GPS disabled"
    }
    
    result_perfect = calculate_master_reward(action_perfect, ground_truth)
    
    print(f"Action: {action_perfect}")
    print(f"\nBreakdown:")
    for key, value in result_perfect["breakdown"].items():
        print(f"  {key:25s}: {value:+.1f}")
    
    raw_score_perfect = sum(result_perfect["breakdown"].values())
    expected_perfect = min(0.999, raw_score_perfect / 2.6)  # Server clamps to 0.999 max
    
    print(f"\nRaw Score Sum: {raw_score_perfect:.1f}")
    print(f"Expected Normalized (raw/2.6): {expected_perfect:.4f}")
    print(f"Actual Score: {result_perfect['score']:.4f}")
    
    if abs(result_perfect['score'] - expected_perfect) < 0.001:
        print("✅ PASS: Perfect score matches expected")
    else:
        print(f"❌ FAIL: Score {result_perfect['score']:.4f} != Expected {expected_perfect:.4f}")
        return False
    
    if result_perfect['score'] > 0.95:
        print("✅ PASS: Perfect match gives high score")
    else:
        print(f"❌ FAIL: Perfect match score {result_perfect['score']:.4f} is too low")
        return False
    
    # Test case 3: Missed anomaly (negative score)
    print("\n" + "="*70)
    print("[TEST 3] Missed Anomaly - Expected ~0.001 (clamped)")
    print("-" * 70)
    
    action_miss = {
        "anomaly_detected": False,
        "agent_id": "",
        "severity": "low",
        "summary": "No issues detected"
    }
    
    result_miss = calculate_master_reward(action_miss, ground_truth)
    
    print(f"Action: {action_miss}")
    print(f"\nBreakdown:")
    for key, value in result_miss["breakdown"].items():
        print(f"  {key:25s}: {value:+.1f}")
    
    raw_score_miss = sum(result_miss["breakdown"].values())
    expected_miss = max(0.001, min(0.999, raw_score_miss / 2.6))
    
    print(f"\nRaw Score Sum: {raw_score_miss:.1f}")
    print(f"Expected Normalized (clamped): {expected_miss:.4f}")
    print(f"Actual Score: {result_miss['score']:.4f}")
    
    if result_miss['score'] == 0.001:
        print("✅ PASS: Missed anomaly correctly clamped to minimum")
    else:
        print(f"❌ FAIL: Missed anomaly score {result_miss['score']:.4f} should be 0.001")
        return False
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✅")
    print("="*70)
    print("\nSummary:")
    print(f"  Partial match:  {result['score']:.4f} (expected ~0.615)")
    print(f"  Perfect match:  {result_perfect['score']:.4f} (expected ~0.999)")
    print(f"  Missed anomaly: {result_miss['score']:.4f} (expected 0.001)")
    print("\nReward calculation is working correctly!")
    
    return True

if __name__ == "__main__":
    success = test_reward_calculation()
    sys.exit(0 if success else 1)
