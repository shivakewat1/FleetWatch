"""
Task 3 Specialist Training - Adversarial Scenario Focus
=====================================================
Specialized training for improving Task 3 (adversarial) performance
"""

import os
import sys
import time
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from self_improvement_system import EnhancedFleetWatchAgent

print("""
╔══════════════════════════════════════════════════════════╗
║         🎯 Task 3 Specialist Training                    ║
║                                                          ║
║  🕵️ Adversarial Scenario Focus                           ║
║  🎯 Target: Improve Task 3 from 0.001 → 0.7+            ║
║  🧠 Enhanced Deception Detection                         ║
╚══════════════════════════════════════════════════════════╝
""")

# Configuration
API_BASE = "https://shiva0999-fleet-watch.hf.space"
TRAINING_EPISODES = 100  # Focus on Task 3
TASK3_FOCUS_EPISODES = 80  # 80% on Task 3

class Task3SpecialistAgent(EnhancedFleetWatchAgent):
    """Specialized agent for Task 3 adversarial scenarios"""
    
    def __init__(self):
        super().__init__()
        
        # Task 3 specific knowledge
        self.adversarial_patterns = {
            "deception_indicators": [
                "no incident", "sensor", "glitch", "recalibration", "noise",
                "pothole", "road hazard", "minor", "nothing", "normal"
            ],
            "tampering_evidence": [
                "reset", "diagnostic", "unauthorized", "marked as", "retroactively",
                "altered", "modified", "changed", "deleted", "overwritten"
            ],
            "coercion_patterns": [
                "just say", "help me out", "don't mention", "between us",
                "keep quiet", "say nothing", "cover for", "back me up"
            ],
            "evidence_contradictions": [
                "but", "however", "actually", "confirms", "shows", "reveals",
                "inconsistent", "contradicts", "doesn't match", "different from"
            ],
            "cover_up_indicators": [
                "cover-up", "hide", "conceal", "false report", "lie", "deception",
                "coordination", "planned", "deliberate", "intentional"
            ]
        }
        
        # Enhanced Task 3 detection logic
        self.task3_confidence_boost = 0.3
        
    def analyze_task3_specifically(self, logs, task_desc):
        """Specialized analysis for Task 3 adversarial scenarios"""
        
        combined_text = (logs + " " + task_desc).lower()
        
        # Adversarial scenario detection
        adversarial_score = 0.0
        detected_patterns = []
        
        # Check for deception indicators
        deception_count = sum(1 for indicator in self.adversarial_patterns["deception_indicators"] 
                             if indicator in combined_text)
        if deception_count >= 2:
            adversarial_score += 0.4
            detected_patterns.append("deception_indicators")
        
        # Check for tampering evidence
        tampering_count = sum(1 for indicator in self.adversarial_patterns["tampering_evidence"] 
                             if indicator in combined_text)
        if tampering_count >= 2:
            adversarial_score += 0.5
            detected_patterns.append("tampering_evidence")
        
        # Check for coercion patterns
        coercion_count = sum(1 for indicator in self.adversarial_patterns["coercion_patterns"] 
                            if indicator in combined_text)
        if coercion_count >= 1:
            adversarial_score += 0.4
            detected_patterns.append("coercion_patterns")
        
        # Check for evidence contradictions
        contradiction_count = sum(1 for indicator in self.adversarial_patterns["evidence_contradictions"] 
                                 if indicator in combined_text)
        if contradiction_count >= 2:
            adversarial_score += 0.3
            detected_patterns.append("evidence_contradictions")
        
        # Check for cover-up indicators
        coverup_count = sum(1 for indicator in self.adversarial_patterns["cover_up_indicators"] 
                           if indicator in combined_text)
        if coverup_count >= 1:
            adversarial_score += 0.4
            detected_patterns.append("cover_up_indicators")
        
        return adversarial_score, detected_patterns
    
    def analyze_logs(self, task_data):
        """Enhanced analysis with Task 3 specialization"""
        
        task_id = task_data.get("task_id", "")
        
        if "task3" in task_id or "adversarial" in task_id:
            # Use specialized Task 3 analysis
            logs = " ".join(task_data.get("input_logs", []))
            task_desc = task_data.get("task_description", "")
            
            # Get base analysis
            base_result = super().analyze_logs(task_data)
            
            # Apply Task 3 specialization
            adversarial_score, patterns = self.analyze_task3_specifically(logs, task_desc)
            
            # Enhanced agent extraction for Task 3
            import re
            agent_pattern = r'(DRIVER|MECHANIC|DISPATCHER|SUPERVISOR|MAINTENANCE|FLEET-MGMT|SECURITY|INVESTIGATOR)-(\d+)'
            agent_matches = re.findall(agent_pattern, logs + task_desc)
            all_agents = [f"{match[0]}-{match[1]}" for match in agent_matches]
            
            # Focus on primary actors (usually drivers in adversarial scenarios)
            primary_agents = [agent for agent in all_agents if "DRIVER" in agent]
            
            # If high adversarial score, boost confidence
            if adversarial_score > 1.0:
                enhanced_summary = f"ADVERSARIAL SCENARIO DETECTED: {', '.join(patterns)}. "
                enhanced_summary += f"Evidence shows {base_result['summary']}"
                
                # Multiple agents likely in adversarial scenarios
                if len(primary_agents) >= 2:
                    agent_ids = ", ".join(primary_agents[:2])  # Top 2 agents
                elif len(all_agents) >= 2:
                    agent_ids = ", ".join([agent for agent in all_agents if "DRIVER" in agent or "SUPERVISOR" in agent][:2])
                else:
                    agent_ids = primary_agents[0] if primary_agents else all_agents[0] if all_agents else ""
                
                return {
                    "anomaly_detected": True,
                    "agent_id": agent_ids,
                    "severity": "critical",  # Adversarial scenarios are always critical
                    "summary": enhanced_summary
                }
            else:
                # Use base analysis but enhance for Task 3
                if base_result["anomaly_detected"]:
                    enhanced_summary = f"Potential adversarial behavior: {base_result['summary']}"
                    return {
                        "anomaly_detected": True,
                        "agent_id": base_result["agent_id"],
                        "severity": "high",  # Upgrade severity for Task 3
                        "summary": enhanced_summary
                    }
                else:
                    return base_result
        else:
            # Use base analysis for other tasks
            return super().analyze_logs(task_data)

def test_api_connection():
    """Test if API is available"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def force_task3_reset():
    """Force environment to Task 3 by testing it directly"""
    try:
        response = requests.post(f"{API_BASE}/test/3", 
                               json={
                                   "anomaly_detected": True,
                                   "agent_id": "DRIVER-22, DRIVER-08",
                                   "severity": "critical",
                                   "summary": "Test call to get Task 3"
                               }, timeout=10)
        if response.status_code == 200:
            # Now reset to get actual Task 3
            response = requests.post(f"{API_BASE}/reset", timeout=10)
            if response.status_code == 200:
                return response.json()
        return None
    except Exception as e:
        print(f"⚠️ Force Task 3 failed: {e}")
        return None

def run_task3_specialist_training():
    """Run specialized Task 3 training"""
    
    # Initialize specialist agent
    agent = Task3SpecialistAgent()
    
    # Check API
    if not test_api_connection():
        print("❌ API not available. Please check Hugging Face Space.")
        return
    
    print("✅ API connection successful!")
    print("🕵️ Task 3 specialist agent loaded!")
    print(f"🎯 Starting Task 3 focused training for {TRAINING_EPISODES} episodes...\n")
    
    # Training tracking
    episode_rewards = []
    task3_rewards = []
    other_task_rewards = []
    learning_progress = []
    
    start_time = time.time()
    task3_successes = 0
    task3_attempts = 0
    
    for episode in range(1, TRAINING_EPISODES + 1):
        print(f"Episode {episode:3d}/{TRAINING_EPISODES}", end=" | ")
        
        # Force Task 3 for focused training (80% of episodes)
        if episode <= TASK3_FOCUS_EPISODES:
            task_data = force_task3_reset()
            if not task_data:
                # Fallback to regular reset
                task_data = requests.post(f"{API_BASE}/reset", timeout=10).json()
        else:
            # Regular reset for variety (20% of episodes)
            response = requests.post(f"{API_BASE}/reset", timeout=10)
            task_data = response.json() if response.status_code == 200 else None
        
        if not task_data:
            print("❌ Reset failed")
            continue
        
        task_id = task_data.get("task_id", "unknown")
        is_task3 = "task3" in task_id or "adversarial" in task_id
        print(f"Task: {task_id}", end=" | ")
        
        if is_task3:
            task3_attempts += 1
        
        # Specialist agent analysis
        action = agent.analyze_logs(task_data)
        
        # Submit action and get reward
        try:
            response = requests.post(f"{API_BASE}/step", json=action, timeout=10)
            if response.status_code == 200:
                reward_data = response.json().get("reward", {})
                reward = reward_data.get("score", 0.0)
            else:
                print("❌ Step failed")
                continue
        except Exception as e:
            print(f"❌ Step error: {e}")
            continue
        
        print(f"Reward: {reward:.4f}", end=" | ")
        
        # Track Task 3 specific performance
        if is_task3:
            task3_rewards.append(reward)
            if reward > 0.6:  # Success threshold
                task3_successes += 1
                print("🎉 Task 3 SUCCESS!", end=" | ")
        else:
            other_task_rewards.append(reward)
        
        # Self-improvement learning
        agent.learn_from_feedback(task_data, action, reward_data)
        
        # Track overall performance
        episode_rewards.append(reward)
        
        # Performance metrics
        current_perf = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else np.mean(episode_rewards)
        task3_avg = np.mean(task3_rewards) if task3_rewards else 0.0
        print(f"Avg(5): {current_perf:.4f} | Task3 Avg: {task3_avg:.4f}")
        
        # Progress tracking
        if episode % 20 == 0:
            task3_success_rate = (task3_successes / task3_attempts) if task3_attempts > 0 else 0.0
            learning_progress.append({
                "episode": episode,
                "task3_avg": task3_avg,
                "task3_success_rate": task3_success_rate,
                "overall_avg": current_perf
            })
            
            print(f"\n{'='*70}")
            print(f"📊 TASK 3 SPECIALIST PROGRESS - Episode {episode}")
            print(f"{'='*70}")
            print(f"Task 3 Average Reward: {task3_avg:.4f}")
            print(f"Task 3 Success Rate: {task3_success_rate:.1%} ({task3_successes}/{task3_attempts})")
            print(f"Task 3 Episodes: {len(task3_rewards)}")
            print(f"Overall Performance: {current_perf:.4f}")
            
            if task3_avg > 0.5:
                print("🎯 Task 3 showing improvement!")
            if task3_avg > 0.7:
                print("🔥 Task 3 mastery achieved!")
            
            print(f"{'='*70}\n")
            
            # Save progress
            agent.save_progress()
        
        # Brief pause
        time.sleep(0.2)
    
    # Final results
    end_time = time.time()
    duration = end_time - start_time
    
    final_task3_avg = np.mean(task3_rewards) if task3_rewards else 0.0
    final_success_rate = (task3_successes / task3_attempts) if task3_attempts > 0 else 0.0
    
    print(f"\n{'='*70}")
    print("🎉 TASK 3 SPECIALIST TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Total Episodes: {TRAINING_EPISODES}")
    print(f"Task 3 Episodes: {len(task3_rewards)}")
    print(f"Task 3 Average Reward: {final_task3_avg:.4f}")
    print(f"Task 3 Success Rate: {final_success_rate:.1%}")
    print(f"Task 3 Best Reward: {max(task3_rewards) if task3_rewards else 0:.4f}")
    print(f"Overall Average: {np.mean(episode_rewards):.4f}")
    
    # Check improvement
    baseline_task3 = 0.001
    if final_task3_avg > baseline_task3:
        improvement = ((final_task3_avg - baseline_task3) / baseline_task3) * 100
        print(f"🚀 TASK 3 IMPROVEMENT: {improvement:.0f}x better than baseline!")
    
    if final_task3_avg > 0.7:
        print("🏆 TASK 3 MASTERY ACHIEVED!")
    elif final_task3_avg > 0.5:
        print("🎯 TASK 3 SIGNIFICANT IMPROVEMENT!")
    elif final_task3_avg > 0.1:
        print("📈 TASK 3 GOOD PROGRESS!")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "episodes": TRAINING_EPISODES,
        "task3_episodes": len(task3_rewards),
        "task3_rewards": [float(r) for r in task3_rewards],
        "other_task_rewards": [float(r) for r in other_task_rewards],
        "episode_rewards": [float(r) for r in episode_rewards],
        "learning_progress": learning_progress,
        "final_stats": {
            "task3_average": float(final_task3_avg),
            "task3_success_rate": float(final_success_rate),
            "task3_best": float(max(task3_rewards) if task3_rewards else 0),
            "overall_average": float(np.mean(episode_rewards)),
            "improvement_factor": float(final_task3_avg / baseline_task3) if baseline_task3 > 0 else 0
        }
    }
    
    with open("task3_specialist_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    create_task3_plot(task3_rewards, other_task_rewards, learning_progress)
    
    print(f"\n💾 Results saved to task3_specialist_results.json")
    print(f"📊 Plot saved to task3_specialist_plot.png")
    print(f"🧠 Specialist knowledge saved")
    print(f"{'='*70}")

def create_task3_plot(task3_rewards, other_rewards, progress):
    """Create Task 3 specialist training visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Task 3 progress over time
    ax1.plot(task3_rewards, marker='o', linewidth=2, markersize=4, color='red', alpha=0.7)
    ax1.axhline(y=0.7, color='green', linestyle='--', label="Target (0.7)")
    ax1.axhline(y=0.001, color='orange', linestyle='--', label="Baseline (0.001)")
    ax1.set_xlabel("Task 3 Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Task 3 Adversarial Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Task comparison
    ax2.hist([task3_rewards, other_rewards], bins=20, alpha=0.7, 
             label=['Task 3', 'Other Tasks'], color=['red', 'blue'])
    ax2.set_xlabel("Reward")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Task 3 vs Other Tasks Performance")
    ax2.legend()
    
    # Learning progress
    if progress:
        episodes = [p["episode"] for p in progress]
        task3_avgs = [p["task3_avg"] for p in progress]
        success_rates = [p["task3_success_rate"] for p in progress]
        
        ax3.plot(episodes, task3_avgs, marker='s', linewidth=3, color='red', label='Task 3 Avg')
        ax3.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Average Reward")
        ax3.set_title("Task 3 Learning Curve")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Success rate
        ax4.plot(episodes, success_rates, marker='^', linewidth=3, color='purple', label='Success Rate')
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Success Rate")
        ax4.set_title("Task 3 Success Rate Progress")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("task3_specialist_plot.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    try:
        run_task3_specialist_training()
    except KeyboardInterrupt:
        print("\n\n⏹️ Task 3 specialist training interrupted by user")
    except Exception as e:
        print(f"\n❌ Task 3 specialist training failed: {e}")
        import traceback
        traceback.print_exc()