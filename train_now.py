"""
FleetWatch Live Training Session
===============================
Real-time training with self-improvement mechanisms
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

print("""
╔══════════════════════════════════════════════════════════╗
║              🚀 FleetWatch Live Training                 ║
║                                                          ║
║  Starting enhanced training with self-improvement...     ║
║  Target: Beat 0.74 baseline → Achieve 0.85+             ║
╚══════════════════════════════════════════════════════════╝
""")

# Configuration
API_BASE = "https://shiva0999-fleet-watch.hf.space"
TRAINING_EPISODES = 50  # Start with 50 for quick results
SAVE_INTERVAL = 10

# Self-improvement tracking
class SelfImprovementTracker:
    def __init__(self):
        self.episode_rewards = []
        self.task_performance = {i: [] for i in range(1, 6)}
        self.learning_phases = []
        self.best_reward = 0.0
        self.improvement_rate = 0.0
        self.adaptive_threshold = 0.5
        
    def update(self, episode, reward, task_id):
        self.episode_rewards.append(reward)
        
        # Track task-specific performance
        task_num = int(task_id.split('-')[0].replace('task', ''))
        if 1 <= task_num <= 5:
            self.task_performance[task_num].append(reward)
        
        # Update best reward
        if reward > self.best_reward:
            self.best_reward = reward
            print(f"🎉 NEW BEST REWARD: {reward:.4f} (Episode {episode})")
        
        # Calculate improvement rate
        if len(self.episode_rewards) >= 10:
            recent_avg = np.mean(self.episode_rewards[-10:])
            old_avg = np.mean(self.episode_rewards[-20:-10]) if len(self.episode_rewards) >= 20 else np.mean(self.episode_rewards[:-10])
            self.improvement_rate = (recent_avg - old_avg) / old_avg if old_avg > 0 else 0
        
        # Adaptive threshold adjustment
        if len(self.episode_rewards) >= 5:
            recent_performance = np.mean(self.episode_rewards[-5:])
            if recent_performance > self.adaptive_threshold:
                self.adaptive_threshold = min(0.9, self.adaptive_threshold + 0.05)
                print(f"📈 Adaptive threshold increased to {self.adaptive_threshold:.3f}")
    
    def get_current_performance(self):
        if not self.episode_rewards:
            return 0.0
        return np.mean(self.episode_rewards[-5:]) if len(self.episode_rewards) >= 5 else np.mean(self.episode_rewards)
    
    def should_increase_difficulty(self):
        return self.get_current_performance() > self.adaptive_threshold
    
    def get_learning_insights(self):
        if len(self.episode_rewards) < 10:
            return "Insufficient data for insights"
        
        insights = []
        
        # Trend analysis
        if self.improvement_rate > 0.1:
            insights.append("🚀 Strong learning trend detected")
        elif self.improvement_rate > 0.05:
            insights.append("📈 Moderate improvement observed")
        else:
            insights.append("📊 Learning plateau - may need adjustment")
        
        # Task-specific analysis
        for task_num, rewards in self.task_performance.items():
            if rewards:
                avg_reward = np.mean(rewards)
                if avg_reward > 0.8:
                    insights.append(f"✅ Task {task_num}: Excellent ({avg_reward:.3f})")
                elif avg_reward > 0.6:
                    insights.append(f"🎯 Task {task_num}: Good ({avg_reward:.3f})")
                else:
                    insights.append(f"⚠️ Task {task_num}: Needs improvement ({avg_reward:.3f})")
        
        return "\n".join(insights)

# API interaction functions
def test_api_connection():
    """Test if API is available"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def reset_environment():
    """Reset environment and get new task"""
    try:
        response = requests.post(f"{API_BASE}/reset", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"⚠️ Reset failed: {e}")
        return None

def submit_action(action_dict):
    """Submit action and get reward"""
    try:
        response = requests.post(f"{API_BASE}/step", json=action_dict, timeout=10)
        if response.status_code == 200:
            result = response.json()
            return result.get("reward", {})
        return None
    except Exception as e:
        print(f"⚠️ Step failed: {e}")
        return None

# Simple AI agent simulation (placeholder for actual model)
class SimpleFleetWatchAgent:
    def __init__(self):
        self.knowledge_base = {
            "gps": ["disabled", "lost", "off", "tampered"],
            "route": ["deviation", "off-route", "detour"],
            "time": ["late", "overtime", "delay"],
            "fuel": ["siphon", "theft", "excess", "inflated"],
            "inspection": ["skip", "fake", "fraudulent"],
            "collusion": ["coordinated", "together", "multiple"]
        }
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        
    def analyze_logs(self, task_data):
        """Analyze logs and detect anomalies"""
        logs = " ".join(task_data.get("input_logs", [])).lower()
        task_desc = task_data.get("task_description", "").lower()
        
        # Anomaly detection logic
        anomaly_score = 0.0
        detected_issues = []
        potential_agents = []
        
        # Check for known fraud patterns
        for category, keywords in self.knowledge_base.items():
            for keyword in keywords:
                if keyword in logs or keyword in task_desc:
                    anomaly_score += 0.2
                    detected_issues.append(keyword)
        
        # Extract agent IDs
        import re
        agent_pattern = r'(DRIVER|MECHANIC|DISPATCHER|FUEL-MANAGER)-\d+'
        agents = re.findall(agent_pattern, logs + task_desc)
        unique_agents = list(set([f"{agent[0]}-{agent[1:]}" for agent in agents]))
        
        # Determine severity
        severity = "low"
        if anomaly_score > 0.8:
            severity = "critical"
        elif anomaly_score > 0.6:
            severity = "high"
        elif anomaly_score > 0.4:
            severity = "medium"
        
        # Generate summary
        summary = f"Detected {len(detected_issues)} anomaly indicators: {', '.join(detected_issues[:3])}"
        if len(unique_agents) > 1:
            summary += f". Multi-agent scenario involving {len(unique_agents)} agents."
        
        return {
            "anomaly_detected": anomaly_score > self.confidence_threshold,
            "agent_id": ", ".join(unique_agents[:3]) if unique_agents else "",
            "severity": severity,
            "summary": summary
        }
    
    def learn_from_feedback(self, reward_data):
        """Self-improvement based on reward feedback"""
        score = reward_data.get("score", 0.0)
        feedback = reward_data.get("feedback", "")
        
        # Adjust confidence threshold based on performance
        if score > 0.8:
            self.confidence_threshold = max(0.5, self.confidence_threshold - 0.01)
        elif score < 0.4:
            self.confidence_threshold = min(0.9, self.confidence_threshold + 0.01)
        
        # Learn from feedback keywords
        if "missing" in feedback.lower() or "incorrect" in feedback.lower():
            self.learning_rate = min(0.2, self.learning_rate + 0.01)
        elif "excellent" in feedback.lower() or "correct" in feedback.lower():
            self.learning_rate = max(0.05, self.learning_rate - 0.005)

def run_training_session():
    """Run the complete training session"""
    
    # Initialize
    tracker = SelfImprovementTracker()
    agent = SimpleFleetWatchAgent()
    
    # Check API
    if not test_api_connection():
        print("❌ API not available. Please check Hugging Face Space.")
        return
    
    print("✅ API connection successful!")
    print(f"🎯 Starting training for {TRAINING_EPISODES} episodes...\n")
    
    start_time = time.time()
    
    for episode in range(1, TRAINING_EPISODES + 1):
        print(f"Episode {episode:3d}/{TRAINING_EPISODES}", end=" | ")
        
        # Reset environment
        task_data = reset_environment()
        if not task_data:
            print("❌ Reset failed")
            continue
        
        task_id = task_data.get("task_id", "unknown")
        print(f"Task: {task_id}", end=" | ")
        
        # Agent analyzes and acts
        action = agent.analyze_logs(task_data)
        
        # Submit action and get reward
        reward_data = submit_action(action)
        if not reward_data:
            print("❌ Step failed")
            continue
        
        reward = reward_data.get("score", 0.0)
        print(f"Reward: {reward:.4f}", end=" | ")
        
        # Self-improvement
        agent.learn_from_feedback(reward_data)
        tracker.update(episode, reward, task_id)
        
        # Performance metrics
        current_perf = tracker.get_current_performance()
        print(f"Avg(5): {current_perf:.4f}")
        
        # Periodic insights
        if episode % SAVE_INTERVAL == 0:
            print(f"\n{'='*60}")
            print(f"📊 PROGRESS REPORT - Episode {episode}")
            print(f"{'='*60}")
            print(f"Current Performance: {current_perf:.4f}")
            print(f"Best Reward: {tracker.best_reward:.4f}")
            print(f"Improvement Rate: {tracker.improvement_rate:.2%}")
            print(f"Agent Confidence: {agent.confidence_threshold:.3f}")
            print(f"\n🧠 Learning Insights:")
            print(tracker.get_learning_insights())
            print(f"{'='*60}\n")
        
        # Brief pause to avoid overwhelming API
        time.sleep(0.5)
    
    # Final results
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Episodes: {TRAINING_EPISODES}")
    print(f"Average Reward: {np.mean(tracker.episode_rewards):.4f}")
    print(f"Best Reward: {tracker.best_reward:.4f}")
    print(f"Final Performance: {tracker.get_current_performance():.4f}")
    
    # Check if we beat baseline
    baseline = 0.74
    final_perf = tracker.get_current_performance()
    if final_perf > baseline:
        improvement = ((final_perf - baseline) / baseline) * 100
        print(f"🚀 SUCCESS! Beat baseline by {improvement:.1f}%")
    else:
        print(f"📈 Progress made, continue training to beat baseline")
    
    print(f"\n🧠 Final Learning Insights:")
    print(tracker.get_learning_insights())
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "episodes": TRAINING_EPISODES,
        "episode_rewards": tracker.episode_rewards,
        "task_performance": tracker.task_performance,
        "best_reward": tracker.best_reward,
        "final_performance": final_perf,
        "beat_baseline": final_perf > baseline
    }
    
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    create_training_plot(tracker)
    
    print(f"\n💾 Results saved to training_results.json")
    print(f"📊 Plot saved to training_progress.png")
    print(f"{'='*60}")

def create_training_plot(tracker):
    """Create training progress visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    ax1.plot(tracker.episode_rewards, alpha=0.6, label="Episode Reward")
    if len(tracker.episode_rewards) >= 5:
        rolling_avg = [np.mean(tracker.episode_rewards[max(0, i-4):i+1]) 
                      for i in range(len(tracker.episode_rewards))]
        ax1.plot(rolling_avg, linewidth=2, label="Rolling Avg (5)")
    ax1.axhline(y=0.74, color='r', linestyle='--', label="Baseline (0.74)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Task performance
    task_names = ["T1: Obvious", "T2: Pattern", "T3: Adversarial", "T4: Cascade", "T5: Collusion"]
    task_avgs = [np.mean(tracker.task_performance[i]) if tracker.task_performance[i] else 0 
                 for i in range(1, 6)]
    colors = ['green', 'blue', 'orange', 'red', 'purple']
    ax2.bar(task_names, task_avgs, color=colors, alpha=0.7)
    ax2.set_ylabel("Average Reward")
    ax2.set_title("Performance by Task")
    ax2.set_ylim(0, 1)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Reward distribution
    ax3.hist(tracker.episode_rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=np.mean(tracker.episode_rewards), color='r', linestyle='--', 
                label=f'Mean: {np.mean(tracker.episode_rewards):.3f}')
    ax3.set_xlabel("Reward")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Reward Distribution")
    ax3.legend()
    
    # Learning curve smoothed
    if len(tracker.episode_rewards) >= 10:
        window = min(10, len(tracker.episode_rewards) // 5)
        smoothed = [np.mean(tracker.episode_rewards[max(0, i-window):i+1]) 
                   for i in range(len(tracker.episode_rewards))]
        ax4.plot(smoothed, linewidth=3, color='green', label=f"Smoothed (window={window})")
        ax4.axhline(y=0.74, color='r', linestyle='--', label="Baseline")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Smoothed Reward")
        ax4.set_title("Learning Curve")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    try:
        run_training_session()
    except KeyboardInterrupt:
        print("\n\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()