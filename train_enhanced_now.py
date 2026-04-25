"""
Enhanced FleetWatch Training with Self-Improvement
=================================================
Using the advanced self-improvement system
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
║         🧠 Enhanced FleetWatch Training                  ║
║                                                          ║
║  🚀 Self-Improvement Engine Active                       ║
║  🎯 Target: 0.85+ Average Reward                         ║
║  🧪 Adaptive Learning & Knowledge Evolution              ║
╚══════════════════════════════════════════════════════════╝
""")

# Configuration
API_BASE = "https://shiva0999-fleet-watch.hf.space"
TRAINING_EPISODES = 75
SAVE_INTERVAL = 15

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

def run_enhanced_training():
    """Run enhanced training with self-improvement"""
    
    # Initialize enhanced agent
    agent = EnhancedFleetWatchAgent()
    
    # Check API
    if not test_api_connection():
        print("❌ API not available. Please check Hugging Face Space.")
        return
    
    print("✅ API connection successful!")
    print("🧠 Self-improvement engine loaded!")
    print(f"🎯 Starting enhanced training for {TRAINING_EPISODES} episodes...\n")
    
    # Training tracking
    episode_rewards = []
    task_performance = {i: [] for i in range(1, 6)}
    learning_milestones = []
    
    start_time = time.time()
    
    for episode in range(1, TRAINING_EPISODES + 1):
        print(f"Episode {episode:3d}/{TRAINING_EPISODES}", end=" | ")
        
        # Reset environment
        task_data = reset_environment()
        if not task_data:
            print("❌ Reset failed")
            continue
        
        task_id = task_data.get("task_id", "unknown")
        task_num = int(task_id.split('-')[0].replace('task', '')) if 'task' in task_id else 1
        print(f"Task: {task_id}", end=" | ")
        
        # Enhanced agent analysis
        action = agent.analyze_logs(task_data)
        
        # Submit action and get reward
        reward_data = submit_action(action)
        if not reward_data:
            print("❌ Step failed")
            continue
        
        reward = reward_data.get("score", 0.0)
        print(f"Reward: {reward:.4f}", end=" | ")
        
        # Self-improvement learning
        agent.learn_from_feedback(task_data, action, reward_data)
        
        # Track performance
        episode_rewards.append(reward)
        if 1 <= task_num <= 5:
            task_performance[task_num].append(reward)
        
        # Performance metrics
        current_perf = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else np.mean(episode_rewards)
        print(f"Avg(5): {current_perf:.4f}")
        
        # Learning milestones
        if reward > 0.8:
            milestone = {
                "episode": episode,
                "reward": reward,
                "task": task_id,
                "action": action
            }
            learning_milestones.append(milestone)
            print(f"🎉 MILESTONE: High reward {reward:.4f} on {task_id}")
        
        # Periodic detailed reports
        if episode % SAVE_INTERVAL == 0:
            print(f"\n{'='*70}")
            print(f"📊 ENHANCED PROGRESS REPORT - Episode {episode}")
            print(f"{'='*70}")
            
            # Basic stats
            print(f"Current Performance: {current_perf:.4f}")
            print(f"Best Reward: {max(episode_rewards):.4f}")
            print(f"Episodes with >0.7 reward: {sum(1 for r in episode_rewards if r > 0.7)}")
            
            # Self-improvement stats
            improvement_stats = agent.get_improvement_stats()
            print(f"\n🧠 Self-Improvement Status:")
            print(f"Performance Trend: {improvement_stats['performance_trend']}")
            print(f"Learning Rate: {improvement_stats['current_parameters']['learning_rate']:.4f}")
            print(f"Confidence Threshold: {improvement_stats['current_parameters']['confidence_threshold']:.3f}")
            print(f"Current Strategy: {improvement_stats['current_parameters']['current_strategy']}")
            
            print(f"\n📚 Knowledge Evolution:")
            print(f"Dynamic Keywords: {improvement_stats['knowledge_evolution']['dynamic_keywords_count']}")
            print(f"Pattern Library: {improvement_stats['knowledge_evolution']['pattern_library_size']}")
            print(f"Mistakes Learned From: {improvement_stats['knowledge_evolution']['mistakes_learned_from']}")
            
            print(f"\n🎯 Task Expertise:")
            for task_num, expertise in improvement_stats['task_expertise'].items():
                if expertise['attempts'] > 0:
                    success_rate = expertise['successes'] / expertise['attempts']
                    print(f"  Task {task_num}: {expertise['avg_reward']:.3f} avg, {success_rate:.1%} success rate")
            
            # Recent milestones
            recent_milestones = [m for m in learning_milestones if m['episode'] > episode - SAVE_INTERVAL]
            if recent_milestones:
                print(f"\n🏆 Recent Milestones:")
                for milestone in recent_milestones[-3:]:
                    print(f"  Episode {milestone['episode']}: {milestone['reward']:.3f} on {milestone['task']}")
            
            print(f"{'='*70}\n")
            
            # Save progress
            agent.save_progress()
        
        # Brief pause
        time.sleep(0.3)
    
    # Final results
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*70}")
    print("🎉 ENHANCED TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Episodes: {TRAINING_EPISODES}")
    print(f"Average Reward: {np.mean(episode_rewards):.4f}")
    print(f"Best Reward: {max(episode_rewards):.4f}")
    print(f"Final Performance: {current_perf:.4f}")
    print(f"High-reward episodes (>0.7): {sum(1 for r in episode_rewards if r > 0.7)}")
    print(f"Milestones achieved: {len(learning_milestones)}")
    
    # Check if we beat baseline
    baseline = 0.74
    if current_perf > baseline:
        improvement = ((current_perf - baseline) / baseline) * 100
        print(f"🚀 SUCCESS! Beat baseline by {improvement:.1f}%")
    elif np.mean(episode_rewards) > baseline:
        improvement = ((np.mean(episode_rewards) - baseline) / baseline) * 100
        print(f"🎯 OVERALL SUCCESS! Average beat baseline by {improvement:.1f}%")
    else:
        print(f"📈 Good progress! Continue training to beat baseline")
    
    # Final self-improvement stats
    final_stats = agent.get_improvement_stats()
    print(f"\n🧠 Final Self-Improvement Status:")
    print(f"Performance Trend: {final_stats['performance_trend']}")
    print(f"Knowledge Base Size: {final_stats['knowledge_evolution']['dynamic_keywords_count']} keywords")
    print(f"Patterns Learned: {final_stats['knowledge_evolution']['pattern_library_size']}")
    print(f"Recent Performance: {final_stats['recent_performance']:.4f}")
    
    # Save comprehensive results
    results = {
        "timestamp": datetime.now().isoformat(),
        "episodes": TRAINING_EPISODES,
        "episode_rewards": [float(r) for r in episode_rewards],  # Convert numpy types
        "task_performance": {k: [float(r) for r in v] for k, v in task_performance.items()},
        "learning_milestones": learning_milestones,
        "final_stats": {
            "average_reward": float(np.mean(episode_rewards)),
            "best_reward": float(max(episode_rewards)),
            "final_performance": float(current_perf),
            "beat_baseline": bool(current_perf > baseline),
            "high_reward_episodes": int(sum(1 for r in episode_rewards if r > 0.7))
        },
        "self_improvement_stats": final_stats
    }
    
    # Save results
    with open("enhanced_training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)  # default=str handles any remaining type issues
    
    # Create visualization
    create_enhanced_plot(episode_rewards, task_performance, learning_milestones)
    
    print(f"\n💾 Results saved to enhanced_training_results.json")
    print(f"📊 Plot saved to enhanced_training_plot.png")
    print(f"🧠 Self-improvement state saved automatically")
    print(f"{'='*70}")

def create_enhanced_plot(episode_rewards, task_performance, milestones):
    """Create enhanced training visualization"""
    fig = plt.figure(figsize=(16, 12))
    
    # Main training curve
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(episode_rewards, alpha=0.6, label="Episode Reward", color='blue')
    if len(episode_rewards) >= 5:
        rolling_avg = [np.mean(episode_rewards[max(0, i-4):i+1]) 
                      for i in range(len(episode_rewards))]
        ax1.plot(rolling_avg, linewidth=2, label="Rolling Avg (5)", color='green')
    ax1.axhline(y=0.74, color='r', linestyle='--', label="Baseline (0.74)")
    
    # Mark milestones
    if milestones:
        milestone_episodes = [m['episode']-1 for m in milestones]  # -1 for 0-indexing
        milestone_rewards = [m['reward'] for m in milestones]
        ax1.scatter(milestone_episodes, milestone_rewards, color='gold', s=100, 
                   marker='*', label=f"Milestones ({len(milestones)})", zorder=5)
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Enhanced Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Task performance
    ax2 = plt.subplot(2, 3, 2)
    task_names = ["T1: Obvious", "T2: Pattern", "T3: Adversarial", "T4: Cascade", "T5: Collusion"]
    task_avgs = [np.mean(task_performance[i]) if task_performance[i] else 0 
                 for i in range(1, 6)]
    colors = ['green', 'blue', 'orange', 'red', 'purple']
    bars = ax2.bar(task_names, task_avgs, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, avg in zip(bars, task_avgs):
        if avg > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{avg:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel("Average Reward")
    ax2.set_title("Performance by Task Type")
    ax2.set_ylim(0, 1)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Reward distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(episode_rewards, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=np.mean(episode_rewards), color='r', linestyle='--', 
                label=f'Mean: {np.mean(episode_rewards):.3f}')
    ax3.axvline(x=0.74, color='orange', linestyle='--', label='Baseline: 0.74')
    ax3.set_xlabel("Reward")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Reward Distribution")
    ax3.legend()
    
    # Learning progression (phases)
    ax4 = plt.subplot(2, 3, 4)
    phase_size = len(episode_rewards) // 5
    if phase_size > 0:
        phases = []
        phase_avgs = []
        for i in range(5):
            start = i * phase_size
            end = (i + 1) * phase_size if i < 4 else len(episode_rewards)
            if start < len(episode_rewards):
                phase_avg = np.mean(episode_rewards[start:end])
                phases.append(f"Phase {i+1}")
                phase_avgs.append(phase_avg)
        
        ax4.plot(phases, phase_avgs, marker='o', linewidth=3, markersize=8, color='green')
        ax4.set_ylabel("Average Reward")
        ax4.set_title("Learning Progression by Phase")
        ax4.grid(True, alpha=0.3)
    
    # Performance improvement over time
    ax5 = plt.subplot(2, 3, 5)
    if len(episode_rewards) >= 10:
        window = 10
        improvement_curve = []
        for i in range(window, len(episode_rewards)):
            recent = np.mean(episode_rewards[i-window:i])
            older = np.mean(episode_rewards[max(0, i-2*window):i-window])
            improvement = (recent - older) / older if older > 0 else 0
            improvement_curve.append(improvement)
        
        ax5.plot(range(window, len(episode_rewards)), improvement_curve, 
                linewidth=2, color='purple', label='Improvement Rate')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Improvement Rate")
        ax5.set_title("Learning Rate Over Time")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Milestone timeline
    ax6 = plt.subplot(2, 3, 6)
    if milestones:
        milestone_episodes = [m['episode'] for m in milestones]
        milestone_rewards = [m['reward'] for m in milestones]
        milestone_tasks = [m['task'].split('-')[0] for m in milestones]
        
        colors_map = {'task1': 'green', 'task2': 'blue', 'task3': 'orange', 
                     'task4': 'red', 'task5': 'purple'}
        colors = [colors_map.get(task, 'gray') for task in milestone_tasks]
        
        ax6.scatter(milestone_episodes, milestone_rewards, c=colors, s=100, alpha=0.8)
        ax6.set_xlabel("Episode")
        ax6.set_ylabel("Milestone Reward")
        ax6.set_title(f"Learning Milestones ({len(milestones)} total)")
        ax6.grid(True, alpha=0.3)
        
        # Add legend for task colors
        for task, color in colors_map.items():
            ax6.scatter([], [], c=color, label=task.upper(), s=50)
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("enhanced_training_plot.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    try:
        run_enhanced_training()
    except KeyboardInterrupt:
        print("\n\n⏹️ Enhanced training interrupted by user")
    except Exception as e:
        print(f"\n❌ Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()