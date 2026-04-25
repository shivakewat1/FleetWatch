"""
Advanced Self-Improvement System for FleetWatch
==============================================
Implements multiple self-improvement mechanisms:
1. Adaptive Learning Rate
2. Dynamic Difficulty Adjustment  
3. Knowledge Base Evolution
4. Performance-Based Strategy Selection
5. Meta-Learning from Mistakes
"""

import json
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import pickle

class SelfImprovementEngine:
    """Advanced self-improvement system for FleetWatch AI"""
    
    def __init__(self):
        # Core learning parameters
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.exploration_rate = 0.2
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.task_expertise = {i: {"attempts": 0, "successes": 0, "avg_reward": 0.0} 
                              for i in range(1, 6)}
        
        # Knowledge evolution
        self.dynamic_keywords = defaultdict(float)  # keyword -> importance score
        self.pattern_library = {}  # pattern_id -> pattern_data
        self.mistake_memory = []  # stores mistakes for learning
        
        # Strategy selection
        self.strategies = {
            "conservative": {"threshold": 0.8, "weight": 1.0},
            "aggressive": {"threshold": 0.5, "weight": 1.0},
            "balanced": {"threshold": 0.65, "weight": 1.0}
        }
        self.current_strategy = "balanced"
        
        # Meta-learning
        self.meta_patterns = {
            "multi_agent_indicators": ["coordinated", "together", "multiple", "collusion"],
            "temporal_patterns": ["pattern", "recurring", "repeated", "consistent"],
            "deception_indicators": ["fake", "fraudulent", "tampered", "disabled"],
            "severity_escalators": ["critical", "emergency", "collision", "injury"]
        }
        
        print("🧠 Self-Improvement Engine initialized")
    
    def analyze_performance_trend(self):
        """Analyze recent performance trends"""
        if len(self.performance_history) < 10:
            return "insufficient_data"
        
        recent = list(self.performance_history)[-10:]
        older = list(self.performance_history)[-20:-10] if len(self.performance_history) >= 20 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg * 1.1:
            return "improving"
        elif recent_avg < older_avg * 0.9:
            return "declining"
        else:
            return "stable"
    
    def adapt_learning_parameters(self, reward, feedback):
        """Dynamically adjust learning parameters based on performance"""
        self.performance_history.append(reward)
        trend = self.analyze_performance_trend()
        
        # Adaptive learning rate
        if trend == "improving":
            self.learning_rate = max(0.05, self.learning_rate * 0.95)  # Slow down when improving
        elif trend == "declining":
            self.learning_rate = min(0.3, self.learning_rate * 1.1)   # Speed up when declining
        
        # Adaptive confidence threshold
        if reward > 0.8:
            self.confidence_threshold = max(0.5, self.confidence_threshold - 0.01)
        elif reward < 0.4:
            self.confidence_threshold = min(0.9, self.confidence_threshold + 0.02)
        
        # Adaptive exploration
        if trend == "stable":
            self.exploration_rate = min(0.4, self.exploration_rate + 0.05)  # Explore more when stable
        else:
            self.exploration_rate = max(0.1, self.exploration_rate - 0.02)  # Explore less when changing
        
        print(f"🔧 Adapted: LR={self.learning_rate:.3f}, Conf={self.confidence_threshold:.3f}, Exp={self.exploration_rate:.3f}")
    
    def evolve_knowledge_base(self, logs, reward, feedback):
        """Evolve knowledge base based on successful/failed predictions"""
        words = logs.lower().split()
        
        # Reward-based keyword importance adjustment
        for word in words:
            if len(word) > 3:  # Skip short words
                if reward > 0.7:
                    self.dynamic_keywords[word] += 0.1  # Increase importance for successful cases
                elif reward < 0.3:
                    self.dynamic_keywords[word] -= 0.05  # Decrease for failed cases
        
        # Extract patterns from high-reward scenarios
        if reward > 0.8:
            pattern_id = f"pattern_{len(self.pattern_library)}"
            self.pattern_library[pattern_id] = {
                "logs_sample": logs[:200],
                "reward": reward,
                "timestamp": datetime.now().isoformat(),
                "keywords": [w for w in words if self.dynamic_keywords.get(w, 0) > 0.5]
            }
        
        # Learn from mistakes
        if reward < 0.4:
            mistake = {
                "logs": logs[:200],
                "reward": reward,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            }
            self.mistake_memory.append(mistake)
            if len(self.mistake_memory) > 50:
                self.mistake_memory.pop(0)  # Keep only recent mistakes
    
    def select_optimal_strategy(self, task_complexity):
        """Select best strategy based on task complexity and recent performance"""
        recent_performance = np.mean(list(self.performance_history)[-5:]) if len(self.performance_history) >= 5 else 0.5
        
        if task_complexity > 0.8:  # Very complex task
            if recent_performance > 0.7:
                self.current_strategy = "aggressive"  # Take risks when performing well
            else:
                self.current_strategy = "conservative"  # Play safe when struggling
        else:  # Simpler task
            self.current_strategy = "balanced"
        
        return self.strategies[self.current_strategy]
    
    def estimate_task_complexity(self, logs, task_description):
        """Estimate task complexity based on various factors"""
        complexity_score = 0.0
        
        # Multi-agent complexity
        agent_count = len(set([word for word in logs.split() if "DRIVER" in word or "MECHANIC" in word or "DISPATCHER" in word]))
        complexity_score += min(0.4, agent_count * 0.1)
        
        # Temporal complexity (multiple time periods)
        time_indicators = ["week", "day", "month", "hour"]
        time_complexity = sum(1 for indicator in time_indicators if indicator in logs.lower())
        complexity_score += min(0.3, time_complexity * 0.1)
        
        # Deception complexity
        deception_words = ["fake", "fraudulent", "tampered", "disabled", "cover", "hide"]
        deception_score = sum(1 for word in deception_words if word in logs.lower())
        complexity_score += min(0.3, deception_score * 0.1)
        
        return min(1.0, complexity_score)
    
    def generate_enhanced_analysis(self, logs, task_description):
        """Generate analysis using self-improved knowledge"""
        complexity = self.estimate_task_complexity(logs, task_description)
        strategy = self.select_optimal_strategy(complexity)
        
        # Use evolved knowledge base
        important_keywords = [word for word, score in self.dynamic_keywords.items() if score > 0.3]
        
        # Pattern matching against successful cases
        pattern_matches = []
        for pattern_id, pattern_data in self.pattern_library.items():
            common_keywords = set(pattern_data["keywords"]) & set(logs.lower().split())
            if len(common_keywords) >= 2:
                pattern_matches.append((pattern_id, len(common_keywords), pattern_data["reward"]))
        
        # Sort by relevance
        pattern_matches.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        # Enhanced anomaly detection
        anomaly_score = 0.0
        detected_issues = []
        
        # Use meta-patterns
        for pattern_type, keywords in self.meta_patterns.items():
            matches = sum(1 for kw in keywords if kw in logs.lower())
            if matches > 0:
                anomaly_score += matches * 0.2
                detected_issues.append(pattern_type)
        
        # Use dynamic keywords
        for word in important_keywords:
            if word in logs.lower():
                anomaly_score += self.dynamic_keywords[word] * 0.1
        
        # Apply strategy threshold
        threshold = strategy["threshold"]
        anomaly_detected = anomaly_score > threshold
        
        # Enhanced agent extraction
        import re
        agent_pattern = r'(DRIVER|MECHANIC|DISPATCHER|FUEL-MANAGER)-(\d+)'
        agent_matches = re.findall(agent_pattern, logs + task_description)
        unique_agents = list(set([f"{match[0]}-{match[1]}" for match in agent_matches]))
        
        # Severity assessment with meta-learning
        severity = "low"
        if any(word in logs.lower() for word in self.meta_patterns["severity_escalators"]):
            severity = "critical"
        elif complexity > 0.7:
            severity = "high"
        elif complexity > 0.4:
            severity = "medium"
        
        # Enhanced summary with learned patterns
        summary_parts = []
        if pattern_matches:
            best_match = pattern_matches[0]
            summary_parts.append(f"Similar to successful pattern {best_match[0]}")
        
        summary_parts.append(f"Complexity: {complexity:.2f}")
        summary_parts.extend(detected_issues[:3])
        
        if len(unique_agents) > 1:
            summary_parts.append(f"Multi-agent scenario ({len(unique_agents)} agents)")
        
        summary = ". ".join(summary_parts)
        
        return {
            "anomaly_detected": anomaly_detected,
            "agent_id": ", ".join(unique_agents[:3]) if unique_agents else "",
            "severity": severity,
            "summary": summary,
            "confidence": min(1.0, anomaly_score),
            "complexity": complexity,
            "strategy_used": self.current_strategy
        }
    
    def learn_from_feedback(self, logs, action, reward_data):
        """Comprehensive learning from feedback"""
        reward = reward_data.get("score", 0.0)
        feedback = reward_data.get("feedback", "")
        
        # Update task expertise
        task_id = getattr(self, '_current_task_id', 1)
        if 1 <= task_id <= 5:
            expertise = self.task_expertise[task_id]
            expertise["attempts"] += 1
            if reward > 0.6:
                expertise["successes"] += 1
            expertise["avg_reward"] = (expertise["avg_reward"] * (expertise["attempts"] - 1) + reward) / expertise["attempts"]
        
        # Adapt parameters
        self.adapt_learning_parameters(reward, feedback)
        
        # Evolve knowledge
        self.evolve_knowledge_base(logs, reward, feedback)
        
        # Meta-learning from mistakes
        if reward < 0.4:
            self._analyze_mistake(logs, action, reward_data)
        
        print(f"📚 Learning update: Task {task_id} expertise: {self.task_expertise[task_id]['avg_reward']:.3f}")
    
    def _analyze_mistake(self, logs, action, reward_data):
        """Analyze mistakes for meta-learning"""
        feedback = reward_data.get("feedback", "").lower()
        
        # Common mistake patterns
        if "missing" in feedback and "agent" in feedback:
            # Missed agent identification
            self.meta_patterns["agent_indicators"] = self.meta_patterns.get("agent_indicators", [])
            # Extract potential agent indicators from logs
            words = logs.lower().split()
            for i, word in enumerate(words):
                if "driver" in word or "mechanic" in word:
                    if i > 0:
                        self.meta_patterns["agent_indicators"].append(words[i-1])
                    if i < len(words) - 1:
                        self.meta_patterns["agent_indicators"].append(words[i+1])
        
        if "severity" in feedback:
            # Severity misjudgment
            actual_severity = action.get("severity", "").lower()
            if "critical" in logs.lower() and actual_severity != "critical":
                self.meta_patterns["severity_escalators"].extend(["critical", "emergency", "failure"])
    
    def get_learning_stats(self):
        """Get comprehensive learning statistics"""
        stats = {
            "performance_trend": self.analyze_performance_trend(),
            "current_parameters": {
                "learning_rate": self.learning_rate,
                "confidence_threshold": self.confidence_threshold,
                "exploration_rate": self.exploration_rate,
                "current_strategy": self.current_strategy
            },
            "knowledge_evolution": {
                "dynamic_keywords_count": len(self.dynamic_keywords),
                "pattern_library_size": len(self.pattern_library),
                "mistakes_learned_from": len(self.mistake_memory)
            },
            "task_expertise": self.task_expertise,
            "recent_performance": np.mean(list(self.performance_history)[-10:]) if len(self.performance_history) >= 10 else 0.0
        }
        return stats
    
    def save_state(self, filename="self_improvement_state.pkl"):
        """Save the current state of the self-improvement system"""
        state = {
            "learning_rate": self.learning_rate,
            "confidence_threshold": self.confidence_threshold,
            "exploration_rate": self.exploration_rate,
            "performance_history": list(self.performance_history),
            "task_expertise": self.task_expertise,
            "dynamic_keywords": dict(self.dynamic_keywords),
            "pattern_library": self.pattern_library,
            "mistake_memory": self.mistake_memory,
            "strategies": self.strategies,
            "current_strategy": self.current_strategy,
            "meta_patterns": self.meta_patterns
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 Self-improvement state saved to {filename}")
    
    def load_state(self, filename="self_improvement_state.pkl"):
        """Load a previously saved state"""
        try:
            with open(filename, 'rb') as f:
                state = pickle.load(f)
            
            self.learning_rate = state["learning_rate"]
            self.confidence_threshold = state["confidence_threshold"]
            self.exploration_rate = state["exploration_rate"]
            self.performance_history = deque(state["performance_history"], maxlen=100)
            self.task_expertise = state["task_expertise"]
            self.dynamic_keywords = defaultdict(float, state["dynamic_keywords"])
            self.pattern_library = state["pattern_library"]
            self.mistake_memory = state["mistake_memory"]
            self.strategies = state["strategies"]
            self.current_strategy = state["current_strategy"]
            self.meta_patterns = state["meta_patterns"]
            
            print(f"📂 Self-improvement state loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"⚠️ State file {filename} not found, starting fresh")
            return False
        except Exception as e:
            print(f"❌ Error loading state: {e}")
            return False

# Integration with main training
class EnhancedFleetWatchAgent:
    """FleetWatch agent with advanced self-improvement"""
    
    def __init__(self):
        self.improvement_engine = SelfImprovementEngine()
        self.improvement_engine.load_state()  # Try to load previous state
        
    def analyze_logs(self, task_data):
        """Enhanced analysis with self-improvement"""
        logs = " ".join(task_data.get("input_logs", []))
        task_desc = task_data.get("task_description", "")
        
        # Store current task for learning
        task_id = task_data.get("task_id", "task1-obvious")
        self.improvement_engine._current_task_id = int(task_id.split('-')[0].replace('task', ''))
        
        # Generate enhanced analysis
        result = self.improvement_engine.generate_enhanced_analysis(logs, task_desc)
        
        return {
            "anomaly_detected": result["anomaly_detected"],
            "agent_id": result["agent_id"],
            "severity": result["severity"],
            "summary": result["summary"]
        }
    
    def learn_from_feedback(self, task_data, action, reward_data):
        """Learn and improve from feedback"""
        logs = " ".join(task_data.get("input_logs", []))
        self.improvement_engine.learn_from_feedback(logs, action, reward_data)
    
    def get_improvement_stats(self):
        """Get self-improvement statistics"""
        return self.improvement_engine.get_learning_stats()
    
    def save_progress(self):
        """Save learning progress"""
        self.improvement_engine.save_state()

if __name__ == "__main__":
    # Test the self-improvement system
    agent = EnhancedFleetWatchAgent()
    
    # Simulate some learning
    test_task = {
        "task_id": "task4-cascade",
        "input_logs": [
            "DRIVER-33 skipped inspection",
            "MECHANIC-05 fake countersignature", 
            "DISPATCHER-07 ignored brake alert",
            "Vehicle collision occurred"
        ]
    }
    
    action = agent.analyze_logs(test_task)
    print("🧪 Test analysis:", action)
    
    # Simulate feedback
    reward_data = {"score": 0.85, "feedback": "Excellent multi-agent detection"}
    agent.learn_from_feedback(test_task, action, reward_data)
    
    # Show stats
    stats = agent.get_improvement_stats()
    print("📊 Learning stats:", json.dumps(stats, indent=2))
    
    agent.save_progress()