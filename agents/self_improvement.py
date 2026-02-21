"""
Self-Improvement System for MarketMinds Agents

This module implements a feedback loop that tracks agent predictions,
compares them to actual outcomes, and adjusts agent confidence/weights
to improve future predictions.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Store historical data in a JSON file
HISTORY_FILE = Path(__file__).parent.parent / "data" / "agent_history.json"
PERFORMANCE_FILE = Path(__file__).parent.parent / "data" / "agent_performance.json"


@dataclass
class Prediction:
    """Single prediction record."""
    timestamp: str
    ticker: str
    agent_name: str
    sentiment_score: float
    decision: str
    confidence: str
    actual_price_start: Optional[float] = None
    actual_price_end: Optional[float] = None
    actual_outcome: Optional[str] = None  # "correct", "incorrect", "pending"
    price_change_pct: Optional[float] = None


@dataclass
class AgentPerformance:
    """Tracks performance metrics for each agent."""
    agent_name: str
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    avg_confidence_when_correct: float = 0.0
    avg_confidence_when_wrong: float = 0.0
    recent_accuracy: float = 0.0  # Last 10 predictions
    weight_adjustment: float = 1.0  # Multiplier for this agent's influence
    last_updated: str = ""


class SelfImprovementSystem:
    """
    Manages the self-improvement feedback loop for all agents.
    
    Features:
    1. Records all predictions with timestamps
    2. Tracks actual outcomes (price changes)
    3. Calculates agent-specific accuracy metrics
    4. Adjusts agent weights based on performance
    5. Provides insights for improving prompts
    """
    
    def __init__(self):
        self.history: List[Prediction] = []
        self.performance: Dict[str, AgentPerformance] = {}
        self._ensure_data_directory()
        self._load_history()
        self._load_performance()
    
    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_history(self):
        """Load prediction history from disk."""
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    self.history = [Prediction(**item) for item in data]
                logger.info(f"[SELF_IMPROVE] Loaded {len(self.history)} historical predictions")
            except Exception as e:
                logger.error(f"[SELF_IMPROVE] Error loading history: {e}")
                self.history = []
    
    def _load_performance(self):
        """Load agent performance metrics from disk."""
        if PERFORMANCE_FILE.exists():
            try:
                with open(PERFORMANCE_FILE, 'r') as f:
                    data = json.load(f)
                    self.performance = {
                        name: AgentPerformance(**metrics)
                        for name, metrics in data.items()
                    }
                logger.info(f"[SELF_IMPROVE] Loaded performance for {len(self.performance)} agents")
            except Exception as e:
                logger.error(f"[SELF_IMPROVE] Error loading performance: {e}")
                self.performance = {}
    
    def _save_history(self):
        """Save prediction history to disk."""
        try:
            with open(HISTORY_FILE, 'w') as f:
                data = [asdict(pred) for pred in self.history]
                json.dump(data, f, indent=2)
            logger.info(f"[SELF_IMPROVE] Saved {len(self.history)} predictions to history")
        except Exception as e:
            logger.error(f"[SELF_IMPROVE] Error saving history: {e}")
    
    def _save_performance(self):
        """Save agent performance metrics to disk."""
        try:
            with open(PERFORMANCE_FILE, 'w') as f:
                data = {
                    name: asdict(metrics)
                    for name, metrics in self.performance.items()
                }
                json.dump(data, f, indent=2)
            logger.info(f"[SELF_IMPROVE] Saved performance for {len(self.performance)} agents")
        except Exception as e:
            logger.error(f"[SELF_IMPROVE] Error saving performance: {e}")
    
    def record_prediction(
        self,
        ticker: str,
        agent_results: Dict,
        judge_decision: Dict,
        current_price: Optional[float] = None
    ):
        """
        Record a new prediction from the agents.
        
        Args:
            ticker: Stock ticker
            agent_results: Dict with gambler, gossip, video_gossip results
            judge_decision: Judge's final decision
            current_price: Current stock price (if available)
        """
        timestamp = datetime.now().isoformat()
        
        # Record individual agent predictions
        predictions = [
            Prediction(
                timestamp=timestamp,
                ticker=ticker,
                agent_name="gambler",
                sentiment_score=agent_results.get("gambler", {}).get("sentiment_score", 5),
                decision=judge_decision.get("decision", "HOLD"),
                confidence=judge_decision.get("confidence", "medium"),
                actual_price_start=current_price
            ),
            Prediction(
                timestamp=timestamp,
                ticker=ticker,
                agent_name="gossip",
                sentiment_score=agent_results.get("gossip", {}).get("sentiment_score", 5),
                decision=judge_decision.get("decision", "HOLD"),
                confidence=judge_decision.get("confidence", "medium"),
                actual_price_start=current_price
            ),
            Prediction(
                timestamp=timestamp,
                ticker=ticker,
                agent_name="video_gossip",
                sentiment_score=agent_results.get("video_gossip", {}).get("sentiment_score", 5),
                decision=judge_decision.get("decision", "HOLD"),
                confidence=judge_decision.get("confidence", "medium"),
                actual_price_start=current_price
            ),
            Prediction(
                timestamp=timestamp,
                ticker=ticker,
                agent_name="judge",
                sentiment_score=(
                    agent_results.get("gambler", {}).get("sentiment_score", 5) +
                    agent_results.get("gossip", {}).get("sentiment_score", 5) +
                    agent_results.get("video_gossip", {}).get("sentiment_score", 5)
                ) / 3,
                decision=judge_decision.get("decision", "HOLD"),
                confidence=judge_decision.get("confidence", "medium"),
                actual_price_start=current_price
            )
        ]
        
        self.history.extend(predictions)
        self._save_history()
        
        logger.info(f"[SELF_IMPROVE] Recorded prediction for {ticker} at {timestamp}")
        logger.info(f"[SELF_IMPROVE] Decision: {judge_decision.get('decision')} (confidence: {judge_decision.get('confidence')})")
    
    def update_outcomes(self, ticker: str, current_price: float):
        """
        Update outcomes for pending predictions when we have the actual price.
        Checks predictions from the last 7 days.
        
        Args:
            ticker: Stock ticker
            current_price: Current price to compare against
        """
        cutoff_date = datetime.now() - timedelta(days=7)
        updated_count = 0
        
        for pred in self.history:
            if pred.ticker != ticker or pred.actual_outcome is not None:
                continue
            
            pred_time = datetime.fromisoformat(pred.timestamp)
            if pred_time < cutoff_date:
                continue
            
            if pred.actual_price_start is None:
                continue
            
            # Calculate price change
            pred.actual_price_end = current_price
            pred.price_change_pct = ((current_price - pred.actual_price_start) / pred.actual_price_start) * 100
            
            # Determine if prediction was correct
            if pred.decision == "BUY" and pred.price_change_pct > 2:
                pred.actual_outcome = "correct"
            elif pred.decision == "SELL" and pred.price_change_pct < -2:
                pred.actual_outcome = "correct"
            elif pred.decision == "HOLD" and abs(pred.price_change_pct) <= 2:
                pred.actual_outcome = "correct"
            else:
                pred.actual_outcome = "incorrect"
            
            updated_count += 1
        
        if updated_count > 0:
            self._save_history()
            self._recalculate_performance()
            logger.info(f"[SELF_IMPROVE] Updated {updated_count} predictions for {ticker}")
    
    def _recalculate_performance(self):
        """Recalculate performance metrics for all agents."""
        agent_stats = {
            "gambler": {"correct": 0, "total": 0, "recent": [], "conf_correct": [], "conf_wrong": []},
            "gossip": {"correct": 0, "total": 0, "recent": [], "conf_correct": [], "conf_wrong": []},
            "video_gossip": {"correct": 0, "total": 0, "recent": [], "conf_correct": [], "conf_wrong": []},
            "judge": {"correct": 0, "total": 0, "recent": [], "conf_correct": [], "conf_wrong": []}
        }
        
        # Process all predictions with outcomes
        for pred in self.history:
            if pred.actual_outcome is None:
                continue
            
            if pred.agent_name not in agent_stats:
                continue
            
            stats = agent_stats[pred.agent_name]
            stats["total"] += 1
            stats["recent"].append(pred.actual_outcome == "correct")
            
            if pred.actual_outcome == "correct":
                stats["correct"] += 1
                if pred.confidence == "high":
                    stats["conf_correct"].append(1.0)
                elif pred.confidence == "medium":
                    stats["conf_correct"].append(0.5)
                else:
                    stats["conf_correct"].append(0.25)
            else:
                if pred.confidence == "high":
                    stats["conf_wrong"].append(1.0)
                elif pred.confidence == "medium":
                    stats["conf_wrong"].append(0.5)
                else:
                    stats["conf_wrong"].append(0.25)
            
            # Keep only last 10 for recent accuracy
            if len(stats["recent"]) > 10:
                stats["recent"] = stats["recent"][-10:]
        
        # Update performance metrics
        for agent_name, stats in agent_stats.items():
            if stats["total"] == 0:
                continue
            
            accuracy = stats["correct"] / stats["total"]
            recent_accuracy = sum(stats["recent"]) / len(stats["recent"]) if stats["recent"] else accuracy
            
            # Calculate weight adjustment based on accuracy
            # High accuracy (>70%) gets boosted weight
            # Low accuracy (<50%) gets reduced weight
            if accuracy >= 0.7:
                weight_adjustment = 1.0 + (accuracy - 0.7) * 1.5  # Up to 1.45x at 100%
            elif accuracy < 0.5:
                weight_adjustment = 0.5 + accuracy  # Down to 0.5x at 0%
            else:
                weight_adjustment = 1.0
            
            self.performance[agent_name] = AgentPerformance(
                agent_name=agent_name,
                total_predictions=stats["total"],
                correct_predictions=stats["correct"],
                accuracy=accuracy,
                avg_confidence_when_correct=sum(stats["conf_correct"]) / len(stats["conf_correct"]) if stats["conf_correct"] else 0,
                avg_confidence_when_wrong=sum(stats["conf_wrong"]) / len(stats["conf_wrong"]) if stats["conf_wrong"] else 0,
                recent_accuracy=recent_accuracy,
                weight_adjustment=weight_adjustment,
                last_updated=datetime.now().isoformat()
            )
        
        self._save_performance()
        logger.info(f"[SELF_IMPROVE] Recalculated performance for {len(self.performance)} agents")
    
    def get_agent_weights(self) -> Dict[str, float]:
        """
        Get the current weight adjustments for each agent.
        These should be applied when the Judge makes its decision.
        
        Returns:
            Dict mapping agent name to weight multiplier
        """
        weights = {
            "gambler": 1.0,
            "gossip": 1.0,
            "video_gossip": 1.0
        }
        
        for agent_name, perf in self.performance.items():
            if agent_name in weights:
                weights[agent_name] = perf.weight_adjustment
        
        logger.info(f"[SELF_IMPROVE] Current agent weights: {weights}")
        return weights
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of all agent performance metrics."""
        return {
            agent_name: {
                "accuracy": f"{perf.accuracy * 100:.1f}%",
                "recent_accuracy": f"{perf.recent_accuracy * 100:.1f}%",
                "total_predictions": perf.total_predictions,
                "correct_predictions": perf.correct_predictions,
                "weight_adjustment": f"{perf.weight_adjustment:.2f}x",
                "last_updated": perf.last_updated
            }
            for agent_name, perf in self.performance.items()
        }
    
    def get_insights(self) -> List[str]:
        """
        Generate insights and recommendations for improving the system.
        
        Returns:
            List of actionable insights
        """
        insights = []
        
        for agent_name, perf in self.performance.items():
            if perf.total_predictions < 5:
                continue
            
            # Low accuracy warning
            if perf.accuracy < 0.5:
                insights.append(
                    f"⚠️ {agent_name.upper()}: Low accuracy ({perf.accuracy*100:.1f}%). "
                    f"Consider adjusting prompts or data sources."
                )
            
            # Recent decline
            if perf.recent_accuracy < perf.accuracy - 0.15:
                insights.append(
                    f"📉 {agent_name.upper()}: Recent performance declining "
                    f"({perf.recent_accuracy*100:.1f}% vs {perf.accuracy*100:.1f}% overall). "
                    f"Market conditions may have changed."
                )
            
            # Overconfident when wrong
            if perf.avg_confidence_when_wrong > perf.avg_confidence_when_correct:
                insights.append(
                    f"⚡ {agent_name.upper()}: More confident when wrong than when right. "
                    f"Calibration needed."
                )
            
            # High performer
            if perf.accuracy >= 0.7 and perf.total_predictions >= 10:
                insights.append(
                    f"✅ {agent_name.upper()}: Consistently high accuracy ({perf.accuracy*100:.1f}%). "
                    f"Good performance!"
                )
        
        if not insights:
            insights.append("ℹ️ Not enough data yet to generate insights. Keep making predictions!")
        
        return insights


# Global instance
_system = None


def get_improvement_system() -> SelfImprovementSystem:
    """Get the global self-improvement system instance."""
    global _system
    if _system is None:
        _system = SelfImprovementSystem()
    return _system
