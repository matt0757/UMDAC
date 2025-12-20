"""
Feedback Collection and Performance Tracking
=============================================

Implements the feedback loop for continuous learning including
feedback collection, performance monitoring, and retraining triggers.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import uuid

from ..core.models import (
    AnomalyFlag, Feedback, FeedbackType, RuleScore
)
from ..core.knowledge_base import KnowledgeBase


@dataclass
class PerformanceSnapshot:
    """Snapshot of agent/rule performance at a point in time."""
    timestamp: datetime
    agent_id: str
    entity: str
    precision: float
    recall: float
    f1_score: float
    total_flags: int
    true_positives: int
    false_positives: int
    false_negatives: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'entity': self.entity,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'total_flags': self.total_flags,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives
        }


class FeedbackCollector:
    """
    Collects and processes human feedback on detected anomalies.
    
    Features:
    - Simple API for marking flags as TP/FP/FN
    - Batch feedback processing
    - Feedback statistics
    - Integration with knowledge base
    """
    
    def __init__(self, knowledge_base: KnowledgeBase = None):
        """
        Initialize the feedback collector.
        
        Args:
            knowledge_base: Shared knowledge base for persistence
        """
        self.kb = knowledge_base
        
        # In-memory feedback storage (before persistence)
        self.pending_feedback: List[Feedback] = []
        
        # Callbacks for feedback events
        self.on_feedback_received: Optional[Callable[[Feedback], None]] = None
        self.on_high_fp_rate: Optional[Callable[[str, float], None]] = None
    
    def record_feedback(self, flag_id: str, feedback_type: FeedbackType,
                        reviewer: str = "system", comments: str = None,
                        correct_label: str = None) -> Feedback:
        """
        Record feedback for a detected anomaly.
        
        Args:
            flag_id: The anomaly flag ID
            feedback_type: Type of feedback (TP, FP, FN)
            reviewer: Who provided the feedback
            comments: Optional comments
            correct_label: What the label should have been
        
        Returns:
            The created Feedback object
        """
        feedback = Feedback(
            feedback_id=str(uuid.uuid4())[:8],
            flag_id=flag_id,
            feedback_type=feedback_type,
            reviewer=reviewer,
            comments=comments,
            correct_label=correct_label
        )
        
        self.pending_feedback.append(feedback)
        
        # Persist to knowledge base
        if self.kb:
            self.kb.save_feedback(feedback)
        
        # Trigger callback
        if self.on_feedback_received:
            self.on_feedback_received(feedback)
        
        return feedback
    
    def mark_true_positive(self, flag_id: str, reviewer: str = "system",
                           comments: str = None) -> Feedback:
        """Convenience method to mark a flag as true positive."""
        return self.record_feedback(
            flag_id=flag_id,
            feedback_type=FeedbackType.TRUE_POSITIVE,
            reviewer=reviewer,
            comments=comments
        )
    
    def mark_false_positive(self, flag_id: str, reviewer: str = "system",
                            comments: str = None) -> Feedback:
        """Convenience method to mark a flag as false positive."""
        return self.record_feedback(
            flag_id=flag_id,
            feedback_type=FeedbackType.FALSE_POSITIVE,
            reviewer=reviewer,
            comments=comments
        )
    
    def mark_false_negative(self, transaction_id: str, reviewer: str = "system",
                            comments: str = None, agent_id: str = None) -> Feedback:
        """
        Record a missed anomaly (false negative).
        
        This creates a synthetic flag ID since no flag was raised.
        """
        synthetic_flag_id = f"fn_{transaction_id}_{uuid.uuid4().hex[:6]}"
        
        return self.record_feedback(
            flag_id=synthetic_flag_id,
            feedback_type=FeedbackType.FALSE_NEGATIVE,
            reviewer=reviewer,
            comments=comments,
            correct_label=agent_id  # Which agent should have caught it
        )
    
    def process_batch_feedback(self, feedback_items: List[Dict[str, Any]]) -> int:
        """
        Process a batch of feedback items.
        
        Args:
            feedback_items: List of dicts with flag_id, feedback_type, etc.
        
        Returns:
            Number of feedback items processed
        """
        count = 0
        
        for item in feedback_items:
            try:
                self.record_feedback(
                    flag_id=item['flag_id'],
                    feedback_type=FeedbackType(item['feedback_type']),
                    reviewer=item.get('reviewer', 'batch'),
                    comments=item.get('comments')
                )
                count += 1
            except Exception as e:
                print(f"Error processing feedback item: {e}")
        
        return count
    
    def get_feedback_statistics(self, agent_id: str = None,
                                 last_n_days: int = 30) -> Dict[str, Any]:
        """
        Get feedback statistics.
        
        Args:
            agent_id: Filter by agent ID (optional)
            last_n_days: Time window for statistics
        
        Returns:
            Statistics dictionary
        """
        cutoff = datetime.now() - timedelta(days=last_n_days)
        
        # Filter feedback
        relevant = [
            fb for fb in self.pending_feedback
            if fb.created_at >= cutoff
        ]
        
        # If agent_id specified, we'd need to look up the flags
        # For now, just count all feedback
        
        stats = {
            'total_feedback': len(relevant),
            'true_positives': sum(1 for fb in relevant if fb.feedback_type == FeedbackType.TRUE_POSITIVE),
            'false_positives': sum(1 for fb in relevant if fb.feedback_type == FeedbackType.FALSE_POSITIVE),
            'false_negatives': sum(1 for fb in relevant if fb.feedback_type == FeedbackType.FALSE_NEGATIVE),
            'unverified': sum(1 for fb in relevant if fb.feedback_type == FeedbackType.UNVERIFIED),
            'period_days': last_n_days
        }
        
        # Calculate rates
        total_verified = stats['true_positives'] + stats['false_positives']
        stats['precision'] = stats['true_positives'] / total_verified if total_verified > 0 else 0
        
        total_actual = stats['true_positives'] + stats['false_negatives']
        stats['recall'] = stats['true_positives'] / total_actual if total_actual > 0 else 0
        
        if stats['precision'] + stats['recall'] > 0:
            stats['f1'] = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall'])
        else:
            stats['f1'] = 0
        
        return stats
    
    def get_pending_review(self, limit: int = 50) -> List[AnomalyFlag]:
        """
        Get flags that haven't been reviewed yet.
        
        Args:
            limit: Maximum number of flags to return
        
        Returns:
            List of AnomalyFlag objects pending review
        """
        if not self.kb:
            return []
        
        # Get all flags
        # In a real implementation, we'd track review status
        # For now, return empty list
        return []


class PerformanceTracker:
    """
    Tracks agent and rule performance over time.
    
    Features:
    - Time-series performance metrics
    - Degradation detection
    - Automatic retraining triggers
    - Performance comparison across entities
    """
    
    def __init__(self, knowledge_base: KnowledgeBase = None,
                 degradation_threshold: float = 0.15,
                 retraining_f1_threshold: float = 0.6):
        """
        Initialize the performance tracker.
        
        Args:
            knowledge_base: Shared knowledge base
            degradation_threshold: F1 drop that triggers alert
            retraining_f1_threshold: F1 below which retraining is triggered
        """
        self.kb = knowledge_base
        
        self.config = {
            'degradation_threshold': degradation_threshold,
            'retraining_threshold': retraining_f1_threshold,
            'snapshot_frequency_hours': 24,
            'history_retention_days': 90
        }
        
        # In-memory performance history
        self.history: Dict[str, List[PerformanceSnapshot]] = defaultdict(list)
        
        # Callbacks
        self.on_performance_degradation: Optional[Callable[[str, PerformanceSnapshot, PerformanceSnapshot], None]] = None
        self.on_retraining_needed: Optional[Callable[[str, PerformanceSnapshot], None]] = None
    
    def record_snapshot(self, agent_id: str, entity: str,
                        precision: float, recall: float,
                        total_flags: int, tp: int, fp: int, fn: int) -> PerformanceSnapshot:
        """
        Record a performance snapshot.
        
        Args:
            agent_id: Agent identifier
            entity: Entity identifier
            precision: Current precision
            recall: Current recall
            total_flags: Total flags raised
            tp: True positives
            fp: False positives
            fn: False negatives
        
        Returns:
            The created snapshot
        """
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            agent_id=agent_id,
            entity=entity,
            precision=precision,
            recall=recall,
            f1_score=f1,
            total_flags=total_flags,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn
        )
        
        key = f"{agent_id}_{entity}"
        self.history[key].append(snapshot)
        
        # Check for degradation
        self._check_degradation(key, snapshot)
        
        # Check if retraining needed
        if f1 < self.config['retraining_threshold'] and total_flags >= 20:
            if self.on_retraining_needed:
                self.on_retraining_needed(key, snapshot)
        
        # Prune old history
        self._prune_history(key)
        
        return snapshot
    
    def _check_degradation(self, key: str, current: PerformanceSnapshot) -> None:
        """Check for performance degradation."""
        history = self.history.get(key, [])
        
        if len(history) < 2:
            return
        
        # Compare to previous week's average
        week_ago = datetime.now() - timedelta(days=7)
        previous = [s for s in history[:-1] if s.timestamp >= week_ago]
        
        if not previous:
            return
        
        avg_previous_f1 = sum(s.f1_score for s in previous) / len(previous)
        degradation = avg_previous_f1 - current.f1_score
        
        if degradation > self.config['degradation_threshold']:
            if self.on_performance_degradation:
                self.on_performance_degradation(key, previous[-1], current)
    
    def _prune_history(self, key: str) -> None:
        """Remove old history entries."""
        cutoff = datetime.now() - timedelta(days=self.config['history_retention_days'])
        self.history[key] = [
            s for s in self.history[key]
            if s.timestamp >= cutoff
        ]
    
    def get_performance_trend(self, agent_id: str, entity: str,
                               last_n_days: int = 30) -> Dict[str, Any]:
        """
        Get performance trend for an agent/entity combination.
        
        Args:
            agent_id: Agent identifier
            entity: Entity identifier
            last_n_days: Time window
        
        Returns:
            Trend analysis dictionary
        """
        key = f"{agent_id}_{entity}"
        history = self.history.get(key, [])
        
        cutoff = datetime.now() - timedelta(days=last_n_days)
        recent = [s for s in history if s.timestamp >= cutoff]
        
        if len(recent) < 2:
            return {
                'trend': 'insufficient_data',
                'data_points': len(recent)
            }
        
        # Calculate trend
        f1_values = [s.f1_score for s in recent]
        first_half = f1_values[:len(f1_values)//2]
        second_half = f1_values[len(f1_values)//2:]
        
        avg_first = sum(first_half) / len(first_half) if first_half else 0
        avg_second = sum(second_half) / len(second_half) if second_half else 0
        
        change = avg_second - avg_first
        
        if change > 0.05:
            trend = 'improving'
        elif change < -0.05:
            trend = 'degrading'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change': change,
            'current_f1': recent[-1].f1_score if recent else 0,
            'current_precision': recent[-1].precision if recent else 0,
            'current_recall': recent[-1].recall if recent else 0,
            'data_points': len(recent),
            'first_half_avg': avg_first,
            'second_half_avg': avg_second
        }
    
    def get_agent_comparison(self, entity: str = None) -> Dict[str, Dict[str, float]]:
        """
        Compare performance across agents.
        
        Args:
            entity: Optional entity filter
        
        Returns:
            Dictionary mapping agent_id to latest metrics
        """
        comparison = {}
        
        for key, history in self.history.items():
            if not history:
                continue
            
            parts = key.split('_', 1)
            agent_id = parts[0]
            ent = parts[1] if len(parts) > 1 else 'all'
            
            if entity and ent != entity:
                continue
            
            latest = history[-1]
            
            if agent_id not in comparison:
                comparison[agent_id] = {
                    'precision': latest.precision,
                    'recall': latest.recall,
                    'f1_score': latest.f1_score,
                    'total_flags': latest.total_flags,
                    'entities': [ent]
                }
            else:
                # Average across entities
                existing = comparison[agent_id]
                n = len(existing['entities'])
                existing['precision'] = (existing['precision'] * n + latest.precision) / (n + 1)
                existing['recall'] = (existing['recall'] * n + latest.recall) / (n + 1)
                existing['f1_score'] = (existing['f1_score'] * n + latest.f1_score) / (n + 1)
                existing['total_flags'] += latest.total_flags
                existing['entities'].append(ent)
        
        return comparison
    
    def should_retrain(self, agent_id: str, entity: str) -> bool:
        """
        Check if an agent should be retrained.
        
        Args:
            agent_id: Agent identifier
            entity: Entity identifier
        
        Returns:
            True if retraining is recommended
        """
        trend = self.get_performance_trend(agent_id, entity, last_n_days=14)
        
        if trend['trend'] == 'insufficient_data':
            return False
        
        return (
            trend['current_f1'] < self.config['retraining_threshold'] or
            trend['trend'] == 'degrading' and trend['change'] < -self.config['degradation_threshold']
        )
    
    def get_summary_report(self) -> Dict[str, Any]:
        """
        Get a summary report of all tracked performance.
        
        Returns:
            Summary report dictionary
        """
        all_agents = set()
        all_entities = set()
        
        for key in self.history.keys():
            parts = key.split('_', 1)
            all_agents.add(parts[0])
            if len(parts) > 1:
                all_entities.add(parts[1])
        
        report = {
            'total_agents_tracked': len(all_agents),
            'total_entities_tracked': len(all_entities),
            'total_snapshots': sum(len(h) for h in self.history.values()),
            'agents': {}
        }
        
        for agent_id in all_agents:
            agent_keys = [k for k in self.history.keys() if k.startswith(agent_id)]
            
            all_f1 = []
            for key in agent_keys:
                if self.history[key]:
                    all_f1.append(self.history[key][-1].f1_score)
            
            report['agents'][agent_id] = {
                'entities_covered': len(agent_keys),
                'avg_f1': sum(all_f1) / len(all_f1) if all_f1 else 0,
                'min_f1': min(all_f1) if all_f1 else 0,
                'max_f1': max(all_f1) if all_f1 else 0
            }
        
        return report

