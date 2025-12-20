"""
Data models for the multi-agent anomaly detection system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class Severity(Enum):
    """Anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    RULE_VIOLATION = "rule_violation"
    TEMPORAL = "temporal"
    CATEGORY = "category"
    EXTERNAL = "external"
    ENSEMBLE = "ensemble"


class FeedbackType(Enum):
    """Types of feedback for detected anomalies."""
    TRUE_POSITIVE = "TP"
    FALSE_POSITIVE = "FP"
    FALSE_NEGATIVE = "FN"
    UNVERIFIED = "unverified"


@dataclass
class AnomalyFlag:
    """Represents a detected anomaly."""
    flag_id: str
    transaction_id: Optional[str]
    entity: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: Severity
    confidence: float
    
    # Detection details
    agent_id: str
    rule_id: Optional[str]
    metric_name: str
    metric_value: float
    threshold: float
    
    # Context
    description: str
    explanation: str
    decision_path: List[str] = field(default_factory=list)
    contributing_factors: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    feedback: Optional['Feedback'] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'flag_id': self.flag_id,
            'transaction_id': self.transaction_id,
            'entity': self.entity,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity.value,
            'confidence': self.confidence,
            'agent_id': self.agent_id,
            'rule_id': self.rule_id,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'description': self.description,
            'explanation': self.explanation,
            'decision_path': self.decision_path,
            'contributing_factors': self.contributing_factors,
            'created_at': self.created_at.isoformat(),
            'feedback': self.feedback.to_dict() if self.feedback else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnomalyFlag':
        """Create from dictionary."""
        feedback = None
        if data.get('feedback'):
            feedback = Feedback.from_dict(data['feedback'])
        
        return cls(
            flag_id=data['flag_id'],
            transaction_id=data.get('transaction_id'),
            entity=data['entity'],
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None,
            anomaly_type=AnomalyType(data['anomaly_type']),
            severity=Severity(data['severity']),
            confidence=data['confidence'],
            agent_id=data['agent_id'],
            rule_id=data.get('rule_id'),
            metric_name=data['metric_name'],
            metric_value=data['metric_value'],
            threshold=data['threshold'],
            description=data['description'],
            explanation=data['explanation'],
            decision_path=data.get('decision_path', []),
            contributing_factors=data.get('contributing_factors', {}),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            feedback=feedback
        )


@dataclass
class Feedback:
    """Human feedback on a detected anomaly."""
    feedback_id: str
    flag_id: str
    feedback_type: FeedbackType
    reviewer: str
    comments: Optional[str]
    correct_label: Optional[str]  # What it should have been
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feedback_id': self.feedback_id,
            'flag_id': self.flag_id,
            'feedback_type': self.feedback_type.value,
            'reviewer': self.reviewer,
            'comments': self.comments,
            'correct_label': self.correct_label,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Feedback':
        return cls(
            feedback_id=data['feedback_id'],
            flag_id=data['flag_id'],
            feedback_type=FeedbackType(data['feedback_type']),
            reviewer=data['reviewer'],
            comments=data.get('comments'),
            correct_label=data.get('correct_label'),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now()
        )


@dataclass
class RuleScore:
    """Performance metrics for a specific rule."""
    rule_id: str
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_activations: int
    evaluation_period_days: int
    evaluated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'total_activations': self.total_activations,
            'evaluation_period_days': self.evaluation_period_days,
            'evaluated_at': self.evaluated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuleScore':
        return cls(
            rule_id=data['rule_id'],
            precision=data['precision'],
            recall=data['recall'],
            f1_score=data['f1_score'],
            true_positives=data['true_positives'],
            false_positives=data['false_positives'],
            false_negatives=data['false_negatives'],
            total_activations=data['total_activations'],
            evaluation_period_days=data['evaluation_period_days'],
            evaluated_at=datetime.fromisoformat(data['evaluated_at']) if data.get('evaluated_at') else datetime.now()
        )


class MutationType(Enum):
    """Types of rule mutations."""
    THRESHOLD_INCREASE = "threshold_increase"
    THRESHOLD_DECREASE = "threshold_decrease"
    ADD_CONDITION = "add_condition"
    REMOVE_CONDITION = "remove_condition"
    MODIFY_CONDITION = "modify_condition"
    CHANGE_EDGE = "change_edge"


@dataclass
class Mutation:
    """Represents a proposed or applied rule mutation."""
    mutation_id: str
    rule_id: str
    mutation_type: MutationType
    original_value: Any
    new_value: Any
    justification: str
    confidence: float
    applied: bool = False
    applied_at: Optional[datetime] = None
    rollback_available: bool = True
    performance_before: Optional[RuleScore] = None
    performance_after: Optional[RuleScore] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mutation_id': self.mutation_id,
            'rule_id': self.rule_id,
            'mutation_type': self.mutation_type.value,
            'original_value': self.original_value,
            'new_value': self.new_value,
            'justification': self.justification,
            'confidence': self.confidence,
            'applied': self.applied,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
            'rollback_available': self.rollback_available,
            'performance_before': self.performance_before.to_dict() if self.performance_before else None,
            'performance_after': self.performance_after.to_dict() if self.performance_after else None,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Mutation':
        return cls(
            mutation_id=data['mutation_id'],
            rule_id=data['rule_id'],
            mutation_type=MutationType(data['mutation_type']),
            original_value=data['original_value'],
            new_value=data['new_value'],
            justification=data['justification'],
            confidence=data['confidence'],
            applied=data.get('applied', False),
            applied_at=datetime.fromisoformat(data['applied_at']) if data.get('applied_at') else None,
            rollback_available=data.get('rollback_available', True),
            performance_before=RuleScore.from_dict(data['performance_before']) if data.get('performance_before') else None,
            performance_after=RuleScore.from_dict(data['performance_after']) if data.get('performance_after') else None,
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now()
        )


@dataclass
class DetectionContext:
    """Context information for anomaly detection."""
    entity: str
    timestamp: datetime
    
    # External context signals
    market_stress: str = "NORMAL"  # LOW, NORMAL, HIGH, CRITICAL
    sentiment_score: float = 0.0  # -1 to 1
    threshold_modifier: float = 1.0  # Multiplier for thresholds
    
    # Recent events
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Entity-specific context
    entity_volatility: float = 1.0
    entity_trend: str = "STABLE"  # UP, DOWN, STABLE
    
    # Flags
    is_month_end: bool = False
    is_quarter_end: bool = False
    is_holiday_period: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity': self.entity,
            'timestamp': self.timestamp.isoformat(),
            'market_stress': self.market_stress,
            'sentiment_score': self.sentiment_score,
            'threshold_modifier': self.threshold_modifier,
            'recent_events': self.recent_events,
            'entity_volatility': self.entity_volatility,
            'entity_trend': self.entity_trend,
            'is_month_end': self.is_month_end,
            'is_quarter_end': self.is_quarter_end,
            'is_holiday_period': self.is_holiday_period
        }


@dataclass
class AgentResult:
    """Result from a single agent's detection."""
    agent_id: str
    agent_type: str
    flags: List[AnomalyFlag]
    confidence: float
    execution_time_ms: float
    context_used: Optional[DetectionContext] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'flags': [f.to_dict() for f in self.flags],
            'confidence': self.confidence,
            'execution_time_ms': self.execution_time_ms,
            'context_used': self.context_used.to_dict() if self.context_used else None
        }


@dataclass
class EnsembleVerdict:
    """Final verdict after combining all agent results."""
    verdict_id: str
    entity: str
    timestamp: datetime
    
    # Final decision
    is_anomaly: bool
    final_confidence: float
    severity: Severity
    
    # Agent contributions
    agent_results: List[AgentResult]
    agreeing_agents: List[str]
    disagreeing_agents: List[str]
    
    # Aggregated flags
    primary_flags: List[AnomalyFlag]  # Most significant
    secondary_flags: List[AnomalyFlag]  # Supporting evidence
    
    # Explanation
    explanation: str
    decision_factors: List[str]
    recommended_action: str
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'verdict_id': self.verdict_id,
            'entity': self.entity,
            'timestamp': self.timestamp.isoformat(),
            'is_anomaly': self.is_anomaly,
            'final_confidence': self.final_confidence,
            'severity': self.severity.value,
            'agent_results': [r.to_dict() for r in self.agent_results],
            'agreeing_agents': self.agreeing_agents,
            'disagreeing_agents': self.disagreeing_agents,
            'primary_flags': [f.to_dict() for f in self.primary_flags],
            'secondary_flags': [f.to_dict() for f in self.secondary_flags],
            'explanation': self.explanation,
            'decision_factors': self.decision_factors,
            'recommended_action': self.recommended_action,
            'created_at': self.created_at.isoformat()
        }

