"""
Base Detector Agent
===================

Abstract base class for all detection agents in the multi-agent system.
Defines the standard interface for detection, explanation, and rule updates.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import pandas as pd

from ..core.models import (
    AnomalyFlag, Feedback, DetectionContext, AgentResult,
    AnomalyType, Severity
)
from ..core.rule_graph import RuleGraph
from ..core.knowledge_base import KnowledgeBase


class BaseDetectorAgent(ABC):
    """
    Abstract base class for all detector agents.
    
    Each agent is responsible for a specific type of anomaly detection
    and must implement the core interface methods.
    """
    
    def __init__(self, agent_id: str = None, name: str = "", 
                 knowledge_base: KnowledgeBase = None):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            knowledge_base: Reference to the shared knowledge base
        """
        self.agent_id = agent_id or f"{self.__class__.__name__}_{uuid.uuid4().hex[:6]}"
        self.name = name or self.__class__.__name__
        self.kb = knowledge_base
        
        # Agent state
        self.is_active = True
        self.confidence_threshold = 0.5
        self.last_run_at: Optional[datetime] = None
        self.total_detections = 0
        self.true_positives = 0
        self.false_positives = 0
        
        # Rule graphs managed by this agent
        self.rule_graphs: Dict[str, RuleGraph] = {}
        
        # Configuration
        self.config: Dict[str, Any] = {}
    
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the type identifier for this agent."""
        pass
    
    @property
    @abstractmethod
    def anomaly_types(self) -> List[AnomalyType]:
        """Return the types of anomalies this agent can detect."""
        pass
    
    @abstractmethod
    def detect(self, data: pd.DataFrame, 
               context: DetectionContext = None) -> List[AnomalyFlag]:
        """
        Detect anomalies in the provided data.
        
        Args:
            data: DataFrame containing transaction/feature data
            context: Optional detection context with external signals
        
        Returns:
            List of AnomalyFlag objects for detected anomalies
        """
        pass
    
    @abstractmethod
    def explain(self, flag: AnomalyFlag) -> str:
        """
        Generate a human-readable explanation for a detected anomaly.
        
        Args:
            flag: The anomaly flag to explain
        
        Returns:
            Human-readable explanation string
        """
        pass
    
    def get_confidence(self) -> float:
        """
        Get the overall confidence score for this agent.
        
        Based on historical performance (precision).
        """
        total = self.true_positives + self.false_positives
        if total == 0:
            return self.confidence_threshold
        return self.true_positives / total
    
    def update_rules(self, feedback: Feedback) -> None:
        """
        Update rules based on feedback.
        
        Args:
            feedback: Feedback on a previously detected anomaly
        """
        # Update performance counters
        if feedback.feedback_type.value == "TP":
            self.true_positives += 1
        elif feedback.feedback_type.value == "FP":
            self.false_positives += 1
        
        # Subclasses can override for specific rule updates
        self._process_feedback(feedback)
    
    def _process_feedback(self, feedback: Feedback) -> None:
        """
        Process feedback for rule-specific updates.
        Override in subclasses for custom behavior.
        """
        pass
    
    def export_knowledge(self) -> Dict[str, Any]:
        """
        Export the agent's knowledge for interpretability.
        
        Returns:
            Dictionary containing all rules, configurations, and metrics
        """
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'name': self.name,
            'is_active': self.is_active,
            'confidence': self.get_confidence(),
            'config': self.config,
            'rule_graphs': {k: v.to_dict() for k, v in self.rule_graphs.items()},
            'statistics': {
                'total_detections': self.total_detections,
                'true_positives': self.true_positives,
                'false_positives': self.false_positives,
                'precision': self.get_confidence()
            },
            'last_run_at': self.last_run_at.isoformat() if self.last_run_at else None
        }
    
    def run(self, data: pd.DataFrame, context: DetectionContext = None) -> AgentResult:
        """
        Execute detection and return structured results.
        
        Args:
            data: Input data for detection
            context: Optional detection context
        
        Returns:
            AgentResult with flags and metadata
        """
        start_time = datetime.now()
        
        flags = self.detect(data, context)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        self.total_detections += len(flags)
        self.last_run_at = end_time
        
        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            flags=flags,
            confidence=self.get_confidence(),
            execution_time_ms=execution_time,
            context_used=context
        )
    
    def load_rule_graphs(self) -> None:
        """Load rule graphs from knowledge base."""
        if self.kb:
            graphs = self.kb.get_graphs_by_agent(self.agent_type)
            for graph in graphs:
                self.rule_graphs[graph.graph_id] = graph
    
    def save_rule_graphs(self) -> None:
        """Save rule graphs to knowledge base."""
        if self.kb:
            for graph in self.rule_graphs.values():
                self.kb.save_rule_graph(graph, agent_type=self.agent_type)
    
    def create_flag(self, entity: str, timestamp: datetime,
                    anomaly_type: AnomalyType, severity: Severity,
                    confidence: float, metric_name: str, metric_value: float,
                    threshold: float, description: str, explanation: str,
                    rule_id: str = None, transaction_id: str = None,
                    decision_path: List[str] = None,
                    contributing_factors: Dict[str, Any] = None) -> AnomalyFlag:
        """
        Helper method to create a standardized AnomalyFlag.
        """
        return AnomalyFlag(
            flag_id=f"{self.agent_id}_{uuid.uuid4().hex[:8]}",
            transaction_id=transaction_id,
            entity=entity,
            timestamp=timestamp,
            anomaly_type=anomaly_type,
            severity=severity,
            confidence=confidence,
            agent_id=self.agent_id,
            rule_id=rule_id,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            description=description,
            explanation=explanation,
            decision_path=decision_path or [],
            contributing_factors=contributing_factors or {}
        )
    
    def apply_context_modifier(self, base_threshold: float, 
                                context: DetectionContext) -> float:
        """
        Apply context-based threshold modification.
        
        Args:
            base_threshold: Original threshold value
            context: Detection context with modifier
        
        Returns:
            Adjusted threshold
        """
        if context is None:
            return base_threshold
        
        return base_threshold * context.threshold_modifier
    
    def severity_from_deviation(self, deviation: float, 
                                 thresholds: Dict[str, float] = None) -> Severity:
        """
        Determine severity based on deviation magnitude.
        
        Args:
            deviation: The deviation value (e.g., z-score)
            thresholds: Optional custom thresholds for severity levels
        
        Returns:
            Appropriate Severity level
        """
        thresholds = thresholds or {
            'critical': 5.0,
            'high': 4.0,
            'medium': 3.0,
            'low': 2.0
        }
        
        abs_deviation = abs(deviation)
        
        if abs_deviation >= thresholds['critical']:
            return Severity.CRITICAL
        elif abs_deviation >= thresholds['high']:
            return Severity.HIGH
        elif abs_deviation >= thresholds['medium']:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, confidence={self.get_confidence():.2f})"

