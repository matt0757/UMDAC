"""
Statistical Anomaly Detection Agent
====================================

Detects anomalies using statistical methods including Z-scores,
rolling statistics, and level shifts.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
import uuid

from .base_agent import BaseDetectorAgent
from ..core.models import (
    AnomalyFlag, Feedback, DetectionContext, AnomalyType, Severity
)
from ..core.rule_graph import RuleGraph, RuleNode, NodeType, EdgeType, ThresholdConfig
from ..core.knowledge_base import KnowledgeBase


class StatisticalAgent(BaseDetectorAgent):
    """
    Agent for statistical anomaly detection.
    
    Methods:
    - Z-score analysis
    - Rolling statistics deviation
    - Level shift detection
    - IQR-based outlier detection
    """
    
    def __init__(self, agent_id: str = None, name: str = "Statistical Agent",
                 knowledge_base: KnowledgeBase = None,
                 zscore_threshold: float = 3.0,
                 rolling_window: int = 4,
                 level_shift_threshold: float = 2.5):
        """
        Initialize the statistical agent.
        
        Args:
            agent_id: Unique identifier
            name: Agent name
            knowledge_base: Shared knowledge base
            zscore_threshold: Base Z-score threshold
            rolling_window: Window size for rolling statistics
            level_shift_threshold: Threshold for level shift detection
        """
        super().__init__(agent_id, name, knowledge_base)
        
        # Configuration
        self.config = {
            'zscore_threshold': zscore_threshold,
            'rolling_window': rolling_window,
            'level_shift_threshold': level_shift_threshold,
            'min_samples_for_stats': 4,
            'iqr_multiplier': 1.5
        }
        
        # Initialize rule graphs
        self._init_rule_graphs()
    
    @property
    def agent_type(self) -> str:
        return "statistical"
    
    @property
    def anomaly_types(self) -> List[AnomalyType]:
        return [AnomalyType.STATISTICAL]
    
    def _init_rule_graphs(self) -> None:
        """Initialize the statistical rule graphs."""
        # Z-score rule graph
        zscore_graph = self._create_zscore_graph()
        self.rule_graphs['zscore'] = zscore_graph
        
        # Level shift graph
        level_shift_graph = self._create_level_shift_graph()
        self.rule_graphs['level_shift'] = level_shift_graph
    
    def _create_zscore_graph(self) -> RuleGraph:
        """Create the Z-score detection rule graph."""
        graph = RuleGraph(
            graph_id="stat_zscore",
            name="Z-Score Anomaly Detection",
            description="Detects statistical outliers based on Z-score"
        )
        
        # Root: Check if Z-score exceeds threshold
        zscore_check = RuleNode(
            node_id="zscore_check",
            node_type=NodeType.THRESHOLD,
            name="Z-Score Check",
            description="Check if Z-score indicates anomaly",
            condition="abs(zscore) > threshold",
            condition_code="abs(data.get('zscore', 0)) > threshold",
            field_name="zscore",
            operator="abs>",
            threshold=ThresholdConfig(
                base=self.config['zscore_threshold'],
                min_value=1.5,
                max_value=5.0
            )
        )
        graph.add_node(zscore_check, is_root=True)
        
        # Critical threshold check
        critical_check = RuleNode(
            node_id="critical_check",
            node_type=NodeType.THRESHOLD,
            name="Critical Threshold",
            description="Check for critical anomaly (very high Z-score)",
            condition="abs(zscore) > critical_threshold",
            condition_code="abs(data.get('zscore', 0)) > threshold",
            field_name="zscore",
            operator="abs>",
            threshold=ThresholdConfig(base=self.config['zscore_threshold'] * 1.67)
        )
        graph.add_node(critical_check)
        
        # Flag critical
        flag_critical = RuleNode(
            node_id="flag_critical",
            node_type=NodeType.ACTION,
            name="Flag Critical",
            description="Critical statistical anomaly",
            condition="Critical anomaly detected",
            action="flag_anomaly",
            severity="critical"
        )
        graph.add_node(flag_critical)
        
        # High threshold check
        high_check = RuleNode(
            node_id="high_check",
            node_type=NodeType.THRESHOLD,
            name="High Threshold",
            description="Check for high-severity anomaly",
            condition="abs(zscore) > high_threshold",
            condition_code="abs(data.get('zscore', 0)) > threshold",
            field_name="zscore",
            operator="abs>",
            threshold=ThresholdConfig(base=self.config['zscore_threshold'] * 1.33)
        )
        graph.add_node(high_check)
        
        # Flag high
        flag_high = RuleNode(
            node_id="flag_high",
            node_type=NodeType.ACTION,
            name="Flag High",
            description="High-severity statistical anomaly",
            condition="High anomaly detected",
            action="flag_anomaly",
            severity="high"
        )
        graph.add_node(flag_high)
        
        # Flag medium
        flag_medium = RuleNode(
            node_id="flag_medium",
            node_type=NodeType.ACTION,
            name="Flag Medium",
            description="Medium-severity statistical anomaly",
            condition="Medium anomaly detected",
            action="flag_anomaly",
            severity="medium"
        )
        graph.add_node(flag_medium)
        
        # Normal
        normal = RuleNode(
            node_id="normal",
            node_type=NodeType.ACTION,
            name="Normal",
            description="No statistical anomaly",
            condition="Within normal range",
            action="normal",
            severity="low"
        )
        graph.add_node(normal)
        
        # Add edges
        graph.add_edge("zscore_check", "critical_check", EdgeType.CHAIN)
        graph.add_edge("critical_check", "flag_critical", EdgeType.ESCALATE)
        graph.add_edge("critical_check", "high_check", EdgeType.ELSE)
        graph.add_edge("high_check", "flag_high", EdgeType.ESCALATE)
        graph.add_edge("high_check", "flag_medium", EdgeType.ELSE)
        graph.add_edge("zscore_check", "normal", EdgeType.ELSE)
        
        return graph
    
    def _create_level_shift_graph(self) -> RuleGraph:
        """Create the level shift detection rule graph."""
        graph = RuleGraph(
            graph_id="stat_level_shift",
            name="Level Shift Detection",
            description="Detects sudden level changes in time series"
        )
        
        # Root: Check rolling mean deviation
        level_check = RuleNode(
            node_id="level_shift_check",
            node_type=NodeType.THRESHOLD,
            name="Level Shift Check",
            description="Check for significant level shift",
            condition="rolling_deviation > threshold",
            condition_code="abs(data.get('level_shift_score', 0)) > threshold",
            field_name="level_shift_score",
            operator="abs>",
            threshold=ThresholdConfig(
                base=self.config['level_shift_threshold'],
                min_value=1.5,
                max_value=4.0
            )
        )
        graph.add_node(level_check, is_root=True)
        
        # Flag level shift
        flag_shift = RuleNode(
            node_id="flag_level_shift",
            node_type=NodeType.ACTION,
            name="Flag Level Shift",
            description="Significant level shift detected",
            condition="Level shift detected",
            action="flag_anomaly",
            severity="medium"
        )
        graph.add_node(flag_shift)
        
        # Normal
        normal = RuleNode(
            node_id="normal",
            node_type=NodeType.ACTION,
            name="Normal",
            description="No level shift",
            condition="Stable level",
            action="normal",
            severity="low"
        )
        graph.add_node(normal)
        
        graph.add_edge("level_shift_check", "flag_level_shift", EdgeType.ESCALATE)
        graph.add_edge("level_shift_check", "normal", EdgeType.ELSE)
        
        return graph
    
    def detect(self, data: pd.DataFrame, 
               context: DetectionContext = None) -> List[AnomalyFlag]:
        """
        Detect statistical anomalies in the data.
        
        Expects columns: Entity, Total_Net, Net_Rolling4_Mean, Net_Rolling4_Std,
        Week_Start or timestamp column.
        """
        flags = []
        
        # Get context modifier
        modifier = context.threshold_modifier if context else 1.0
        
        # Process each entity if multiple entities
        if 'Entity' in data.columns:
            entities = data['Entity'].unique()
        else:
            entities = [context.entity if context else 'unknown']
            data = data.copy()
            data['Entity'] = entities[0]
        
        for entity in entities:
            entity_data = data[data['Entity'] == entity] if 'Entity' in data.columns else data
            
            # Z-score detection
            zscore_flags = self._detect_zscore_anomalies(entity_data, entity, modifier)
            flags.extend(zscore_flags)
            
            # Level shift detection
            level_flags = self._detect_level_shifts(entity_data, entity, modifier)
            flags.extend(level_flags)
            
            # IQR-based detection
            iqr_flags = self._detect_iqr_outliers(entity_data, entity, modifier)
            flags.extend(iqr_flags)
        
        return flags
    
    def _detect_zscore_anomalies(self, data: pd.DataFrame, entity: str,
                                   modifier: float) -> List[AnomalyFlag]:
        """Detect Z-score based anomalies."""
        flags = []
        zscore_graph = self.rule_graphs.get('zscore')
        
        if len(data) < self.config['min_samples_for_stats']:
            return flags
        
        # Calculate Z-scores for key metrics
        metrics_to_check = ['Total_Net', 'Total_Inflow', 'Total_Outflow']
        
        for metric in metrics_to_check:
            if metric not in data.columns:
                continue
            
            values = data[metric].dropna()
            if len(values) < 2:
                continue
            
            mean = values.mean()
            std = values.std()
            
            if std == 0 or pd.isna(std):
                continue
            
            # Check each row
            for idx, row in data.iterrows():
                value = row.get(metric)
                if pd.isna(value):
                    continue
                
                zscore = (value - mean) / std
                threshold = self.config['zscore_threshold'] * modifier
                
                if abs(zscore) > threshold:
                    # Evaluate through rule graph
                    eval_data = {'zscore': zscore, 'value': value, 'mean': mean, 'std': std}
                    results = zscore_graph.evaluate(eval_data, entity, modifier)
                    
                    for result in results:
                        if result['action'] == 'flag_anomaly':
                            severity = Severity(result['severity'])
                            
                            # Get timestamp
                            timestamp = row.get('Week_Start') or row.get('timestamp')
                            if isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(timestamp)
                            elif not isinstance(timestamp, datetime):
                                timestamp = datetime.now()
                            
                            flag = self.create_flag(
                                entity=entity,
                                timestamp=timestamp,
                                anomaly_type=AnomalyType.STATISTICAL,
                                severity=severity,
                                confidence=result.get('confidence', 0.8),
                                metric_name=f"{metric}_zscore",
                                metric_value=zscore,
                                threshold=threshold,
                                description=f"Z-score anomaly in {metric}: {zscore:.2f}σ from mean",
                                explanation=f"The value {value:,.2f} is {abs(zscore):.2f} standard deviations from the mean ({mean:,.2f}). "
                                           f"This is considered statistically unusual.",
                                rule_id="stat_zscore",
                                decision_path=result.get('decision_path', []),
                                contributing_factors={
                                    'value': value,
                                    'mean': mean,
                                    'std': std,
                                    'zscore': zscore
                                }
                            )
                            flags.append(flag)
        
        return flags
    
    def _detect_level_shifts(self, data: pd.DataFrame, entity: str,
                              modifier: float) -> List[AnomalyFlag]:
        """Detect level shifts in the data."""
        flags = []
        
        if 'Total_Net' not in data.columns or len(data) < self.config['rolling_window'] * 2:
            return flags
        
        # Calculate level shift score
        values = data['Total_Net'].values
        window = self.config['rolling_window']
        
        for i in range(window, len(values) - 1):
            # Compare current window to previous window
            prev_window = values[max(0, i-window):i]
            curr_window = values[i:min(len(values), i+window)]
            
            if len(prev_window) < 2 or len(curr_window) < 2:
                continue
            
            prev_mean = np.mean(prev_window)
            curr_mean = np.mean(curr_window)
            pooled_std = np.std(np.concatenate([prev_window, curr_window]))
            
            if pooled_std == 0:
                continue
            
            level_shift_score = abs(curr_mean - prev_mean) / pooled_std
            threshold = self.config['level_shift_threshold'] * modifier
            
            if level_shift_score > threshold:
                row = data.iloc[i]
                timestamp = row.get('Week_Start') or row.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                flag = self.create_flag(
                    entity=entity,
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=Severity.MEDIUM if level_shift_score < 4 else Severity.HIGH,
                    confidence=min(0.95, 0.6 + level_shift_score * 0.1),
                    metric_name="level_shift_score",
                    metric_value=level_shift_score,
                    threshold=threshold,
                    description=f"Level shift detected: {level_shift_score:.2f}",
                    explanation=f"A significant level shift was detected. The average changed from "
                               f"{prev_mean:,.2f} to {curr_mean:,.2f}, a shift of {level_shift_score:.2f} "
                               f"standard deviations.",
                    rule_id="stat_level_shift",
                    contributing_factors={
                        'prev_mean': prev_mean,
                        'curr_mean': curr_mean,
                        'shift_magnitude': abs(curr_mean - prev_mean)
                    }
                )
                flags.append(flag)
        
        return flags
    
    def _detect_iqr_outliers(self, data: pd.DataFrame, entity: str,
                              modifier: float) -> List[AnomalyFlag]:
        """Detect outliers using IQR method."""
        flags = []
        
        if 'Total_Net' not in data.columns or len(data) < 4:
            return flags
        
        values = data['Total_Net'].dropna()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.config['iqr_multiplier'] * iqr * modifier
        upper_bound = q3 + self.config['iqr_multiplier'] * iqr * modifier
        
        for idx, row in data.iterrows():
            value = row.get('Total_Net')
            if pd.isna(value):
                continue
            
            if value < lower_bound or value > upper_bound:
                timestamp = row.get('Week_Start') or row.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                distance = min(abs(value - lower_bound), abs(value - upper_bound))
                deviation_iqr = distance / iqr if iqr > 0 else 0
                
                flag = self.create_flag(
                    entity=entity,
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=Severity.LOW if deviation_iqr < 1 else Severity.MEDIUM,
                    confidence=0.7,
                    metric_name="iqr_outlier",
                    metric_value=value,
                    threshold=upper_bound if value > upper_bound else lower_bound,
                    description=f"IQR outlier: {value:,.2f}",
                    explanation=f"The value {value:,.2f} falls outside the interquartile range. "
                               f"Normal range: [{lower_bound:,.2f}, {upper_bound:,.2f}]",
                    rule_id="stat_iqr",
                    contributing_factors={
                        'q1': q1,
                        'q3': q3,
                        'iqr': iqr,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                )
                flags.append(flag)
        
        return flags
    
    def explain(self, flag: AnomalyFlag) -> str:
        """Generate detailed explanation for a statistical anomaly."""
        lines = [
            f"Statistical Anomaly Detected",
            f"============================",
            f"Entity: {flag.entity}",
            f"Time: {flag.timestamp}",
            f"Severity: {flag.severity.value.upper()}",
            f"",
            f"Metric: {flag.metric_name}",
            f"Value: {flag.metric_value:.4f}",
            f"Threshold: {flag.threshold:.4f}",
            f"",
            f"Description: {flag.description}",
            f"",
            f"Explanation: {flag.explanation}",
            f"",
            "Decision Path:"
        ]
        
        for step in flag.decision_path:
            lines.append(f"  -> {step}")
        
        if flag.contributing_factors:
            lines.append("")
            lines.append("Contributing Factors:")
            for key, value in flag.contributing_factors.items():
                if isinstance(value, float):
                    lines.append(f"  - {key}: {value:.4f}")
                else:
                    lines.append(f"  - {key}: {value}")
        
        return "\n".join(lines)
    
    def _process_feedback(self, feedback: Feedback) -> None:
        """Process feedback to adjust thresholds."""
        # If high false positive rate, increase threshold
        if self.false_positives > 10 and self.get_confidence() < 0.7:
            current = self.config['zscore_threshold']
            new_threshold = min(current * 1.1, 5.0)
            self.config['zscore_threshold'] = new_threshold
            
            # Update rule graph
            zscore_graph = self.rule_graphs.get('zscore')
            if zscore_graph:
                root = zscore_graph.get_node('zscore_check')
                if root and root.threshold:
                    root.threshold.adjusted = new_threshold

