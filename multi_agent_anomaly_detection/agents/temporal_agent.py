"""
Temporal Anomaly Detection Agent
=================================

Detects anomalies based on temporal comparisons including
week-over-week, month-over-month, and year-over-year changes.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base_agent import BaseDetectorAgent
from ..core.models import (
    AnomalyFlag, Feedback, DetectionContext, AnomalyType, Severity
)
from ..core.rule_graph import RuleGraph, RuleNode, NodeType, EdgeType, ThresholdConfig
from ..core.interpretable_tree import InterpretableTreeAgent
from ..core.knowledge_base import KnowledgeBase


class TemporalAgent(BaseDetectorAgent):
    """
    Agent for temporal anomaly detection.
    
    Uses hybrid tree + graph approach for:
    - Week-over-week comparisons
    - Month-over-month changes
    - Trend detection
    - Lag-based anomalies
    """
    
    def __init__(self, agent_id: str = None, name: str = "Temporal Agent",
                 knowledge_base: KnowledgeBase = None,
                 wow_threshold: float = 0.5,
                 mom_threshold: float = 0.3):
        """
        Initialize the temporal agent.
        
        Args:
            agent_id: Unique identifier
            name: Agent name
            knowledge_base: Shared knowledge base
            wow_threshold: Week-over-week change threshold (50% by default)
            mom_threshold: Month-over-month change threshold (30% by default)
        """
        super().__init__(agent_id, name, knowledge_base)
        
        self.config = {
            'wow_threshold': wow_threshold,  # Week-over-week
            'mom_threshold': mom_threshold,  # Month-over-month
            'trend_window': 4,               # Weeks to consider for trend
            'min_historical_data': 4,        # Minimum weeks of history
            'lag_deviation_threshold': 2.5   # Z-score for lag deviation
        }
        
        # Temporal trees per entity
        self.entity_trees: Dict[str, InterpretableTreeAgent] = {}
        
        # Initialize rule graphs
        self._init_rule_graphs()
    
    @property
    def agent_type(self) -> str:
        return "temporal"
    
    @property
    def anomaly_types(self) -> List[AnomalyType]:
        return [AnomalyType.TEMPORAL]
    
    def _init_rule_graphs(self) -> None:
        """Initialize temporal rule graphs."""
        # Week-over-week graph
        self.rule_graphs['wow'] = self._create_wow_graph()
        
        # Month-over-month graph
        self.rule_graphs['mom'] = self._create_mom_graph()
        
        # Trend graph
        self.rule_graphs['trend'] = self._create_trend_graph()
    
    def _create_wow_graph(self) -> RuleGraph:
        """Create week-over-week comparison rule graph."""
        graph = RuleGraph(
            graph_id="temp_wow",
            name="Week-over-Week Change Detection",
            description="Detects significant changes from the previous week"
        )
        
        # Check WoW change
        wow_check = RuleNode(
            node_id="wow_change",
            node_type=NodeType.THRESHOLD,
            name="WoW Change Check",
            description="Check week-over-week percentage change",
            condition="abs(wow_pct_change) > threshold",
            field_name="wow_pct_change",
            operator="abs>",
            threshold=ThresholdConfig(
                base=self.config['wow_threshold'],
                min_value=0.2,
                max_value=2.0
            )
        )
        graph.add_node(wow_check, is_root=True)
        
        # Check if very large change
        large_change = RuleNode(
            node_id="large_wow",
            node_type=NodeType.THRESHOLD,
            name="Large WoW Change",
            description="Check for very large week-over-week change",
            condition="abs(wow_pct_change) > large_threshold",
            field_name="wow_pct_change",
            operator="abs>",
            threshold=ThresholdConfig(base=self.config['wow_threshold'] * 2)
        )
        graph.add_node(large_change)
        
        # Flag critical
        flag_critical = RuleNode(
            node_id="flag_critical",
            node_type=NodeType.ACTION,
            name="Flag Critical WoW",
            description="Critical week-over-week change",
            condition="Critical change",
            action="flag_anomaly",
            severity="high"
        )
        graph.add_node(flag_critical)
        
        # Flag medium
        flag_medium = RuleNode(
            node_id="flag_medium",
            node_type=NodeType.ACTION,
            name="Flag Medium WoW",
            description="Notable week-over-week change",
            condition="Notable change",
            action="flag_anomaly",
            severity="medium"
        )
        graph.add_node(flag_medium)
        
        # Normal
        normal = RuleNode(
            node_id="normal",
            node_type=NodeType.ACTION,
            name="Normal",
            description="Normal week-over-week variation",
            condition="Normal",
            action="normal",
            severity="low"
        )
        graph.add_node(normal)
        
        graph.add_edge("wow_change", "large_wow", EdgeType.CHAIN)
        graph.add_edge("large_wow", "flag_critical", EdgeType.ESCALATE)
        graph.add_edge("large_wow", "flag_medium", EdgeType.ELSE)
        graph.add_edge("wow_change", "normal", EdgeType.ELSE)
        
        return graph
    
    def _create_mom_graph(self) -> RuleGraph:
        """Create month-over-month comparison rule graph."""
        graph = RuleGraph(
            graph_id="temp_mom",
            name="Month-over-Month Change Detection",
            description="Detects significant changes from the same week last month"
        )
        
        # Check MoM change
        mom_check = RuleNode(
            node_id="mom_change",
            node_type=NodeType.THRESHOLD,
            name="MoM Change Check",
            description="Check month-over-month percentage change",
            condition="abs(mom_pct_change) > threshold",
            field_name="mom_pct_change",
            operator="abs>",
            threshold=ThresholdConfig(
                base=self.config['mom_threshold'],
                min_value=0.1,
                max_value=1.0
            )
        )
        graph.add_node(mom_check, is_root=True)
        
        # Flag MoM anomaly
        flag_mom = RuleNode(
            node_id="flag_mom",
            node_type=NodeType.ACTION,
            name="Flag MoM Anomaly",
            description="Significant month-over-month change",
            condition="MoM anomaly",
            action="flag_anomaly",
            severity="medium"
        )
        graph.add_node(flag_mom)
        
        # Normal
        normal = RuleNode(
            node_id="normal",
            node_type=NodeType.ACTION,
            name="Normal",
            description="Normal month-over-month variation",
            condition="Normal",
            action="normal",
            severity="low"
        )
        graph.add_node(normal)
        
        graph.add_edge("mom_change", "flag_mom", EdgeType.ESCALATE)
        graph.add_edge("mom_change", "normal", EdgeType.ELSE)
        
        return graph
    
    def _create_trend_graph(self) -> RuleGraph:
        """Create trend detection rule graph."""
        graph = RuleGraph(
            graph_id="temp_trend",
            name="Trend Anomaly Detection",
            description="Detects anomalies in trend direction and magnitude"
        )
        
        # Check trend reversal
        trend_check = RuleNode(
            node_id="trend_reversal",
            node_type=NodeType.CONDITION,
            name="Trend Reversal",
            description="Check for sudden trend reversal",
            condition="trend direction changed significantly",
            condition_code="data.get('trend_reversal', False)"
        )
        graph.add_node(trend_check, is_root=True)
        
        # Flag trend anomaly
        flag_trend = RuleNode(
            node_id="flag_trend",
            node_type=NodeType.ACTION,
            name="Flag Trend Anomaly",
            description="Trend reversal or acceleration detected",
            condition="Trend anomaly",
            action="flag_anomaly",
            severity="medium"
        )
        graph.add_node(flag_trend)
        
        # Check acceleration
        accel_check = RuleNode(
            node_id="acceleration",
            node_type=NodeType.THRESHOLD,
            name="Trend Acceleration",
            description="Check for unusual acceleration/deceleration",
            condition="abs(trend_acceleration) > threshold",
            field_name="trend_acceleration",
            operator="abs>",
            threshold=ThresholdConfig(base=2.0)
        )
        graph.add_node(accel_check)
        
        # Flag acceleration
        flag_accel = RuleNode(
            node_id="flag_accel",
            node_type=NodeType.ACTION,
            name="Flag Acceleration",
            description="Unusual trend acceleration",
            condition="Acceleration anomaly",
            action="flag_anomaly",
            severity="low"
        )
        graph.add_node(flag_accel)
        
        # Normal
        normal = RuleNode(
            node_id="normal",
            node_type=NodeType.ACTION,
            name="Normal",
            description="Normal trend behavior",
            condition="Normal",
            action="normal",
            severity="low"
        )
        graph.add_node(normal)
        
        graph.add_edge("trend_reversal", "flag_trend", EdgeType.ESCALATE)
        graph.add_edge("trend_reversal", "acceleration", EdgeType.ELSE)
        graph.add_edge("acceleration", "flag_accel", EdgeType.ESCALATE)
        graph.add_edge("acceleration", "normal", EdgeType.ELSE)
        
        return graph
    
    def detect(self, data: pd.DataFrame, 
               context: DetectionContext = None) -> List[AnomalyFlag]:
        """Detect temporal anomalies."""
        flags = []
        modifier = context.threshold_modifier if context else 1.0
        
        if 'Entity' in data.columns:
            entities = data['Entity'].unique()
        else:
            entities = [context.entity if context else 'default']
        
        for entity in entities:
            entity_data = data[data['Entity'] == entity] if 'Entity' in data.columns else data
            
            if len(entity_data) < self.config['min_historical_data']:
                continue
            
            # Week-over-week detection
            wow_flags = self._detect_wow_anomalies(entity_data, entity, modifier)
            flags.extend(wow_flags)
            
            # Month-over-month detection (using lag4)
            mom_flags = self._detect_mom_anomalies(entity_data, entity, modifier)
            flags.extend(mom_flags)
            
            # Trend detection
            trend_flags = self._detect_trend_anomalies(entity_data, entity, modifier)
            flags.extend(trend_flags)
            
            # Lag-based anomalies
            lag_flags = self._detect_lag_anomalies(entity_data, entity, modifier)
            flags.extend(lag_flags)
        
        return flags
    
    def _detect_wow_anomalies(self, data: pd.DataFrame, entity: str,
                               modifier: float) -> List[AnomalyFlag]:
        """Detect week-over-week anomalies."""
        flags = []
        
        if 'Total_Net' not in data.columns or 'Net_Lag1' not in data.columns:
            return flags
        
        wow_graph = self.rule_graphs.get('wow')
        threshold = self.config['wow_threshold'] * modifier
        
        for idx, row in data.iterrows():
            current = row.get('Total_Net', 0)
            previous = row.get('Net_Lag1', 0)
            
            if previous == 0:
                continue
            
            pct_change = (current - previous) / abs(previous)
            
            if abs(pct_change) > threshold:
                # Evaluate through rule graph
                eval_data = {'wow_pct_change': pct_change}
                results = wow_graph.evaluate(eval_data, entity, modifier)
                
                for result in results:
                    if result['action'] == 'flag_anomaly':
                        timestamp = row.get('Week_Start') or row.get('timestamp')
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(timestamp)
                        elif not isinstance(timestamp, datetime):
                            timestamp = datetime.now()
                        
                        flag = self.create_flag(
                            entity=entity,
                            timestamp=timestamp,
                            anomaly_type=AnomalyType.TEMPORAL,
                            severity=Severity(result['severity']),
                            confidence=result.get('confidence', 0.8),
                            metric_name="wow_pct_change",
                            metric_value=pct_change,
                            threshold=threshold,
                            description=f"Week-over-week change: {pct_change:+.1%}",
                            explanation=f"Cash flow changed from ${previous:,.2f} to ${current:,.2f} "
                                       f"({pct_change:+.1%}) compared to the previous week. "
                                       f"This exceeds the {threshold:.0%} threshold.",
                            rule_id="temp_wow",
                            decision_path=result.get('decision_path', []),
                            contributing_factors={
                                'current_value': current,
                                'previous_value': previous,
                                'pct_change': pct_change
                            }
                        )
                        flags.append(flag)
        
        return flags
    
    def _detect_mom_anomalies(self, data: pd.DataFrame, entity: str,
                               modifier: float) -> List[AnomalyFlag]:
        """Detect month-over-month anomalies using lag4."""
        flags = []
        
        if 'Total_Net' not in data.columns or 'Net_Lag4' not in data.columns:
            return flags
        
        threshold = self.config['mom_threshold'] * modifier
        
        for idx, row in data.iterrows():
            current = row.get('Total_Net', 0)
            month_ago = row.get('Net_Lag4', 0)  # 4 weeks = ~1 month
            
            if month_ago == 0:
                continue
            
            pct_change = (current - month_ago) / abs(month_ago)
            
            if abs(pct_change) > threshold:
                timestamp = row.get('Week_Start') or row.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                flag = self.create_flag(
                    entity=entity,
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.TEMPORAL,
                    severity=Severity.MEDIUM if abs(pct_change) < threshold * 2 else Severity.HIGH,
                    confidence=0.75,
                    metric_name="mom_pct_change",
                    metric_value=pct_change,
                    threshold=threshold,
                    description=f"Month-over-month change: {pct_change:+.1%}",
                    explanation=f"Cash flow changed from ${month_ago:,.2f} to ${current:,.2f} "
                               f"({pct_change:+.1%}) compared to the same week last month.",
                    rule_id="temp_mom",
                    contributing_factors={
                        'current_value': current,
                        'month_ago_value': month_ago,
                        'pct_change': pct_change
                    }
                )
                flags.append(flag)
        
        return flags
    
    def _detect_trend_anomalies(self, data: pd.DataFrame, entity: str,
                                 modifier: float) -> List[AnomalyFlag]:
        """Detect trend reversals and accelerations."""
        flags = []
        
        if 'Total_Net' not in data.columns or len(data) < self.config['trend_window'] + 1:
            return flags
        
        values = data['Total_Net'].values
        window = self.config['trend_window']
        
        for i in range(window, len(values)):
            # Calculate trend slope for current and previous windows
            current_window = values[i-window:i]
            prev_window = values[max(0, i-window*2):i-window] if i >= window*2 else None
            
            if prev_window is None or len(prev_window) < 2:
                continue
            
            # Simple trend: regression slope
            x = np.arange(len(current_window))
            current_slope = np.polyfit(x, current_window, 1)[0] if len(current_window) > 1 else 0
            prev_slope = np.polyfit(np.arange(len(prev_window)), prev_window, 1)[0] if len(prev_window) > 1 else 0
            
            # Detect trend reversal (sign change in slope)
            trend_reversal = (current_slope * prev_slope < 0) and (abs(current_slope) > 100 and abs(prev_slope) > 100)
            
            # Detect acceleration
            if prev_slope != 0:
                acceleration = (current_slope - prev_slope) / abs(prev_slope)
            else:
                acceleration = 0
            
            if trend_reversal or abs(acceleration) > 2.0:
                row = data.iloc[i]
                timestamp = row.get('Week_Start') or row.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                if trend_reversal:
                    description = f"Trend reversal detected (slope: {prev_slope:.2f} -> {current_slope:.2f})"
                    severity = Severity.MEDIUM
                else:
                    description = f"Unusual trend acceleration: {acceleration:.1%}"
                    severity = Severity.LOW
                
                flag = self.create_flag(
                    entity=entity,
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.TEMPORAL,
                    severity=severity,
                    confidence=0.7,
                    metric_name="trend_change",
                    metric_value=acceleration if not trend_reversal else current_slope - prev_slope,
                    threshold=2.0,
                    description=description,
                    explanation=f"The cash flow trend has {'reversed' if trend_reversal else 'accelerated significantly'}. "
                               f"Previous trend: {prev_slope:+.2f}/week, Current trend: {current_slope:+.2f}/week.",
                    rule_id="temp_trend",
                    contributing_factors={
                        'current_slope': current_slope,
                        'previous_slope': prev_slope,
                        'trend_reversal': trend_reversal,
                        'acceleration': acceleration
                    }
                )
                flags.append(flag)
        
        return flags
    
    def _detect_lag_anomalies(self, data: pd.DataFrame, entity: str,
                               modifier: float) -> List[AnomalyFlag]:
        """Detect anomalies in lag relationships."""
        flags = []
        
        lag_cols = ['Net_Lag1', 'Net_Lag2', 'Net_Lag4']
        available_lags = [c for c in lag_cols if c in data.columns]
        
        if not available_lags or 'Total_Net' not in data.columns:
            return flags
        
        threshold = self.config['lag_deviation_threshold'] * modifier
        
        for idx, row in data.iterrows():
            current = row.get('Total_Net', 0)
            
            # Calculate expected value based on lags
            lag_values = [row.get(lag) for lag in available_lags if not pd.isna(row.get(lag))]
            if not lag_values:
                continue
            
            expected = np.mean(lag_values)
            std_lags = np.std(lag_values) if len(lag_values) > 1 else abs(expected) * 0.2
            
            if std_lags == 0:
                std_lags = abs(expected) * 0.2 or 1
            
            deviation = (current - expected) / std_lags
            
            if abs(deviation) > threshold:
                timestamp = row.get('Week_Start') or row.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                flag = self.create_flag(
                    entity=entity,
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.TEMPORAL,
                    severity=self.severity_from_deviation(deviation),
                    confidence=0.7,
                    metric_name="lag_deviation",
                    metric_value=deviation,
                    threshold=threshold,
                    description=f"Deviation from lag-based expectation: {deviation:.2f}σ",
                    explanation=f"Current value ${current:,.2f} deviates {deviation:.2f} standard deviations "
                               f"from the expected value ${expected:,.2f} based on recent history.",
                    rule_id="temp_lag",
                    contributing_factors={
                        'current': current,
                        'expected': expected,
                        'deviation_sigma': deviation,
                        'lag_values': lag_values
                    }
                )
                flags.append(flag)
        
        return flags
    
    def explain(self, flag: AnomalyFlag) -> str:
        """Generate explanation for temporal anomaly."""
        lines = [
            f"Temporal Anomaly Detected",
            f"=========================",
            f"Entity: {flag.entity}",
            f"Time: {flag.timestamp}",
            f"Severity: {flag.severity.value.upper()}",
            f"",
            f"Type: {flag.metric_name}",
            f"Value: {flag.metric_value:.4f}",
            f"",
            f"Description: {flag.description}",
            f"",
            flag.explanation
        ]
        
        if flag.contributing_factors:
            lines.append("")
            lines.append("Contributing Factors:")
            for key, value in flag.contributing_factors.items():
                if isinstance(value, float):
                    lines.append(f"  - {key}: {value:.4f}")
                elif isinstance(value, list):
                    lines.append(f"  - {key}: {value}")
                else:
                    lines.append(f"  - {key}: {value}")
        
        return "\n".join(lines)

