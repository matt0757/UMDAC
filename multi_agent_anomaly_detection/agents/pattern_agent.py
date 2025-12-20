"""
Pattern Detection Agent
========================

Detects anomalies based on seasonal patterns, day-of-week effects,
and learned temporal patterns using decision trees.
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
from ..core.interpretable_tree import InterpretableTreeAgent
from ..core.knowledge_base import KnowledgeBase


class PatternAgent(BaseDetectorAgent):
    """
    Agent for pattern-based anomaly detection.
    
    Uses decision trees trained on temporal features to detect:
    - Seasonal pattern violations
    - Day-of-week anomalies
    - Week-of-month patterns
    - Quarter-end effects
    """
    
    def __init__(self, agent_id: str = None, name: str = "Pattern Agent",
                 knowledge_base: KnowledgeBase = None,
                 pattern_deviation_threshold: float = 2.0):
        """
        Initialize the pattern agent.
        
        Args:
            agent_id: Unique identifier
            name: Agent name
            knowledge_base: Shared knowledge base
            pattern_deviation_threshold: Threshold for pattern deviation
        """
        super().__init__(agent_id, name, knowledge_base)
        
        self.config = {
            'pattern_deviation_threshold': pattern_deviation_threshold,
            'min_training_samples': 20,
            'tree_max_depth': 4
        }
        
        # Decision trees per entity
        self.entity_trees: Dict[str, InterpretableTreeAgent] = {}
        
        # Learned patterns (mean/std per temporal feature combination)
        self.learned_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Initialize rule graph
        self._init_rule_graphs()
    
    @property
    def agent_type(self) -> str:
        return "pattern"
    
    @property
    def anomaly_types(self) -> List[AnomalyType]:
        return [AnomalyType.PATTERN]
    
    def _init_rule_graphs(self) -> None:
        """Initialize pattern detection rule graph."""
        graph = RuleGraph(
            graph_id="pattern_seasonal",
            name="Seasonal Pattern Detection",
            description="Detects violations of learned seasonal patterns"
        )
        
        # Check week-of-month pattern
        week_check = RuleNode(
            node_id="week_pattern_check",
            node_type=NodeType.PATTERN,
            name="Week-of-Month Pattern",
            description="Check if value matches expected week-of-month pattern",
            condition="deviation from weekly pattern exceeds threshold",
            field_name="week_pattern_deviation",
            operator="abs>",
            threshold=ThresholdConfig(base=self.config['pattern_deviation_threshold'])
        )
        graph.add_node(week_check, is_root=True)
        
        # Check month pattern
        month_check = RuleNode(
            node_id="month_pattern_check",
            node_type=NodeType.PATTERN,
            name="Monthly Pattern",
            description="Check if value matches expected monthly pattern",
            condition="deviation from monthly pattern exceeds threshold",
            field_name="month_pattern_deviation",
            operator="abs>",
            threshold=ThresholdConfig(base=self.config['pattern_deviation_threshold'])
        )
        graph.add_node(month_check)
        
        # Flag pattern violation
        flag_pattern = RuleNode(
            node_id="flag_pattern",
            node_type=NodeType.ACTION,
            name="Flag Pattern Violation",
            description="Pattern violation detected",
            condition="Pattern anomaly",
            action="flag_anomaly",
            severity="medium"
        )
        graph.add_node(flag_pattern)
        
        # Normal
        normal = RuleNode(
            node_id="normal",
            node_type=NodeType.ACTION,
            name="Normal Pattern",
            description="Pattern matches expectations",
            condition="Normal",
            action="normal",
            severity="low"
        )
        graph.add_node(normal)
        
        # Edges
        graph.add_edge("week_pattern_check", "flag_pattern", EdgeType.ESCALATE)
        graph.add_edge("week_pattern_check", "month_pattern_check", EdgeType.ELSE)
        graph.add_edge("month_pattern_check", "flag_pattern", EdgeType.ESCALATE)
        graph.add_edge("month_pattern_check", "normal", EdgeType.ELSE)
        
        self.rule_graphs['seasonal'] = graph
    
    def train_patterns(self, data: pd.DataFrame, entity: str = None) -> Dict[str, Any]:
        """
        Train pattern models on historical data.
        
        Args:
            data: Historical data with temporal features
            entity: Optional entity to train for
        
        Returns:
            Training metrics
        """
        entities = [entity] if entity else data['Entity'].unique() if 'Entity' in data.columns else ['default']
        results = {}
        
        for ent in entities:
            ent_data = data[data['Entity'] == ent] if 'Entity' in data.columns and entity is None else data
            
            if len(ent_data) < self.config['min_training_samples']:
                results[ent] = {'status': 'insufficient_data', 'samples': len(ent_data)}
                continue
            
            # Learn patterns per temporal grouping
            self.learned_patterns[ent] = {}
            
            # Week-of-month patterns
            if 'Week_of_Month' in ent_data.columns and 'Total_Net' in ent_data.columns:
                week_patterns = ent_data.groupby('Week_of_Month')['Total_Net'].agg(['mean', 'std']).to_dict('index')
                self.learned_patterns[ent]['week_of_month'] = week_patterns
            
            # Monthly patterns
            if 'Month' in ent_data.columns and 'Total_Net' in ent_data.columns:
                month_patterns = ent_data.groupby('Month')['Total_Net'].agg(['mean', 'std']).to_dict('index')
                self.learned_patterns[ent]['month'] = month_patterns
            
            # Quarter patterns
            if 'Quarter' in ent_data.columns and 'Total_Net' in ent_data.columns:
                quarter_patterns = ent_data.groupby('Quarter')['Total_Net'].agg(['mean', 'std']).to_dict('index')
                self.learned_patterns[ent]['quarter'] = quarter_patterns
            
            # Month-end effects
            if 'Is_Month_End' in ent_data.columns and 'Total_Net' in ent_data.columns:
                month_end_patterns = ent_data.groupby('Is_Month_End')['Total_Net'].agg(['mean', 'std']).to_dict('index')
                self.learned_patterns[ent]['month_end'] = month_end_patterns
            
            # Train decision tree for anomaly prediction
            tree_features = ['Week_of_Month', 'Month', 'Quarter', 'Is_Month_End',
                           'Net_Lag1', 'Net_Rolling4_Mean', 'Transaction_Count']
            available_features = [f for f in tree_features if f in ent_data.columns]
            
            if len(available_features) >= 3 and 'Total_Net' in ent_data.columns:
                # Create anomaly labels based on z-score
                mean_val = ent_data['Total_Net'].mean()
                std_val = ent_data['Total_Net'].std()
                if std_val > 0:
                    zscore = (ent_data['Total_Net'] - mean_val) / std_val
                    labels = (abs(zscore) > 2).astype(int)
                    
                    tree = InterpretableTreeAgent(
                        tree_id=f"pattern_{ent}",
                        name=f"Pattern Tree for {ent}",
                        max_depth=self.config['tree_max_depth']
                    )
                    
                    X = ent_data[available_features].copy()
                    metrics = tree.train(X, labels, class_names=['Normal', 'Anomaly'])
                    self.entity_trees[ent] = tree
                    
                    results[ent] = {
                        'status': 'trained',
                        'samples': len(ent_data),
                        'tree_accuracy': metrics['training_accuracy'],
                        'patterns_learned': list(self.learned_patterns[ent].keys())
                    }
                else:
                    results[ent] = {'status': 'no_variance', 'samples': len(ent_data)}
            else:
                results[ent] = {
                    'status': 'partial',
                    'samples': len(ent_data),
                    'patterns_learned': list(self.learned_patterns.get(ent, {}).keys())
                }
        
        return results
    
    def detect(self, data: pd.DataFrame, 
               context: DetectionContext = None) -> List[AnomalyFlag]:
        """Detect pattern-based anomalies."""
        flags = []
        modifier = context.threshold_modifier if context else 1.0
        
        if 'Entity' in data.columns:
            entities = data['Entity'].unique()
        else:
            entities = [context.entity if context else 'default']
        
        for entity in entities:
            entity_data = data[data['Entity'] == entity] if 'Entity' in data.columns else data
            
            # Pattern deviation detection
            pattern_flags = self._detect_pattern_deviations(entity_data, entity, modifier)
            flags.extend(pattern_flags)
            
            # Tree-based detection if trained
            if entity in self.entity_trees:
                tree_flags = self._detect_with_tree(entity_data, entity)
                flags.extend(tree_flags)
        
        return flags
    
    def _detect_pattern_deviations(self, data: pd.DataFrame, entity: str,
                                    modifier: float) -> List[AnomalyFlag]:
        """Detect deviations from learned patterns."""
        flags = []
        patterns = self.learned_patterns.get(entity, {})
        
        if not patterns:
            return flags
        
        threshold = self.config['pattern_deviation_threshold'] * modifier
        
        for idx, row in data.iterrows():
            deviations = []
            
            # Check week-of-month pattern
            if 'week_of_month' in patterns and 'Week_of_Month' in row.index:
                week = row.get('Week_of_Month')
                if week in patterns['week_of_month']:
                    expected = patterns['week_of_month'][week]
                    actual = row.get('Total_Net', 0)
                    std = expected.get('std', 1) or 1
                    deviation = abs(actual - expected.get('mean', 0)) / std
                    if deviation > threshold:
                        deviations.append(('week_of_month', deviation, expected, actual))
            
            # Check monthly pattern
            if 'month' in patterns and 'Month' in row.index:
                month = row.get('Month')
                if month in patterns['month']:
                    expected = patterns['month'][month]
                    actual = row.get('Total_Net', 0)
                    std = expected.get('std', 1) or 1
                    deviation = abs(actual - expected.get('mean', 0)) / std
                    if deviation > threshold:
                        deviations.append(('month', deviation, expected, actual))
            
            # Check month-end pattern
            if 'month_end' in patterns and 'Is_Month_End' in row.index:
                is_end = row.get('Is_Month_End')
                if is_end in patterns['month_end']:
                    expected = patterns['month_end'][is_end]
                    actual = row.get('Total_Net', 0)
                    std = expected.get('std', 1) or 1
                    deviation = abs(actual - expected.get('mean', 0)) / std
                    if deviation > threshold:
                        deviations.append(('month_end', deviation, expected, actual))
            
            # Create flags for significant deviations
            if deviations:
                max_deviation = max(deviations, key=lambda x: x[1])
                pattern_type, dev_value, expected_stats, actual_value = max_deviation
                
                timestamp = row.get('Week_Start') or row.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                flag = self.create_flag(
                    entity=entity,
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.PATTERN,
                    severity=Severity.MEDIUM if dev_value < 3 else Severity.HIGH,
                    confidence=min(0.9, 0.5 + dev_value * 0.1),
                    metric_name=f"{pattern_type}_deviation",
                    metric_value=dev_value,
                    threshold=threshold,
                    description=f"Pattern deviation in {pattern_type}: {dev_value:.2f}σ",
                    explanation=f"The value {actual_value:,.2f} deviates from the expected "
                               f"{pattern_type} pattern (expected: {expected_stats.get('mean', 0):,.2f} ± "
                               f"{expected_stats.get('std', 0):,.2f})",
                    rule_id="pattern_seasonal",
                    contributing_factors={
                        'pattern_type': pattern_type,
                        'expected_mean': expected_stats.get('mean', 0),
                        'expected_std': expected_stats.get('std', 0),
                        'actual_value': actual_value,
                        'deviation_sigma': dev_value
                    }
                )
                flags.append(flag)
        
        return flags
    
    def _detect_with_tree(self, data: pd.DataFrame, entity: str) -> List[AnomalyFlag]:
        """Use trained decision tree for detection."""
        flags = []
        tree = self.entity_trees.get(entity)
        
        if tree is None or tree.tree is None:
            return flags
        
        # Prepare features
        available_features = [f for f in tree.feature_names if f in data.columns]
        if len(available_features) != len(tree.feature_names):
            return flags
        
        X = data[tree.feature_names].copy()
        predictions = tree.predict(X)
        probas = tree.predict_proba(X)
        paths = tree.get_decision_path(X)
        
        for i, (pred, proba, path) in enumerate(zip(predictions, probas, paths)):
            if pred == 1:  # Anomaly predicted
                row = data.iloc[i]
                confidence = proba[1]  # Probability of anomaly class
                
                timestamp = row.get('Week_Start') or row.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                flag = self.create_flag(
                    entity=entity,
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.PATTERN,
                    severity=Severity.MEDIUM if confidence < 0.8 else Severity.HIGH,
                    confidence=confidence,
                    metric_name="tree_prediction",
                    metric_value=confidence,
                    threshold=0.5,
                    description="Pattern anomaly detected by decision tree",
                    explanation=tree.explain_prediction(X, i),
                    rule_id=f"tree_{entity}",
                    decision_path=path,
                    contributing_factors=dict(zip(tree.feature_names, X.iloc[i].values))
                )
                flags.append(flag)
        
        return flags
    
    def explain(self, flag: AnomalyFlag) -> str:
        """Generate explanation for pattern anomaly."""
        lines = [
            f"Pattern Anomaly Detected",
            f"========================",
            f"Entity: {flag.entity}",
            f"Time: {flag.timestamp}",
            f"Severity: {flag.severity.value.upper()}",
            f"Confidence: {flag.confidence:.1%}",
            f"",
            f"Description: {flag.description}",
            f"",
            flag.explanation,
            f"",
            "Decision Path:"
        ]
        
        for step in flag.decision_path:
            lines.append(f"  -> {step}")
        
        return "\n".join(lines)
    
    def get_pattern_summary(self, entity: str) -> Dict[str, Any]:
        """Get summary of learned patterns for an entity."""
        patterns = self.learned_patterns.get(entity, {})
        tree = self.entity_trees.get(entity)
        
        return {
            'entity': entity,
            'patterns': patterns,
            'tree_trained': tree is not None,
            'tree_rules': tree.rules_to_english() if tree else []
        }

