"""
Category Anomaly Detection Agent
=================================

Detects anomalies specific to cash flow categories (AP, AR, Payroll, etc.)
using entity-specific decision trees and learned category patterns.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

from .base_agent import BaseDetectorAgent
from ..core.models import (
    AnomalyFlag, Feedback, DetectionContext, AnomalyType, Severity
)
from ..core.rule_graph import RuleGraph, RuleNode, NodeType, EdgeType, ThresholdConfig
from ..core.interpretable_tree import InterpretableTreeAgent
from ..core.knowledge_base import KnowledgeBase


class CategoryAgent(BaseDetectorAgent):
    """
    Agent for category-specific anomaly detection.
    
    Trains entity-specific decision trees for each cash flow category
    to detect unusual patterns in AP, AR, Payroll, etc.;
    """
    
    # Known cash flow categories from the data
    CATEGORIES = [
        'AP', 'AR', 'Payroll', 'Bank_charges', 'Custom_and_Duty',
        'Other_receipt', 'Non_Netting_AP', 'Loan_payment', 'Tax_payable',
        'Statutory_contribution', 'Interest_income', 'Other', 
        'Interest_charges', 'Loan_payment_and_interest_charges',
        'Loan_receipt', 'Dividend_payout', 'Netting_AP', 'Netting_AR',
        'Non_Netting_AR'
    ]
    
    def __init__(self, agent_id: str = None, name: str = "Category Agent",
                 knowledge_base: KnowledgeBase = None,
                 category_deviation_threshold: float = 5.0,  # Increased from 4.5 to further reduce false positives
                 min_category_absolute_value: float = 5000.0,  # Increased from 1000 to $5000 minimum
                 min_category_percentage: float = 0.03,  # Increased from 2% to 3% of total flow
                 max_flags_per_row: int = 2):  # Reduced from 3 to 2 flags per row
        """
        Initialize the category agent.
        
        Args:
            agent_id: Unique identifier
            name: Agent name
            knowledge_base: Shared knowledge base
            category_deviation_threshold: Z-score threshold for category anomalies (default: 4.5)
            min_category_absolute_value: Minimum absolute value to flag a category (default: $1000)
            min_category_percentage: Minimum percentage of total flow to consider significant (default: 2%)
            max_flags_per_row: Maximum number of category flags per row (default: 3)
        """
        super().__init__(agent_id, name, knowledge_base)
        
        self.config = {
            'category_deviation_threshold': category_deviation_threshold,
            'min_training_samples': 10,
            'tree_max_depth': 4,
            'significant_category_threshold': 0.15,  # 15% of total (stricter)
            'min_category_absolute_value': min_category_absolute_value,
            'min_category_percentage': min_category_percentage,
            'max_flags_per_row': max_flags_per_row
        }
        
        # Decision trees per entity per category
        self.category_trees: Dict[str, Dict[str, InterpretableTreeAgent]] = {}
        
        # Learned category statistics
        self.category_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        # Initialize rule graphs
        self._init_rule_graphs()
    
    @property
    def agent_type(self) -> str:
        return "category"
    
    @property
    def anomaly_types(self) -> List[AnomalyType]:
        return [AnomalyType.CATEGORY]
    
    def _init_rule_graphs(self) -> None:
        """Initialize category detection rule graphs."""
        # Create a general category deviation graph
        graph = RuleGraph(
            graph_id="cat_deviation",
            name="Category Deviation Detection",
            description="Detects unusual deviations in category values"
        )
        
        # Root: Check category deviation
        dev_check = RuleNode(
            node_id="cat_deviation_check",
            node_type=NodeType.THRESHOLD,
            name="Category Deviation",
            description="Check if category value deviates from historical norm",
            condition="abs(category_zscore) > threshold",
            field_name="category_zscore",
            operator="abs>",
            threshold=ThresholdConfig(
                base=self.config['category_deviation_threshold'],
                min_value=1.5,
                max_value=4.0
            )
        )
        graph.add_node(dev_check, is_root=True)
        
        # High deviation check
        high_dev = RuleNode(
            node_id="high_deviation",
            node_type=NodeType.THRESHOLD,
            name="High Deviation",
            description="Check for severe deviation",
            condition="abs(category_zscore) > high_threshold",
            field_name="category_zscore",
            operator="abs>",
            threshold=ThresholdConfig(base=self.config['category_deviation_threshold'] * 1.5)
        )
        graph.add_node(high_dev)
        
        # Flag high
        flag_high = RuleNode(
            node_id="flag_high",
            node_type=NodeType.ACTION,
            name="Flag High Category Anomaly",
            description="Severe category deviation",
            condition="High deviation",
            action="flag_anomaly",
            severity="high"
        )
        graph.add_node(flag_high)
        
        # Flag medium
        flag_medium = RuleNode(
            node_id="flag_medium",
            node_type=NodeType.ACTION,
            name="Flag Medium Category Anomaly",
            description="Notable category deviation",
            condition="Medium deviation",
            action="flag_anomaly",
            severity="medium"
        )
        graph.add_node(flag_medium)
        
        # Normal
        normal = RuleNode(
            node_id="normal",
            node_type=NodeType.ACTION,
            name="Normal",
            description="Normal category values",
            condition="Normal",
            action="normal",
            severity="low"
        )
        graph.add_node(normal)
        
        graph.add_edge("cat_deviation_check", "high_deviation", EdgeType.CHAIN)
        graph.add_edge("high_deviation", "flag_high", EdgeType.ESCALATE)
        graph.add_edge("high_deviation", "flag_medium", EdgeType.ELSE)
        graph.add_edge("cat_deviation_check", "normal", EdgeType.ELSE)
        
        self.rule_graphs['deviation'] = graph
    
    def train_category_models(self, data: pd.DataFrame, entity: str = None) -> Dict[str, Any]:
        """
        Train category-specific models on historical data.
        
        Args:
            data: Historical data with category columns
            entity: Optional entity to train for
        
        Returns:
            Training metrics per category
        """
        entities = [entity] if entity else data['Entity'].unique() if 'Entity' in data.columns else ['default']
        results = {}
        
        for ent in entities:
            ent_data = data[data['Entity'] == ent] if 'Entity' in data.columns and entity is None else data
            
            if len(ent_data) < self.config['min_training_samples']:
                results[ent] = {'status': 'insufficient_data'}
                continue
            
            results[ent] = {}
            
            # Initialize storage for this entity
            if ent not in self.category_stats:
                self.category_stats[ent] = {}
            if ent not in self.category_trees:
                self.category_trees[ent] = {}
            
            # Find category columns
            cat_net_cols = [c for c in ent_data.columns if c.startswith('Cat_') and c.endswith('_Net')]
            
            for cat_col in cat_net_cols:
                cat_name = cat_col.replace('Cat_', '').replace('_Net', '')
                values = ent_data[cat_col].dropna()
                
                if len(values) < 2:
                    continue
                
                # Calculate and store statistics
                self.category_stats[ent][cat_name] = {
                    'mean': values.mean(),
                    'std': values.std() if values.std() > 0 else abs(values.mean()) * 0.1,
                    'min': values.min(),
                    'max': values.max(),
                    'q1': values.quantile(0.25),
                    'q3': values.quantile(0.75),
                    'count': len(values)
                }
                
                results[ent][cat_name] = {
                    'samples': len(values),
                    'mean': float(values.mean()),
                    'std': float(values.std())
                }
                
                # Train decision tree for anomaly detection if enough samples
                if len(values) >= self.config['min_training_samples']:
                    # Create features for the tree
                    cat_lag_col = f"Cat_{cat_name}_Lag1"
                    cat_rolling_col = f"Cat_{cat_name}_Rolling4_Mean"
                    
                    features = ['Week_of_Month', 'Month', 'Quarter', 'Is_Month_End']
                    if cat_lag_col in ent_data.columns:
                        features.append(cat_lag_col)
                    if cat_rolling_col in ent_data.columns:
                        features.append(cat_rolling_col)
                    
                    available_features = [f for f in features if f in ent_data.columns]
                    
                    if len(available_features) >= 2:
                        # Create anomaly labels
                        stats = self.category_stats[ent][cat_name]
                        zscore = (values - stats['mean']) / stats['std']
                        labels = (abs(zscore) > self.config['category_deviation_threshold']).astype(int)
                        
                        if labels.sum() > 0:  # At least one anomaly
                            tree = InterpretableTreeAgent(
                                tree_id=f"cat_{ent}_{cat_name}",
                                name=f"Category Tree: {cat_name} for {ent}",
                                max_depth=self.config['tree_max_depth']
                            )
                            
                            # Align data
                            X = ent_data.loc[values.index, available_features]
                            y = labels.loc[values.index]
                            
                            metrics = tree.train(X, y, class_names=['Normal', 'Anomaly'])
                            self.category_trees[ent][cat_name] = tree
                            
                            results[ent][cat_name]['tree_trained'] = True
                            results[ent][cat_name]['tree_accuracy'] = metrics['training_accuracy']
        
        return results
    
    def detect(self, data: pd.DataFrame, 
               context: DetectionContext = None) -> List[AnomalyFlag]:
        """Detect category-specific anomalies with global limits."""
        flags = []
        modifier = context.threshold_modifier if context else 1.0
        
        if 'Entity' in data.columns:
            entities = data['Entity'].unique()
        else:
            entities = [context.entity if context else 'default']
        
        # Global limit: max flags per entity (prevents one entity from dominating)
        max_flags_per_entity = 50  # Reasonable limit per entity
        
        for entity in entities:
            entity_data = data[data['Entity'] == entity] if 'Entity' in data.columns else data
            entity_flags = []
            
            # Statistical category anomalies
            stat_flags = self._detect_category_deviations(entity_data, entity, modifier)
            entity_flags.extend(stat_flags)
            
            # Tree-based detection
            tree_flags = self._detect_with_trees(entity_data, entity)
            entity_flags.extend(tree_flags)
            
            # Ratio anomalies
            ratio_flags = self._detect_ratio_anomalies(entity_data, entity, modifier)
            entity_flags.extend(ratio_flags)
            
            # Sort all entity flags by severity and confidence, keep only top N
            if len(entity_flags) > max_flags_per_entity:
                entity_flags.sort(
                    key=lambda f: (
                        {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(f.severity.value, 0),
                        f.confidence
                    ),
                    reverse=True
                )
                entity_flags = entity_flags[:max_flags_per_entity]
            
            flags.extend(entity_flags)
        
        return flags
    
    def _detect_category_deviations(self, data: pd.DataFrame, entity: str,
                                     modifier: float) -> List[AnomalyFlag]:
        """Detect statistical deviations in categories with filtering to prevent over-flagging."""
        flags = []
        
        entity_stats = self.category_stats.get(entity, {})
        if not entity_stats:
            # Calculate statistics on the fly
            cat_cols = [c for c in data.columns if c.startswith('Cat_') and c.endswith('_Net')]
            for cat_col in cat_cols:
                cat_name = cat_col.replace('Cat_', '').replace('_Net', '')
                values = data[cat_col].dropna()
                if len(values) > 1:
                    entity_stats[cat_name] = {
                        'mean': values.mean(),
                        'std': values.std() if values.std() > 0 else abs(values.mean()) * 0.1 or 1
                    }
        
        threshold = self.config['category_deviation_threshold'] * modifier
        min_abs_value = self.config['min_category_absolute_value']
        min_pct = self.config['min_category_percentage']
        max_per_row = self.config['max_flags_per_row']
        
        # Calculate total flow column for percentage checks
        total_col = 'Total_Net' if 'Total_Net' in data.columns else None
        if total_col is None:
            # Try to calculate from category columns
            cat_cols = [c for c in data.columns if c.startswith('Cat_') and c.endswith('_Net')]
            if cat_cols:
                total_col = '__calculated_total__'
        
        for idx, row in data.iterrows():
            row_flags = []  # Collect flags for this row first
            
            # Calculate total flow for this row if needed
            if total_col == '__calculated_total__':
                row_total = sum(abs(row.get(f"Cat_{cat}_Net", 0)) for cat in entity_stats.keys())
            elif total_col:
                row_total = abs(row.get(total_col, 0))
            else:
                row_total = 1.0  # Fallback to avoid division by zero
            
            for cat_name, stats in entity_stats.items():
                cat_col = f"Cat_{cat_name}_Net"
                if cat_col not in row.index:
                    continue
                
                value = row.get(cat_col, 0)
                if pd.isna(value) or value == 0:
                    continue
                
                abs_value = abs(value)
                
                # Filter 1: Minimum absolute value threshold
                if abs_value < min_abs_value:
                    continue
                
                # Filter 2: Minimum percentage of total flow
                pct_of_total = 0.0
                if row_total > 0:
                    pct_of_total = abs_value / row_total
                    if pct_of_total < min_pct:
                        continue
                
                mean = stats.get('mean', 0)
                std = stats.get('std', 1)
                
                zscore = (value - mean) / std if std > 0 else 0
                
                # Filter 3: Z-score threshold
                if abs(zscore) > threshold:
                    timestamp = row.get('Week_Start') or row.get('timestamp')
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                    elif not isinstance(timestamp, datetime):
                        timestamp = datetime.now()
                    
                    flag = self.create_flag(
                        entity=entity,
                        timestamp=timestamp,
                        anomaly_type=AnomalyType.CATEGORY,
                        severity=self.severity_from_deviation(zscore),
                        confidence=min(0.9, 0.5 + abs(zscore) * 0.1),
                        metric_name=f"{cat_name}_zscore",
                        metric_value=zscore,
                        threshold=threshold,
                        description=f"Unusual {cat_name}: {zscore:.2f}σ from mean",
                        explanation=f"The {cat_name} category value ${value:,.2f} is {abs(zscore):.2f} "
                                   f"standard deviations from the historical mean (${mean:,.2f} ± ${std:,.2f}).",
                        rule_id="cat_deviation",
                        contributing_factors={
                            'category': cat_name,
                            'value': value,
                            'mean': mean,
                            'std': std,
                            'zscore': zscore,
                            'abs_value': abs_value,
                            'pct_of_total': pct_of_total
                        }
                    )
                    row_flags.append((abs(zscore), flag))  # Store with z-score for sorting
            
            # Filter 4: Limit flags per row - keep only top N by severity/confidence
            if row_flags:
                # Sort by z-score (severity) descending
                row_flags.sort(key=lambda x: x[0], reverse=True)
                # Take only top max_per_row
                for _, flag in row_flags[:max_per_row]:
                    flags.append(flag)
        
        return flags
    
    def _detect_with_trees(self, data: pd.DataFrame, entity: str) -> List[AnomalyFlag]:
        """Use trained category trees for detection with filtering."""
        flags = []
        
        entity_trees = self.category_trees.get(entity, {})
        if not entity_trees:
            return flags
        
        min_abs_value = self.config['min_category_absolute_value']
        min_pct = self.config['min_category_percentage']
        max_per_row = self.config['max_flags_per_row']
        
        # Calculate total flow column for percentage checks
        total_col = 'Total_Net' if 'Total_Net' in data.columns else None
        if total_col is None:
            cat_cols = [c for c in data.columns if c.startswith('Cat_') and c.endswith('_Net')]
            if cat_cols:
                total_col = '__calculated_total__'
        
        # Collect all tree flags first, then filter
        all_tree_flags = []
        
        for cat_name, tree in entity_trees.items():
            if tree.tree is None:
                continue
            
            # Prepare features
            available_features = [f for f in tree.feature_names if f in data.columns]
            if len(available_features) != len(tree.feature_names):
                continue
            
            X = data[tree.feature_names].copy()
            
            try:
                predictions = tree.predict(X)
                probas = tree.predict_proba(X)
                paths = tree.get_decision_path(X)
            except Exception:
                continue
            
            for i, (pred, proba, path) in enumerate(zip(predictions, probas, paths)):
                if pred == 1:  # Anomaly
                    row = data.iloc[i]
                    confidence = proba[1]
                    
                    # Filter: Require higher confidence for tree predictions
                    if confidence < 0.7:  # Only flag high-confidence tree predictions
                        continue
                    
                    cat_value = row.get(f"Cat_{cat_name}_Net", 0)
                    abs_value = abs(cat_value)
                    
                    # Filter 1: Minimum absolute value
                    if abs_value < min_abs_value:
                        continue
                    
                    # Filter 2: Minimum percentage of total
                    if total_col == '__calculated_total__':
                        row_total = sum(abs(row.get(f"Cat_{c}_Net", 0)) for c in entity_trees.keys())
                    elif total_col:
                        row_total = abs(row.get(total_col, 0))
                    else:
                        row_total = 1.0
                    
                    if row_total > 0:
                        pct_of_total = abs_value / row_total
                        if pct_of_total < min_pct:
                            continue
                    
                    timestamp = row.get('Week_Start') or row.get('timestamp')
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                    elif not isinstance(timestamp, datetime):
                        timestamp = datetime.now()
                    
                    flag = self.create_flag(
                        entity=entity,
                        timestamp=timestamp,
                        anomaly_type=AnomalyType.CATEGORY,
                        severity=Severity.MEDIUM if confidence < 0.85 else Severity.HIGH,
                        confidence=confidence,
                        metric_name=f"{cat_name}_tree",
                        metric_value=cat_value,
                        threshold=0.7,  # Higher threshold
                        description=f"Category tree anomaly: {cat_name}",
                        explanation=tree.explain_prediction(X, i),
                        rule_id=f"cat_tree_{cat_name}",
                        decision_path=path,
                        contributing_factors={
                            'category': cat_name,
                            'tree_confidence': confidence,
                            'abs_value': abs_value,
                            'pct_of_total': pct_of_total if row_total > 0 else 0,
                            **dict(zip(tree.feature_names, X.iloc[i].values))
                        }
                    )
                    all_tree_flags.append((i, confidence, flag))  # Store with row index and confidence
        
        # Filter 3: Limit flags per row - group by row index
        row_flags_dict = {}
        for row_idx, conf, flag in all_tree_flags:
            if row_idx not in row_flags_dict:
                row_flags_dict[row_idx] = []
            row_flags_dict[row_idx].append((conf, flag))
        
        # Keep only top N flags per row
        for row_idx, row_flag_list in row_flags_dict.items():
            row_flag_list.sort(key=lambda x: x[0], reverse=True)  # Sort by confidence
            for _, flag in row_flag_list[:max_per_row]:
                flags.append(flag)
        
        return flags
    
    def _detect_ratio_anomalies(self, data: pd.DataFrame, entity: str,
                                 modifier: float) -> List[AnomalyFlag]:
        """Detect anomalies in category ratios with stricter filtering."""
        flags = []
        
        # Find ratio columns
        ratio_cols = [c for c in data.columns if c.endswith('_Ratio')]
        if not ratio_cols:
            return flags
        
        threshold = 0.85  # Increased from 0.8 to 0.85 - only flag extreme concentrations
        min_abs_value = self.config['min_category_absolute_value']
        
        for idx, row in data.iterrows():
            # Check for dominant categories (high ratios)
            high_ratios = []
            
            # Calculate total for absolute value check
            total_col = 'Total_Net' if 'Total_Net' in data.columns else None
            if total_col:
                row_total = abs(row.get(total_col, 0))
            else:
                # Try to calculate from category columns
                cat_cols = [c for c in data.columns if c.startswith('Cat_') and c.endswith('_Net')]
                row_total = sum(abs(row.get(c, 0)) for c in cat_cols) if cat_cols else 1.0
            
            for ratio_col in ratio_cols:
                ratio = row.get(ratio_col, 0)
                if pd.isna(ratio):
                    continue
                
                # Stricter threshold and check absolute value
                if ratio > threshold:  # More than 85% in one category (increased from 80%)
                    cat_name = ratio_col.replace('_Ratio', '')
                    cat_col = f"Cat_{cat_name}_Net"
                    cat_value = abs(row.get(cat_col, 0))
                    
                    # Filter: Must meet minimum absolute value
                    if cat_value >= min_abs_value:
                        high_ratios.append((cat_name, ratio, cat_value))
            
            # Only flag if there's a significant concentration AND it's a large absolute value
            if high_ratios:
                # Sort by ratio descending, then by absolute value
                high_ratios.sort(key=lambda x: (x[1], x[2]), reverse=True)
                max_ratio_info = high_ratios[0]
                cat_name, ratio, cat_value = max_ratio_info
                
                timestamp = row.get('Week_Start') or row.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                flag = self.create_flag(
                    entity=entity,
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.CATEGORY,
                    severity=Severity.LOW,
                    confidence=0.65,  # Slightly higher confidence
                    metric_name=f"{cat_name}_ratio",
                    metric_value=ratio,
                    threshold=threshold,
                    description=f"High category concentration: {cat_name} ({ratio:.1%})",
                    explanation=f"The {cat_name} category represents {ratio:.1%} of total "
                               f"cash flow this period (${cat_value:,.2f}), indicating unusual concentration.",
                    rule_id="cat_ratio",
                    contributing_factors={
                        'category': cat_name,
                        'ratio': ratio,
                        'absolute_value': cat_value,
                        'all_high_ratios': {name: r for name, r, _ in high_ratios}
                    }
                )
                flags.append(flag)
        
        return flags
    
    def explain(self, flag: AnomalyFlag) -> str:
        """Generate explanation for category anomaly."""
        lines = [
            f"Category Anomaly Detected",
            f"=========================",
            f"Entity: {flag.entity}",
            f"Time: {flag.timestamp}",
            f"Severity: {flag.severity.value.upper()}",
            f"Confidence: {flag.confidence:.1%}",
            f"",
            f"Category: {flag.contributing_factors.get('category', 'Unknown')}",
            f"",
            f"Description: {flag.description}",
            f"",
            flag.explanation
        ]
        
        if flag.decision_path:
            lines.append("")
            lines.append("Decision Path:")
            for step in flag.decision_path:
                lines.append(f"  -> {step}")
        
        return "\n".join(lines)
    
    def get_category_summary(self, entity: str) -> Dict[str, Any]:
        """Get summary of category patterns for an entity."""
        stats = self.category_stats.get(entity, {})
        trees = self.category_trees.get(entity, {})
        
        summary = {
            'entity': entity,
            'categories': {}
        }
        
        for cat_name in set(list(stats.keys()) + list(trees.keys())):
            summary['categories'][cat_name] = {
                'statistics': stats.get(cat_name, {}),
                'has_tree': cat_name in trees,
                'tree_rules': trees[cat_name].rules_to_english() if cat_name in trees else []
            }
        
        return summary

