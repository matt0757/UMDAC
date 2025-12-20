"""
Rule-Based Detection Agent
===========================

Detects anomalies based on business rules including duplicates,
amount ranges, and category-specific constraints.
"""

from typing import Dict, List, Optional, Any, Set
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


class RuleAgent(BaseDetectorAgent):
    """
    Agent for rule-based anomaly detection.
    
    Implements business rules for:
    - Duplicate transaction detection
    - Amount range violations
    - Category-specific rules
    - Unusual transaction patterns
    """
    
    def __init__(self, agent_id: str = None, name: str = "Rule Agent",
                 knowledge_base: KnowledgeBase = None):
        """Initialize the rule agent."""
        super().__init__(agent_id, name, knowledge_base)
        
        self.config = {
            # Amount thresholds (can be entity-specific)
            'large_transaction_threshold': 100000,
            'very_large_threshold': 500000,
            'small_transaction_threshold': 1,
            
            # Duplicate detection
            'duplicate_time_window_hours': 24,
            'duplicate_amount_tolerance': 0.01,  # 1%
            
            # Category rules
            'unusual_category_ratio_threshold': 0.5,  # 50% of typical
            
            # Transaction count thresholds
            'high_transaction_count_threshold': 100,
            'low_transaction_count_threshold': 5
        }
        
        # Entity-specific overrides
        self.entity_rules: Dict[str, Dict[str, Any]] = {}
        
        # Initialize rule graphs
        self._init_rule_graphs()
    
    @property
    def agent_type(self) -> str:
        return "rule"
    
    @property
    def anomaly_types(self) -> List[AnomalyType]:
        return [AnomalyType.RULE_VIOLATION]
    
    def _init_rule_graphs(self) -> None:
        """Initialize business rule graphs."""
        # Large transaction rule graph
        self.rule_graphs['large_transaction'] = self._create_large_transaction_graph()
        
        # Duplicate detection graph
        self.rule_graphs['duplicate'] = self._create_duplicate_graph()
        
        # Category rules graph
        self.rule_graphs['category'] = self._create_category_rules_graph()
        
        # Volume rules graph
        self.rule_graphs['volume'] = self._create_volume_rules_graph()
    
    def _create_large_transaction_graph(self) -> RuleGraph:
        """Create rule graph for large transaction detection."""
        graph = RuleGraph(
            graph_id="rule_large_txn",
            name="Large Transaction Detection",
            description="Flags unusually large transactions"
        )
        
        # Check very large
        very_large_check = RuleNode(
            node_id="very_large_check",
            node_type=NodeType.THRESHOLD,
            name="Very Large Transaction",
            description="Check for very large transaction",
            condition="abs(amount) > very_large_threshold",
            field_name="abs_amount",
            operator=">",
            threshold=ThresholdConfig(base=self.config['very_large_threshold'])
        )
        graph.add_node(very_large_check, is_root=True)
        
        # Flag critical
        flag_critical = RuleNode(
            node_id="flag_critical",
            node_type=NodeType.ACTION,
            name="Flag Critical",
            description="Very large transaction - critical review",
            condition="Critical transaction",
            action="flag_anomaly",
            severity="critical"
        )
        graph.add_node(flag_critical)
        
        # Check large
        large_check = RuleNode(
            node_id="large_check",
            node_type=NodeType.THRESHOLD,
            name="Large Transaction",
            description="Check for large transaction",
            condition="abs(amount) > large_threshold",
            field_name="abs_amount",
            operator=">",
            threshold=ThresholdConfig(base=self.config['large_transaction_threshold'])
        )
        graph.add_node(large_check)
        
        # Flag high
        flag_high = RuleNode(
            node_id="flag_high",
            node_type=NodeType.ACTION,
            name="Flag High",
            description="Large transaction - review required",
            condition="Large transaction",
            action="flag_anomaly",
            severity="high"
        )
        graph.add_node(flag_high)
        
        # Normal
        normal = RuleNode(
            node_id="normal",
            node_type=NodeType.ACTION,
            name="Normal",
            description="Transaction within normal range",
            condition="Normal amount",
            action="normal",
            severity="low"
        )
        graph.add_node(normal)
        
        graph.add_edge("very_large_check", "flag_critical", EdgeType.ESCALATE)
        graph.add_edge("very_large_check", "large_check", EdgeType.ELSE)
        graph.add_edge("large_check", "flag_high", EdgeType.ESCALATE)
        graph.add_edge("large_check", "normal", EdgeType.ELSE)
        
        return graph
    
    def _create_duplicate_graph(self) -> RuleGraph:
        """Create rule graph for duplicate detection."""
        graph = RuleGraph(
            graph_id="rule_duplicate",
            name="Duplicate Detection",
            description="Detects potential duplicate transactions"
        )
        
        # Check same amount
        amount_check = RuleNode(
            node_id="same_amount",
            node_type=NodeType.CONDITION,
            name="Same Amount",
            description="Check if amount matches another recent transaction",
            condition="amount matches within tolerance",
            condition_code="data.get('is_duplicate_amount', False)"
        )
        graph.add_node(amount_check, is_root=True)
        
        # Check same account
        account_check = RuleNode(
            node_id="same_account",
            node_type=NodeType.CONDITION,
            name="Same Account",
            description="Check if same account involved",
            condition="same account",
            condition_code="data.get('is_same_account', False)"
        )
        graph.add_node(account_check)
        
        # Flag duplicate
        flag_dup = RuleNode(
            node_id="flag_duplicate",
            node_type=NodeType.ACTION,
            name="Flag Potential Duplicate",
            description="Potential duplicate transaction",
            condition="Duplicate detected",
            action="flag_anomaly",
            severity="medium"
        )
        graph.add_node(flag_dup)
        
        # Normal
        normal = RuleNode(
            node_id="normal",
            node_type=NodeType.ACTION,
            name="Normal",
            description="Not a duplicate",
            condition="Unique transaction",
            action="normal",
            severity="low"
        )
        graph.add_node(normal)
        
        graph.add_edge("same_amount", "same_account", EdgeType.AND)
        graph.add_edge("same_account", "flag_duplicate", EdgeType.ESCALATE)
        graph.add_edge("same_account", "normal", EdgeType.ELSE)
        graph.add_edge("same_amount", "normal", EdgeType.ELSE)
        
        return graph
    
    def _create_category_rules_graph(self) -> RuleGraph:
        """Create rule graph for category-specific rules."""
        graph = RuleGraph(
            graph_id="rule_category",
            name="Category Rules",
            description="Category-specific business rules"
        )
        
        # Check unusual category ratio
        ratio_check = RuleNode(
            node_id="unusual_ratio",
            node_type=NodeType.THRESHOLD,
            name="Unusual Category Ratio",
            description="Check if category proportion is unusual",
            condition="category_ratio deviates from typical",
            field_name="category_ratio_deviation",
            operator="abs>",
            threshold=ThresholdConfig(base=self.config['unusual_category_ratio_threshold'])
        )
        graph.add_node(ratio_check, is_root=True)
        
        # Flag unusual
        flag_unusual = RuleNode(
            node_id="flag_unusual_category",
            node_type=NodeType.ACTION,
            name="Flag Unusual Category",
            description="Unusual category distribution",
            condition="Unusual category",
            action="flag_anomaly",
            severity="low"
        )
        graph.add_node(flag_unusual)
        
        # Normal
        normal = RuleNode(
            node_id="normal",
            node_type=NodeType.ACTION,
            name="Normal",
            description="Normal category distribution",
            condition="Normal categories",
            action="normal",
            severity="low"
        )
        graph.add_node(normal)
        
        graph.add_edge("unusual_ratio", "flag_unusual_category", EdgeType.ESCALATE)
        graph.add_edge("unusual_ratio", "normal", EdgeType.ELSE)
        
        return graph
    
    def _create_volume_rules_graph(self) -> RuleGraph:
        """Create rule graph for transaction volume rules."""
        graph = RuleGraph(
            graph_id="rule_volume",
            name="Volume Rules",
            description="Transaction volume anomaly rules"
        )
        
        # Check high volume
        high_vol = RuleNode(
            node_id="high_volume",
            node_type=NodeType.THRESHOLD,
            name="High Volume",
            description="Unusually high transaction count",
            condition="transaction_count > high_threshold",
            field_name="Transaction_Count",
            operator=">",
            threshold=ThresholdConfig(base=self.config['high_transaction_count_threshold'])
        )
        graph.add_node(high_vol, is_root=True)
        
        # Flag high volume
        flag_high_vol = RuleNode(
            node_id="flag_high_volume",
            node_type=NodeType.ACTION,
            name="Flag High Volume",
            description="Unusually high transaction volume",
            condition="High volume",
            action="flag_anomaly",
            severity="medium"
        )
        graph.add_node(flag_high_vol)
        
        # Check low volume
        low_vol = RuleNode(
            node_id="low_volume",
            node_type=NodeType.THRESHOLD,
            name="Low Volume",
            description="Unusually low transaction count",
            condition="transaction_count < low_threshold",
            field_name="Transaction_Count",
            operator="<",
            threshold=ThresholdConfig(base=self.config['low_transaction_count_threshold'])
        )
        graph.add_node(low_vol)
        
        # Flag low volume
        flag_low_vol = RuleNode(
            node_id="flag_low_volume",
            node_type=NodeType.ACTION,
            name="Flag Low Volume",
            description="Unusually low transaction volume",
            condition="Low volume",
            action="flag_anomaly",
            severity="low"
        )
        graph.add_node(flag_low_vol)
        
        # Normal
        normal = RuleNode(
            node_id="normal",
            node_type=NodeType.ACTION,
            name="Normal",
            description="Normal transaction volume",
            condition="Normal volume",
            action="normal",
            severity="low"
        )
        graph.add_node(normal)
        
        graph.add_edge("high_volume", "flag_high_volume", EdgeType.ESCALATE)
        graph.add_edge("high_volume", "low_volume", EdgeType.ELSE)
        graph.add_edge("low_volume", "flag_low_volume", EdgeType.ESCALATE)
        graph.add_edge("low_volume", "normal", EdgeType.ELSE)
        
        return graph
    
    def detect(self, data: pd.DataFrame, 
               context: DetectionContext = None) -> List[AnomalyFlag]:
        """Detect rule-based anomalies."""
        flags = []
        modifier = context.threshold_modifier if context else 1.0
        
        if 'Entity' in data.columns:
            entities = data['Entity'].unique()
        else:
            entities = [context.entity if context else 'default']
        
        for entity in entities:
            entity_data = data[data['Entity'] == entity] if 'Entity' in data.columns else data
            
            # Large transaction detection
            large_flags = self._detect_large_transactions(entity_data, entity, modifier)
            flags.extend(large_flags)
            
            # Volume anomaly detection
            volume_flags = self._detect_volume_anomalies(entity_data, entity, modifier)
            flags.extend(volume_flags)
            
            # Category ratio detection
            category_flags = self._detect_category_anomalies(entity_data, entity, modifier)
            flags.extend(category_flags)
            
            # Duplicate detection (if transaction-level data)
            if 'Amount in USD' in entity_data.columns:
                dup_flags = self._detect_duplicates(entity_data, entity)
                flags.extend(dup_flags)
        
        return flags
    
    def _detect_large_transactions(self, data: pd.DataFrame, entity: str,
                                    modifier: float) -> List[AnomalyFlag]:
        """Detect unusually large transactions."""
        flags = []
        
        # Check for aggregate data (Total_Net, Total_Inflow, Total_Outflow)
        for amount_col in ['Total_Net', 'Total_Inflow', 'Outflow_Abs', 'Amount in USD']:
            if amount_col not in data.columns:
                continue
            
            for idx, row in data.iterrows():
                amount = abs(row.get(amount_col, 0))
                
                # Get entity-specific thresholds or use defaults
                thresholds = self.entity_rules.get(entity, self.config)
                very_large = thresholds.get('very_large_threshold', self.config['very_large_threshold']) * modifier
                large = thresholds.get('large_transaction_threshold', self.config['large_transaction_threshold']) * modifier
                
                if amount > very_large:
                    severity = Severity.CRITICAL
                elif amount > large:
                    severity = Severity.HIGH
                else:
                    continue
                
                timestamp = row.get('Week_Start') or row.get('Pstng Date') or row.get('timestamp')
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except:
                        timestamp = datetime.now()
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                # Extract week info for context
                week_num = row.get('Week_Num') or row.get('week_num') or idx
                week_start = row.get('Week_Start')
                
                # Build description with week context
                week_context = f" (Week {week_num})" if week_num else ""
                
                flag = self.create_flag(
                    entity=entity,
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.RULE_VIOLATION,
                    severity=severity,
                    confidence=0.95,
                    metric_name=f"large_{amount_col}",
                    metric_value=amount,
                    threshold=very_large if severity == Severity.CRITICAL else large,
                    description=f"Large transaction detected{week_context}: ${amount:,.2f}",
                    explanation=f"The {amount_col} of ${amount:,.2f} exceeds the threshold of "
                               f"${very_large if severity == Severity.CRITICAL else large:,.2f}. "
                               f"This transaction requires review.",
                    rule_id="rule_large_txn",
                    decision_path=[
                        f"Check amount: ${amount:,.2f}",
                        f"{'Very large' if severity == Severity.CRITICAL else 'Large'} threshold exceeded"
                    ],
                    contributing_factors={
                        'amount_column': amount_col,
                        'amount': amount,
                        'threshold': very_large if severity == Severity.CRITICAL else large,
                        'week_num': week_num,
                        'week_start': str(week_start) if week_start is not None else None
                    }
                )
                flags.append(flag)
        
        return flags
    
    def _detect_volume_anomalies(self, data: pd.DataFrame, entity: str,
                                  modifier: float) -> List[AnomalyFlag]:
        """Detect transaction volume anomalies."""
        flags = []
        
        if 'Transaction_Count' not in data.columns:
            return flags
        
        high_threshold = self.config['high_transaction_count_threshold'] / modifier
        low_threshold = self.config['low_transaction_count_threshold'] * modifier
        
        for idx, row in data.iterrows():
            count = row.get('Transaction_Count', 0)
            
            if count > high_threshold or count < low_threshold:
                is_high = count > high_threshold
                
                timestamp = row.get('Week_Start') or row.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                flag = self.create_flag(
                    entity=entity,
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.RULE_VIOLATION,
                    severity=Severity.MEDIUM if is_high else Severity.LOW,
                    confidence=0.8,
                    metric_name="transaction_count",
                    metric_value=count,
                    threshold=high_threshold if is_high else low_threshold,
                    description=f"{'High' if is_high else 'Low'} transaction volume: {count}",
                    explanation=f"Transaction count of {count} is {'above' if is_high else 'below'} "
                               f"the expected range ({low_threshold:.0f} - {high_threshold:.0f}). "
                               f"This may indicate unusual activity.",
                    rule_id="rule_volume",
                    contributing_factors={
                        'transaction_count': count,
                        'high_threshold': high_threshold,
                        'low_threshold': low_threshold
                    }
                )
                flags.append(flag)
        
        return flags
    
    def _detect_category_anomalies(self, data: pd.DataFrame, entity: str,
                                    modifier: float) -> List[AnomalyFlag]:
        """Detect unusual category distributions."""
        flags = []
        
        # Find category columns
        category_cols = [c for c in data.columns if c.startswith('Cat_') and c.endswith('_Net')]
        if not category_cols or 'Total_Net' not in data.columns:
            return flags
        
        # Calculate expected category ratios
        for idx, row in data.iterrows():
            total = abs(row.get('Total_Net', 0))
            if total == 0:
                continue
            
            unusual_categories = []
            for cat_col in category_cols:
                cat_value = abs(row.get(cat_col, 0))
                ratio = cat_value / total if total > 0 else 0
                
                # Check if ratio is unusually high (>50% by default)
                if ratio > self.config['unusual_category_ratio_threshold'] / modifier:
                    category_name = cat_col.replace('Cat_', '').replace('_Net', '')
                    unusual_categories.append((category_name, ratio, cat_value))
            
            if unusual_categories:
                max_cat = max(unusual_categories, key=lambda x: x[1])
                
                timestamp = row.get('Week_Start') or row.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                flag = self.create_flag(
                    entity=entity,
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.RULE_VIOLATION,
                    severity=Severity.LOW,
                    confidence=0.7,
                    metric_name=f"category_ratio_{max_cat[0]}",
                    metric_value=max_cat[1],
                    threshold=self.config['unusual_category_ratio_threshold'] / modifier,
                    description=f"Unusual category concentration: {max_cat[0]} ({max_cat[1]:.1%})",
                    explanation=f"The {max_cat[0]} category represents {max_cat[1]:.1%} of total "
                               f"cash flow (${max_cat[2]:,.2f} of ${total:,.2f}). "
                               f"This is an unusually high concentration.",
                    rule_id="rule_category",
                    contributing_factors={
                        'category': max_cat[0],
                        'ratio': max_cat[1],
                        'category_value': max_cat[2],
                        'total_value': total
                    }
                )
                flags.append(flag)
        
        return flags
    
    def _detect_duplicates(self, data: pd.DataFrame, entity: str) -> List[AnomalyFlag]:
        """Detect potential duplicate transactions."""
        flags = []
        
        if 'Amount in USD' not in data.columns:
            return flags
        
        # Group by similar amounts
        tolerance = self.config['duplicate_amount_tolerance']
        amounts = data['Amount in USD'].values
        seen_amounts: Set[float] = set()
        
        for idx, row in data.iterrows():
            amount = row.get('Amount in USD', 0)
            
            # Check if similar amount seen before
            is_duplicate = False
            for seen in seen_amounts:
                if abs(amount - seen) <= abs(seen) * tolerance:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                timestamp = row.get('Pstng Date') or row.get('timestamp')
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except:
                        timestamp = datetime.now()
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                flag = self.create_flag(
                    entity=entity,
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.RULE_VIOLATION,
                    severity=Severity.MEDIUM,
                    confidence=0.6,  # Lower confidence for duplicates
                    metric_name="duplicate_amount",
                    metric_value=amount,
                    threshold=tolerance,
                    description=f"Potential duplicate: ${amount:,.2f}",
                    explanation=f"A transaction with amount ${amount:,.2f} appears similar to "
                               f"another recent transaction. Please verify this is not a duplicate.",
                    rule_id="rule_duplicate",
                    contributing_factors={
                        'amount': amount,
                        'tolerance': tolerance
                    }
                )
                flags.append(flag)
            
            seen_amounts.add(amount)
        
        return flags
    
    def explain(self, flag: AnomalyFlag) -> str:
        """Generate explanation for rule violation."""
        lines = [
            f"Business Rule Violation",
            f"=======================",
            f"Entity: {flag.entity}",
            f"Time: {flag.timestamp}",
            f"Severity: {flag.severity.value.upper()}",
            f"",
            f"Rule: {flag.rule_id}",
            f"Description: {flag.description}",
            f"",
            f"Details: {flag.explanation}",
            f"",
            "Decision Path:"
        ]
        
        for step in flag.decision_path:
            lines.append(f"  -> {step}")
        
        return "\n".join(lines)
    
    def add_entity_rule(self, entity: str, rule_name: str, value: Any) -> None:
        """Add or update an entity-specific rule."""
        if entity not in self.entity_rules:
            self.entity_rules[entity] = {}
        self.entity_rules[entity][rule_name] = value
    
    def export_rules_to_english(self) -> List[str]:
        """Export all rules as English descriptions."""
        rules = []
        for graph_id, graph in self.rule_graphs.items():
            rules.append(f"\n=== {graph.name} ===")
            rules.extend(graph.to_english_rules())
        return rules

