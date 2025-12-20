"""
Rule Migration Script
=====================

Migrates existing anomaly detection rules into the new rule graph format.
Creates initial rule graphs for each agent type.
"""

import json
from pathlib import Path
from datetime import datetime

from core.rule_graph import (
    RuleGraph, RuleNode, NodeType, EdgeType, ThresholdConfig
)
from core.knowledge_base import KnowledgeBase


def create_initial_rule_graphs() -> dict:
    """
    Create initial rule graphs for all agent types.
    
    Returns:
        Dictionary of graph_id -> RuleGraph
    """
    graphs = {}
    
    # Statistical Rules
    graphs['stat_zscore'] = create_zscore_rules()
    graphs['stat_level_shift'] = create_level_shift_rules()
    
    # Pattern Rules
    graphs['pattern_seasonal'] = create_seasonal_pattern_rules()
    
    # Business Rules
    graphs['rule_large_txn'] = create_large_transaction_rules()
    graphs['rule_duplicate'] = create_duplicate_rules()
    graphs['rule_volume'] = create_volume_rules()
    
    # Temporal Rules
    graphs['temp_wow'] = create_wow_rules()
    graphs['temp_mom'] = create_mom_rules()
    
    # Category Rules
    graphs['cat_deviation'] = create_category_deviation_rules()
    
    return graphs


def create_zscore_rules() -> RuleGraph:
    """Create Z-score anomaly detection rules."""
    graph = RuleGraph(
        graph_id="stat_zscore",
        name="Z-Score Anomaly Detection",
        description="Detects statistical outliers based on Z-score deviation from mean"
    )
    
    # Root node
    root = RuleNode(
        node_id="zscore_check",
        node_type=NodeType.THRESHOLD,
        name="Z-Score Threshold",
        description="Check if absolute Z-score exceeds threshold",
        condition="abs(zscore) > threshold",
        condition_code="abs(data.get('zscore', 0)) > threshold",
        field_name="zscore",
        operator="abs>",
        threshold=ThresholdConfig(base=3.0, min_value=1.5, max_value=5.0)
    )
    graph.add_node(root, is_root=True)
    
    # Critical check
    critical = RuleNode(
        node_id="critical_check",
        node_type=NodeType.THRESHOLD,
        name="Critical Threshold",
        description="Check for critical anomaly (Z > 5)",
        condition="abs(zscore) > 5.0",
        field_name="zscore",
        operator="abs>",
        threshold=ThresholdConfig(base=5.0)
    )
    graph.add_node(critical)
    
    # High check
    high = RuleNode(
        node_id="high_check",
        node_type=NodeType.THRESHOLD,
        name="High Threshold",
        description="Check for high anomaly (Z > 4)",
        condition="abs(zscore) > 4.0",
        field_name="zscore",
        operator="abs>",
        threshold=ThresholdConfig(base=4.0)
    )
    graph.add_node(high)
    
    # Action nodes
    flag_critical = RuleNode(
        node_id="flag_critical",
        node_type=NodeType.ACTION,
        name="Flag Critical",
        description="Critical statistical anomaly",
        condition="Critical",
        action="flag_anomaly",
        severity="critical"
    )
    graph.add_node(flag_critical)
    
    flag_high = RuleNode(
        node_id="flag_high",
        node_type=NodeType.ACTION,
        name="Flag High",
        description="High statistical anomaly",
        condition="High",
        action="flag_anomaly",
        severity="high"
    )
    graph.add_node(flag_high)
    
    flag_medium = RuleNode(
        node_id="flag_medium",
        node_type=NodeType.ACTION,
        name="Flag Medium",
        description="Medium statistical anomaly",
        condition="Medium",
        action="flag_anomaly",
        severity="medium"
    )
    graph.add_node(flag_medium)
    
    normal = RuleNode(
        node_id="normal",
        node_type=NodeType.ACTION,
        name="Normal",
        description="Normal value",
        condition="Normal",
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


def create_level_shift_rules() -> RuleGraph:
    """Create level shift detection rules."""
    graph = RuleGraph(
        graph_id="stat_level_shift",
        name="Level Shift Detection",
        description="Detects sudden changes in time series level"
    )
    
    root = RuleNode(
        node_id="level_check",
        node_type=NodeType.THRESHOLD,
        name="Level Shift Check",
        description="Check for significant level change",
        condition="level_shift_score > 2.5",
        field_name="level_shift_score",
        operator="abs>",
        threshold=ThresholdConfig(base=2.5, min_value=1.5, max_value=4.0)
    )
    graph.add_node(root, is_root=True)
    
    flag_shift = RuleNode(
        node_id="flag_shift",
        node_type=NodeType.ACTION,
        name="Flag Level Shift",
        description="Level shift detected",
        condition="Shift detected",
        action="flag_anomaly",
        severity="medium"
    )
    graph.add_node(flag_shift)
    
    normal = RuleNode(
        node_id="normal",
        node_type=NodeType.ACTION,
        name="Normal",
        description="Stable level",
        condition="Stable",
        action="normal",
        severity="low"
    )
    graph.add_node(normal)
    
    graph.add_edge("level_check", "flag_shift", EdgeType.ESCALATE)
    graph.add_edge("level_check", "normal", EdgeType.ELSE)
    
    return graph


def create_seasonal_pattern_rules() -> RuleGraph:
    """Create seasonal pattern rules."""
    graph = RuleGraph(
        graph_id="pattern_seasonal",
        name="Seasonal Pattern Detection",
        description="Detects violations of seasonal patterns"
    )
    
    root = RuleNode(
        node_id="pattern_check",
        node_type=NodeType.PATTERN,
        name="Pattern Deviation",
        description="Check deviation from seasonal pattern",
        condition="pattern_deviation > 2.0",
        field_name="pattern_deviation",
        operator="abs>",
        threshold=ThresholdConfig(base=2.0)
    )
    graph.add_node(root, is_root=True)
    
    flag = RuleNode(
        node_id="flag_pattern",
        node_type=NodeType.ACTION,
        name="Flag Pattern Violation",
        description="Pattern violation detected",
        condition="Violation",
        action="flag_anomaly",
        severity="medium"
    )
    graph.add_node(flag)
    
    normal = RuleNode(
        node_id="normal",
        node_type=NodeType.ACTION,
        name="Normal",
        description="Expected pattern",
        condition="Normal",
        action="normal",
        severity="low"
    )
    graph.add_node(normal)
    
    graph.add_edge("pattern_check", "flag_pattern", EdgeType.ESCALATE)
    graph.add_edge("pattern_check", "normal", EdgeType.ELSE)
    
    return graph


def create_large_transaction_rules() -> RuleGraph:
    """Create large transaction detection rules."""
    graph = RuleGraph(
        graph_id="rule_large_txn",
        name="Large Transaction Detection",
        description="Flags unusually large transactions"
    )
    
    very_large = RuleNode(
        node_id="very_large_check",
        node_type=NodeType.THRESHOLD,
        name="Very Large Transaction",
        description="Check for very large amount",
        condition="abs(amount) > 500000",
        field_name="abs_amount",
        operator=">",
        threshold=ThresholdConfig(base=500000)
    )
    graph.add_node(very_large, is_root=True)
    
    large = RuleNode(
        node_id="large_check",
        node_type=NodeType.THRESHOLD,
        name="Large Transaction",
        description="Check for large amount",
        condition="abs(amount) > 100000",
        field_name="abs_amount",
        operator=">",
        threshold=ThresholdConfig(base=100000)
    )
    graph.add_node(large)
    
    flag_critical = RuleNode(
        node_id="flag_critical",
        node_type=NodeType.ACTION,
        name="Flag Critical",
        description="Very large transaction",
        condition="Very large",
        action="flag_anomaly",
        severity="critical"
    )
    graph.add_node(flag_critical)
    
    flag_high = RuleNode(
        node_id="flag_high",
        node_type=NodeType.ACTION,
        name="Flag High",
        description="Large transaction",
        condition="Large",
        action="flag_anomaly",
        severity="high"
    )
    graph.add_node(flag_high)
    
    normal = RuleNode(
        node_id="normal",
        node_type=NodeType.ACTION,
        name="Normal",
        description="Normal amount",
        condition="Normal",
        action="normal",
        severity="low"
    )
    graph.add_node(normal)
    
    graph.add_edge("very_large_check", "flag_critical", EdgeType.ESCALATE)
    graph.add_edge("very_large_check", "large_check", EdgeType.ELSE)
    graph.add_edge("large_check", "flag_high", EdgeType.ESCALATE)
    graph.add_edge("large_check", "normal", EdgeType.ELSE)
    
    return graph


def create_duplicate_rules() -> RuleGraph:
    """Create duplicate detection rules."""
    graph = RuleGraph(
        graph_id="rule_duplicate",
        name="Duplicate Detection",
        description="Detects potential duplicate transactions"
    )
    
    amount_check = RuleNode(
        node_id="amount_match",
        node_type=NodeType.CONDITION,
        name="Amount Match",
        description="Check for matching amounts",
        condition="is_duplicate_amount",
        condition_code="data.get('is_duplicate_amount', False)"
    )
    graph.add_node(amount_check, is_root=True)
    
    time_check = RuleNode(
        node_id="time_match",
        node_type=NodeType.CONDITION,
        name="Time Window",
        description="Check if within time window",
        condition="within_time_window",
        condition_code="data.get('within_time_window', False)"
    )
    graph.add_node(time_check)
    
    flag = RuleNode(
        node_id="flag_duplicate",
        node_type=NodeType.ACTION,
        name="Flag Duplicate",
        description="Potential duplicate",
        condition="Duplicate",
        action="flag_anomaly",
        severity="medium"
    )
    graph.add_node(flag)
    
    normal = RuleNode(
        node_id="normal",
        node_type=NodeType.ACTION,
        name="Normal",
        description="Unique transaction",
        condition="Unique",
        action="normal",
        severity="low"
    )
    graph.add_node(normal)
    
    graph.add_edge("amount_match", "time_match", EdgeType.AND)
    graph.add_edge("time_match", "flag_duplicate", EdgeType.ESCALATE)
    graph.add_edge("time_match", "normal", EdgeType.ELSE)
    graph.add_edge("amount_match", "normal", EdgeType.ELSE)
    
    return graph


def create_volume_rules() -> RuleGraph:
    """Create transaction volume rules."""
    graph = RuleGraph(
        graph_id="rule_volume",
        name="Volume Anomaly Detection",
        description="Detects unusual transaction volumes"
    )
    
    high_vol = RuleNode(
        node_id="high_volume",
        node_type=NodeType.THRESHOLD,
        name="High Volume",
        description="Unusually high transaction count",
        condition="count > 100",
        field_name="Transaction_Count",
        operator=">",
        threshold=ThresholdConfig(base=100)
    )
    graph.add_node(high_vol, is_root=True)
    
    low_vol = RuleNode(
        node_id="low_volume",
        node_type=NodeType.THRESHOLD,
        name="Low Volume",
        description="Unusually low transaction count",
        condition="count < 5",
        field_name="Transaction_Count",
        operator="<",
        threshold=ThresholdConfig(base=5)
    )
    graph.add_node(low_vol)
    
    flag_high = RuleNode(
        node_id="flag_high_vol",
        node_type=NodeType.ACTION,
        name="Flag High Volume",
        description="High volume anomaly",
        condition="High",
        action="flag_anomaly",
        severity="medium"
    )
    graph.add_node(flag_high)
    
    flag_low = RuleNode(
        node_id="flag_low_vol",
        node_type=NodeType.ACTION,
        name="Flag Low Volume",
        description="Low volume anomaly",
        condition="Low",
        action="flag_anomaly",
        severity="low"
    )
    graph.add_node(flag_low)
    
    normal = RuleNode(
        node_id="normal",
        node_type=NodeType.ACTION,
        name="Normal",
        description="Normal volume",
        condition="Normal",
        action="normal",
        severity="low"
    )
    graph.add_node(normal)
    
    graph.add_edge("high_volume", "flag_high_vol", EdgeType.ESCALATE)
    graph.add_edge("high_volume", "low_volume", EdgeType.ELSE)
    graph.add_edge("low_volume", "flag_low_vol", EdgeType.ESCALATE)
    graph.add_edge("low_volume", "normal", EdgeType.ELSE)
    
    return graph


def create_wow_rules() -> RuleGraph:
    """Create week-over-week comparison rules."""
    graph = RuleGraph(
        graph_id="temp_wow",
        name="Week-over-Week Change Detection",
        description="Detects significant changes from previous week"
    )
    
    wow_check = RuleNode(
        node_id="wow_change",
        node_type=NodeType.THRESHOLD,
        name="WoW Change",
        description="Check week-over-week percentage change",
        condition="abs(wow_pct) > 0.5",
        field_name="wow_pct_change",
        operator="abs>",
        threshold=ThresholdConfig(base=0.5, min_value=0.2, max_value=2.0)
    )
    graph.add_node(wow_check, is_root=True)
    
    large_change = RuleNode(
        node_id="large_wow",
        node_type=NodeType.THRESHOLD,
        name="Large WoW Change",
        description="Check for very large change",
        condition="abs(wow_pct) > 1.0",
        field_name="wow_pct_change",
        operator="abs>",
        threshold=ThresholdConfig(base=1.0)
    )
    graph.add_node(large_change)
    
    flag_high = RuleNode(
        node_id="flag_high",
        node_type=NodeType.ACTION,
        name="Flag High WoW",
        description="Large week-over-week change",
        condition="Large change",
        action="flag_anomaly",
        severity="high"
    )
    graph.add_node(flag_high)
    
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
    
    normal = RuleNode(
        node_id="normal",
        node_type=NodeType.ACTION,
        name="Normal",
        description="Normal variation",
        condition="Normal",
        action="normal",
        severity="low"
    )
    graph.add_node(normal)
    
    graph.add_edge("wow_change", "large_wow", EdgeType.CHAIN)
    graph.add_edge("large_wow", "flag_high", EdgeType.ESCALATE)
    graph.add_edge("large_wow", "flag_medium", EdgeType.ELSE)
    graph.add_edge("wow_change", "normal", EdgeType.ELSE)
    
    return graph


def create_mom_rules() -> RuleGraph:
    """Create month-over-month comparison rules."""
    graph = RuleGraph(
        graph_id="temp_mom",
        name="Month-over-Month Change Detection",
        description="Detects significant changes from same week last month"
    )
    
    mom_check = RuleNode(
        node_id="mom_change",
        node_type=NodeType.THRESHOLD,
        name="MoM Change",
        description="Check month-over-month change",
        condition="abs(mom_pct) > 0.3",
        field_name="mom_pct_change",
        operator="abs>",
        threshold=ThresholdConfig(base=0.3, min_value=0.1, max_value=1.0)
    )
    graph.add_node(mom_check, is_root=True)
    
    flag = RuleNode(
        node_id="flag_mom",
        node_type=NodeType.ACTION,
        name="Flag MoM Anomaly",
        description="Significant month-over-month change",
        condition="MoM anomaly",
        action="flag_anomaly",
        severity="medium"
    )
    graph.add_node(flag)
    
    normal = RuleNode(
        node_id="normal",
        node_type=NodeType.ACTION,
        name="Normal",
        description="Normal variation",
        condition="Normal",
        action="normal",
        severity="low"
    )
    graph.add_node(normal)
    
    graph.add_edge("mom_change", "flag_mom", EdgeType.ESCALATE)
    graph.add_edge("mom_change", "normal", EdgeType.ELSE)
    
    return graph


def create_category_deviation_rules() -> RuleGraph:
    """Create category deviation detection rules."""
    graph = RuleGraph(
        graph_id="cat_deviation",
        name="Category Deviation Detection",
        description="Detects unusual deviations in category values"
    )
    
    dev_check = RuleNode(
        node_id="cat_deviation",
        node_type=NodeType.THRESHOLD,
        name="Category Deviation",
        description="Check category value deviation",
        condition="abs(category_zscore) > 2.5",
        field_name="category_zscore",
        operator="abs>",
        threshold=ThresholdConfig(base=2.5, min_value=1.5, max_value=4.0)
    )
    graph.add_node(dev_check, is_root=True)
    
    high_dev = RuleNode(
        node_id="high_deviation",
        node_type=NodeType.THRESHOLD,
        name="High Deviation",
        description="Check for severe deviation",
        condition="abs(category_zscore) > 4.0",
        field_name="category_zscore",
        operator="abs>",
        threshold=ThresholdConfig(base=4.0)
    )
    graph.add_node(high_dev)
    
    flag_high = RuleNode(
        node_id="flag_high",
        node_type=NodeType.ACTION,
        name="Flag High",
        description="Severe category deviation",
        condition="High deviation",
        action="flag_anomaly",
        severity="high"
    )
    graph.add_node(flag_high)
    
    flag_medium = RuleNode(
        node_id="flag_medium",
        node_type=NodeType.ACTION,
        name="Flag Medium",
        description="Notable category deviation",
        condition="Medium deviation",
        action="flag_anomaly",
        severity="medium"
    )
    graph.add_node(flag_medium)
    
    normal = RuleNode(
        node_id="normal",
        node_type=NodeType.ACTION,
        name="Normal",
        description="Normal category value",
        condition="Normal",
        action="normal",
        severity="low"
    )
    graph.add_node(normal)
    
    graph.add_edge("cat_deviation", "high_deviation", EdgeType.CHAIN)
    graph.add_edge("high_deviation", "flag_high", EdgeType.ESCALATE)
    graph.add_edge("high_deviation", "flag_medium", EdgeType.ELSE)
    graph.add_edge("cat_deviation", "normal", EdgeType.ELSE)
    
    return graph


def migrate_rules_to_knowledge_base(kb: KnowledgeBase = None):
    """
    Migrate all rules to the knowledge base.
    
    Args:
        kb: Knowledge base instance (creates new if None)
    """
    if kb is None:
        kb = KnowledgeBase(use_json_fallback=True)
    
    graphs = create_initial_rule_graphs()
    
    for graph_id, graph in graphs.items():
        # Determine agent type from graph ID
        if graph_id.startswith('stat_'):
            agent_type = 'statistical'
        elif graph_id.startswith('pattern_'):
            agent_type = 'pattern'
        elif graph_id.startswith('rule_'):
            agent_type = 'rule'
        elif graph_id.startswith('temp_'):
            agent_type = 'temporal'
        elif graph_id.startswith('cat_'):
            agent_type = 'category'
        else:
            agent_type = 'general'
        
        kb.save_rule_graph(graph, agent_type=agent_type)
        print(f"Migrated: {graph_id} ({agent_type})")
    
    print(f"\nMigration complete. {len(graphs)} rule graphs created.")
    return kb


def export_rules_to_json(output_dir: str = None):
    """
    Export all rules to JSON files.
    
    Args:
        output_dir: Output directory (defaults to rules/)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'rules'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graphs = create_initial_rule_graphs()
    
    for graph_id, graph in graphs.items():
        filepath = output_dir / f"{graph_id}.json"
        graph.save(str(filepath))
        print(f"Exported: {filepath}")
    
    print(f"\nExport complete. {len(graphs)} rule files created in {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        export_rules_to_json()
    else:
        migrate_rules_to_knowledge_base()

