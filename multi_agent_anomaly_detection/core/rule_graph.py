"""
Rule Graph Implementation
=========================

Defines the RuleGraph class for representing anomaly detection rules
as directed acyclic graphs (DAGs) with interpretable node conditions
and edge transitions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import json
import uuid
from enum import Enum


class EdgeType(Enum):
    """Types of edges in the rule graph."""
    AND = "AND"           # Both conditions must be met
    OR = "OR"             # Either condition triggers
    ESCALATE = "ESCALATE" # Move to next severity level
    ELSE = "ELSE"         # Fallback path
    CHAIN = "CHAIN"       # Sequential check


class NodeType(Enum):
    """Types of nodes in the rule graph."""
    CONDITION = "condition"      # Evaluates a condition
    AGGREGATOR = "aggregator"    # Combines multiple inputs
    ACTION = "action"            # Terminal node with action
    THRESHOLD = "threshold"      # Numeric threshold check
    PATTERN = "pattern"          # Pattern matching
    TEMPORAL = "temporal"        # Time-based condition


@dataclass
class NodeMetadata:
    """Metadata for tracking node performance and history."""
    hits_last_30d: int = 0
    false_positive_rate: float = 0.0
    true_positive_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    activation_history: List[datetime] = field(default_factory=list)
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hits_last_30d': self.hits_last_30d,
            'false_positive_rate': self.false_positive_rate,
            'true_positive_rate': self.true_positive_rate,
            'last_updated': self.last_updated.isoformat(),
            'activation_history': [dt.isoformat() for dt in self.activation_history[-100:]],
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeMetadata':
        return cls(
            hits_last_30d=data.get('hits_last_30d', 0),
            false_positive_rate=data.get('false_positive_rate', 0.0),
            true_positive_rate=data.get('true_positive_rate', 0.0),
            last_updated=datetime.fromisoformat(data['last_updated']) if data.get('last_updated') else datetime.now(),
            activation_history=[datetime.fromisoformat(dt) for dt in data.get('activation_history', [])],
            version=data.get('version', 1)
        )
    
    def record_activation(self, is_true_positive: Optional[bool] = None):
        """Record a node activation."""
        now = datetime.now()
        self.activation_history.append(now)
        self.last_updated = now
        
        # Update 30-day count
        cutoff = datetime.now().timestamp() - (30 * 24 * 60 * 60)
        self.activation_history = [
            dt for dt in self.activation_history 
            if dt.timestamp() > cutoff
        ]
        self.hits_last_30d = len(self.activation_history)


@dataclass
class RuleEdge:
    """Edge connecting two rule nodes."""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    condition: Optional[str] = None  # Additional condition for edge traversal
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'edge_id': self.edge_id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'edge_type': self.edge_type.value,
            'weight': self.weight,
            'condition': self.condition
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuleEdge':
        return cls(
            edge_id=data['edge_id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            edge_type=EdgeType(data['edge_type']),
            weight=data.get('weight', 1.0),
            condition=data.get('condition')
        )


@dataclass
class ThresholdConfig:
    """Configuration for threshold-based conditions."""
    base: float
    adjusted: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    adaptation_rate: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'base': self.base,
            'adjusted': self.adjusted,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'adaptation_rate': self.adaptation_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThresholdConfig':
        return cls(
            base=data['base'],
            adjusted=data.get('adjusted'),
            min_value=data.get('min_value'),
            max_value=data.get('max_value'),
            adaptation_rate=data.get('adaptation_rate', 0.1)
        )
    
    @property
    def effective_value(self) -> float:
        """Get the effective threshold value."""
        return self.adjusted if self.adjusted is not None else self.base
    
    def adjust(self, modifier: float) -> float:
        """Adjust threshold by modifier and return new value."""
        new_value = self.base * modifier
        if self.min_value is not None:
            new_value = max(new_value, self.min_value)
        if self.max_value is not None:
            new_value = min(new_value, self.max_value)
        self.adjusted = new_value
        return new_value


@dataclass
class RuleNode:
    """A node in the rule graph representing a condition or action."""
    node_id: str
    node_type: NodeType
    name: str
    description: str
    
    # Condition specification
    condition: str  # Human-readable condition string
    condition_code: Optional[str] = None  # Python expression for evaluation
    
    # Threshold configuration
    threshold: Optional[ThresholdConfig] = None
    
    # Operator for condition
    operator: str = ">"  # >, <, >=, <=, ==, !=, in, not_in
    
    # Field to check
    field_name: Optional[str] = None
    
    # Action for terminal nodes
    action: Optional[str] = None  # "flag_anomaly", "escalate", "review", "normal"
    severity: Optional[str] = None  # "low", "medium", "high", "critical"
    
    # Metadata
    metadata: NodeMetadata = field(default_factory=NodeMetadata)
    
    # Child edges
    children: List[RuleEdge] = field(default_factory=list)
    
    # Entity-specific overrides
    entity_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'name': self.name,
            'description': self.description,
            'condition': self.condition,
            'condition_code': self.condition_code,
            'threshold': self.threshold.to_dict() if self.threshold else None,
            'operator': self.operator,
            'field_name': self.field_name,
            'action': self.action,
            'severity': self.severity,
            'metadata': self.metadata.to_dict(),
            'children': [c.to_dict() for c in self.children],
            'entity_overrides': self.entity_overrides
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuleNode':
        return cls(
            node_id=data['node_id'],
            node_type=NodeType(data['node_type']),
            name=data['name'],
            description=data['description'],
            condition=data['condition'],
            condition_code=data.get('condition_code'),
            threshold=ThresholdConfig.from_dict(data['threshold']) if data.get('threshold') else None,
            operator=data.get('operator', '>'),
            field_name=data.get('field_name'),
            action=data.get('action'),
            severity=data.get('severity'),
            metadata=NodeMetadata.from_dict(data['metadata']) if data.get('metadata') else NodeMetadata(),
            children=[RuleEdge.from_dict(c) for c in data.get('children', [])],
            entity_overrides=data.get('entity_overrides', {})
        )
    
    def get_threshold_for_entity(self, entity: str) -> float:
        """Get the effective threshold for a specific entity."""
        if entity in self.entity_overrides and 'threshold' in self.entity_overrides[entity]:
            return self.entity_overrides[entity]['threshold']
        return self.threshold.effective_value if self.threshold else 0.0
    
    def evaluate(self, data: Dict[str, Any], entity: str = None, context_modifier: float = 1.0) -> bool:
        """Evaluate the node condition against data."""
        if self.condition_code:
            # Use custom Python expression
            try:
                # Create safe evaluation context
                eval_context = {
                    'abs': abs, 'min': min, 'max': max,
                    'data': data, 'threshold': self.get_threshold_for_entity(entity or '') * context_modifier
                }
                eval_context.update(data)
                return eval(self.condition_code, {"__builtins__": {}}, eval_context)
            except Exception:
                return False
        
        if self.field_name is None:
            return False
        
        value = data.get(self.field_name)
        if value is None:
            return False
        
        threshold = self.get_threshold_for_entity(entity or '') * context_modifier if self.threshold else 0.0
        
        # Evaluate based on operator
        operators = {
            '>': lambda v, t: v > t,
            '<': lambda v, t: v < t,
            '>=': lambda v, t: v >= t,
            '<=': lambda v, t: v <= t,
            '==': lambda v, t: v == t,
            '!=': lambda v, t: v != t,
            'abs>': lambda v, t: abs(v) > t,
            'abs>=': lambda v, t: abs(v) >= t,
        }
        
        op_func = operators.get(self.operator)
        if op_func:
            return op_func(value, threshold)
        
        return False
    
    def to_english(self) -> str:
        """Convert the node condition to human-readable English."""
        if self.action:
            severity_text = f" ({self.severity})" if self.severity else ""
            return f"-> Action: {self.action}{severity_text}"
        
        if self.threshold:
            threshold_val = self.threshold.effective_value
            field = self.field_name or "value"
            op_text = {
                '>': 'greater than', '<': 'less than',
                '>=': 'at least', '<=': 'at most',
                '==': 'equal to', '!=': 'not equal to',
                'abs>': 'absolute value greater than',
                'abs>=': 'absolute value at least'
            }.get(self.operator, self.operator)
            return f"If {field} is {op_text} {threshold_val:.2f}"
        
        return self.condition


class RuleGraph:
    """
    Directed Acyclic Graph (DAG) representing anomaly detection rules.
    
    Provides interpretable rule representation with support for:
    - Multiple condition types (threshold, pattern, temporal)
    - Edge types (AND, OR, ESCALATE, ELSE)
    - Entity-specific overrides
    - Performance tracking and metadata
    - JSON serialization
    """
    
    def __init__(self, graph_id: str = None, name: str = "", description: str = ""):
        self.graph_id = graph_id or str(uuid.uuid4())[:8]
        self.name = name
        self.description = description
        self.nodes: Dict[str, RuleNode] = {}
        self.root_node_id: Optional[str] = None
        self.version: int = 1
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()
        self.author: str = "system"
        self.tags: List[str] = []
    
    def add_node(self, node: RuleNode, is_root: bool = False) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
        if is_root:
            self.root_node_id = node.node_id
        self.updated_at = datetime.now()
    
    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, 
                 weight: float = 1.0, condition: str = None) -> None:
        """Add an edge between two nodes."""
        if source_id not in self.nodes:
            raise ValueError(f"Source node {source_id} not found")
        if target_id not in self.nodes:
            raise ValueError(f"Target node {target_id} not found")
        
        edge = RuleEdge(
            edge_id=f"{source_id}_{target_id}",
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            condition=condition
        )
        self.nodes[source_id].children.append(edge)
        self.updated_at = datetime.now()
    
    def get_node(self, node_id: str) -> Optional[RuleNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def evaluate(self, data: Dict[str, Any], entity: str = None, 
                 context_modifier: float = 1.0) -> List[Dict[str, Any]]:
        """
        Evaluate the rule graph against data.
        
        Returns list of triggered actions with decision paths.
        """
        if not self.root_node_id:
            return []
        
        results = []
        self._traverse(self.root_node_id, data, entity, context_modifier, [], results)
        return results
    
    def _traverse(self, node_id: str, data: Dict[str, Any], entity: str,
                  context_modifier: float, path: List[str], results: List[Dict[str, Any]]) -> bool:
        """Recursively traverse the graph."""
        node = self.nodes.get(node_id)
        if not node:
            return False
        
        current_path = path + [f"{node.name}: {node.to_english()}"]
        
        # Evaluate current node
        triggered = node.evaluate(data, entity, context_modifier)
        
        if triggered:
            # Record activation
            node.metadata.record_activation()
            
            # Check if this is a terminal action node
            if node.action:
                results.append({
                    'node_id': node.node_id,
                    'action': node.action,
                    'severity': node.severity,
                    'decision_path': current_path,
                    'confidence': 1.0 - node.metadata.false_positive_rate
                })
                return True
            
            # Traverse children based on edge types
            for edge in node.children:
                child_triggered = self._traverse(
                    edge.target_id, data, entity, 
                    context_modifier, current_path, results
                )
                
                # Handle OR edges - if one triggers, stop
                if edge.edge_type == EdgeType.OR and child_triggered:
                    return True
        else:
            # Check ELSE edges when condition is false
            for edge in node.children:
                if edge.edge_type == EdgeType.ELSE:
                    self._traverse(
                        edge.target_id, data, entity,
                        context_modifier, current_path, results
                    )
        
        return triggered
    
    def to_english_rules(self) -> List[str]:
        """Convert entire graph to human-readable rules."""
        if not self.root_node_id:
            return ["No rules defined"]
        
        rules = []
        self._rules_to_english(self.root_node_id, [], rules)
        return rules
    
    def _rules_to_english(self, node_id: str, path: List[str], rules: List[str]) -> None:
        """Recursively build English rules."""
        node = self.nodes.get(node_id)
        if not node:
            return
        
        current_path = path + [node.to_english()]
        
        if node.action:
            rule_text = " AND ".join(current_path)
            rules.append(rule_text)
            return
        
        for edge in node.children:
            connector = f" [{edge.edge_type.value}] "
            child_path = current_path if edge.edge_type in [EdgeType.AND, EdgeType.CHAIN] else path
            self._rules_to_english(edge.target_id, child_path, rules)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'graph_id': self.graph_id,
            'name': self.name,
            'description': self.description,
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
            'root_node_id': self.root_node_id,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'author': self.author,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuleGraph':
        """Deserialize from dictionary."""
        graph = cls(
            graph_id=data['graph_id'],
            name=data['name'],
            description=data.get('description', '')
        )
        graph.nodes = {k: RuleNode.from_dict(v) for k, v in data.get('nodes', {}).items()}
        graph.root_node_id = data.get('root_node_id')
        graph.version = data.get('version', 1)
        graph.created_at = datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now()
        graph.updated_at = datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else datetime.now()
        graph.author = data.get('author', 'system')
        graph.tags = data.get('tags', [])
        return graph
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RuleGraph':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def save(self, filepath: str) -> None:
        """Save graph to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> 'RuleGraph':
        """Load graph from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())
    
    def clone(self) -> 'RuleGraph':
        """Create a deep copy of the graph."""
        return RuleGraph.from_dict(self.to_dict())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the rule graph."""
        total_activations = sum(n.metadata.hits_last_30d for n in self.nodes.values())
        action_nodes = [n for n in self.nodes.values() if n.action]
        
        return {
            'total_nodes': len(self.nodes),
            'action_nodes': len(action_nodes),
            'total_edges': sum(len(n.children) for n in self.nodes.values()),
            'total_activations_30d': total_activations,
            'avg_false_positive_rate': sum(n.metadata.false_positive_rate for n in self.nodes.values()) / max(len(self.nodes), 1),
            'version': self.version,
            'last_updated': self.updated_at.isoformat()
        }


def create_zscore_rule_graph(threshold: float = 3.0, entity: str = None) -> RuleGraph:
    """Factory function to create a standard Z-score anomaly detection rule graph."""
    graph = RuleGraph(
        name="Z-Score Anomaly Detection",
        description="Detects anomalies based on statistical Z-score thresholds"
    )
    
    # Root: Check Z-score
    zscore_check = RuleNode(
        node_id="zscore_check",
        node_type=NodeType.THRESHOLD,
        name="Z-Score Check",
        description="Check if absolute Z-score exceeds threshold",
        condition=f"abs(zscore) > {threshold}",
        condition_code="abs(data.get('zscore', 0)) > threshold",
        field_name="zscore",
        operator="abs>",
        threshold=ThresholdConfig(base=threshold, min_value=1.5, max_value=5.0)
    )
    graph.add_node(zscore_check, is_root=True)
    
    # High severity check
    high_severity = RuleNode(
        node_id="high_severity_check",
        node_type=NodeType.THRESHOLD,
        name="High Severity Check",
        description="Check if Z-score indicates critical anomaly",
        condition=f"abs(zscore) > {threshold * 1.5}",
        condition_code=f"abs(data.get('zscore', 0)) > {threshold * 1.5}",
        field_name="zscore",
        operator="abs>",
        threshold=ThresholdConfig(base=threshold * 1.5)
    )
    graph.add_node(high_severity)
    
    # Flag critical anomaly
    flag_critical = RuleNode(
        node_id="flag_critical",
        node_type=NodeType.ACTION,
        name="Flag Critical Anomaly",
        description="Flag as critical anomaly requiring immediate attention",
        condition="Critical anomaly detected",
        action="flag_anomaly",
        severity="critical"
    )
    graph.add_node(flag_critical)
    
    # Flag high anomaly
    flag_high = RuleNode(
        node_id="flag_high",
        node_type=NodeType.ACTION,
        name="Flag High Anomaly",
        description="Flag as high-priority anomaly",
        condition="High anomaly detected",
        action="flag_anomaly",
        severity="high"
    )
    graph.add_node(flag_high)
    
    # Normal state
    normal = RuleNode(
        node_id="normal",
        node_type=NodeType.ACTION,
        name="Normal",
        description="No anomaly detected",
        condition="Within normal range",
        action="normal",
        severity="low"
    )
    graph.add_node(normal)
    
    # Add edges
    graph.add_edge("zscore_check", "high_severity_check", EdgeType.CHAIN)
    graph.add_edge("high_severity_check", "flag_critical", EdgeType.ESCALATE)
    graph.add_edge("high_severity_check", "flag_high", EdgeType.ELSE)
    graph.add_edge("zscore_check", "normal", EdgeType.ELSE)
    
    return graph

