"""
Interpretable Decision Tree Wrapper
====================================

Wrapper for scikit-learn DecisionTreeClassifier with interpretable export
capabilities, converting tree rules to human-readable English.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import json
import uuid
import warnings

try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class InterpretableTreeAgent:
    """
    Wrapper for DecisionTreeClassifier providing interpretable rule extraction.
    
    Features:
    - Train decision trees on feature data
    - Extract rules as human-readable English
    - Convert tree paths to rule graphs
    - Track performance metrics
    - Support entity-specific trees
    """
    
    def __init__(self, tree_id: str = None, name: str = "", 
                 max_depth: int = 5, min_samples_leaf: int = 10,
                 random_state: int = 42):
        """
        Initialize the interpretable tree agent.
        
        Args:
            tree_id: Unique identifier for the tree
            name: Human-readable name
            max_depth: Maximum depth of the decision tree
            min_samples_leaf: Minimum samples required at leaf node
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        self.tree_id = tree_id or str(uuid.uuid4())[:8]
        self.name = name
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.tree: Optional[DecisionTreeClassifier] = None
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        
        # Training metadata
        self.trained_at: Optional[datetime] = None
        self.training_samples: int = 0
        self.training_accuracy: float = 0.0
        self.cv_scores: List[float] = []
        
        # Extracted rules
        self._rules_cache: Optional[List[Dict[str, Any]]] = None
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              class_names: List[str] = None) -> Dict[str, Any]:
        """
        Train the decision tree on provided data.
        
        Args:
            X: Feature DataFrame
            y: Target series (binary labels or class indices)
            class_names: Optional list of class names
        
        Returns:
            Training metrics dictionary
        """
        self.feature_names = list(X.columns)
        self.class_names = class_names or [str(c) for c in sorted(y.unique())]
        
        # Initialize and train tree
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        # Handle missing values
        X_clean = X.fillna(0)
        
        self.tree.fit(X_clean, y)
        
        # Calculate metrics
        self.training_samples = len(y)
        self.training_accuracy = self.tree.score(X_clean, y)
        
        # Cross-validation (suppress warnings for small imbalanced datasets)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, 
                                       message='.*least populated class.*')
                # Use minimum 2 folds, max 5, based on minority class size
                min_class_count = min(sum(y == 0), sum(y == 1)) if len(np.unique(y)) > 1 else len(y)
                n_splits = max(2, min(5, min_class_count))
                self.cv_scores = list(cross_val_score(self.tree, X_clean, y, cv=n_splits))
        except Exception:
            self.cv_scores = [self.training_accuracy]
        
        self.trained_at = datetime.now()
        self._rules_cache = None  # Invalidate cache
        
        return {
            'tree_id': self.tree_id,
            'training_samples': self.training_samples,
            'training_accuracy': self.training_accuracy,
            'cv_mean': np.mean(self.cv_scores),
            'cv_std': np.std(self.cv_scores),
            'feature_importances': dict(zip(self.feature_names, self.tree.feature_importances_))
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.tree is None:
            raise ValueError("Tree not trained. Call train() first.")
        
        X_clean = X.fillna(0)
        return self.tree.predict(X_clean)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.tree is None:
            raise ValueError("Tree not trained. Call train() first.")
        
        X_clean = X.fillna(0)
        return self.tree.predict_proba(X_clean)
    
    def get_decision_path(self, X: pd.DataFrame) -> List[List[str]]:
        """
        Get the decision path for each sample in human-readable format.
        
        Returns:
            List of decision paths, each path is a list of condition strings
        """
        if self.tree is None:
            raise ValueError("Tree not trained. Call train() first.")
        
        X_clean = X.fillna(0)
        node_indicator = self.tree.decision_path(X_clean)
        
        paths = []
        tree_ = self.tree.tree_
        
        for sample_idx in range(len(X_clean)):
            # Get the nodes traversed for this sample
            node_indices = node_indicator[sample_idx].indices
            
            path = []
            for node_id in node_indices:
                # Skip leaf nodes
                if tree_.children_left[node_id] == tree_.children_right[node_id]:
                    continue
                
                # Get the feature and threshold
                feature_idx = tree_.feature[node_id]
                threshold = tree_.threshold[node_id]
                feature_name = self.feature_names[feature_idx]
                
                # Get the actual value
                value = X_clean.iloc[sample_idx, feature_idx]
                
                if value <= threshold:
                    condition = f"{feature_name} <= {threshold:.4f} (value: {value:.4f})"
                else:
                    condition = f"{feature_name} > {threshold:.4f} (value: {value:.4f})"
                
                path.append(condition)
            
            paths.append(path)
        
        return paths
    
    def extract_rules(self) -> List[Dict[str, Any]]:
        """
        Extract all rules from the decision tree.
        
        Returns:
            List of rule dictionaries with conditions and outcomes
        """
        if self._rules_cache is not None:
            return self._rules_cache
        
        if self.tree is None:
            raise ValueError("Tree not trained. Call train() first.")
        
        rules = []
        tree_ = self.tree.tree_
        
        def recurse(node_id: int, path: List[Dict[str, Any]]) -> None:
            """Recursively traverse tree to extract rules."""
            # Check if leaf node
            if tree_.children_left[node_id] == tree_.children_right[node_id]:
                # This is a leaf - create rule
                class_counts = tree_.value[node_id][0]
                predicted_class = np.argmax(class_counts)
                confidence = class_counts[predicted_class] / class_counts.sum()
                
                rules.append({
                    'conditions': path.copy(),
                    'prediction': self.class_names[predicted_class] if predicted_class < len(self.class_names) else str(predicted_class),
                    'confidence': float(confidence),
                    'samples': int(class_counts.sum()),
                    'class_distribution': {
                        self.class_names[i] if i < len(self.class_names) else str(i): int(c)
                        for i, c in enumerate(class_counts)
                    }
                })
                return
            
            # Get feature and threshold
            feature_idx = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]
            feature_name = self.feature_names[feature_idx]
            
            # Left child (<=)
            left_condition = {
                'feature': feature_name,
                'operator': '<=',
                'threshold': float(threshold)
            }
            recurse(tree_.children_left[node_id], path + [left_condition])
            
            # Right child (>)
            right_condition = {
                'feature': feature_name,
                'operator': '>',
                'threshold': float(threshold)
            }
            recurse(tree_.children_right[node_id], path + [right_condition])
        
        recurse(0, [])
        self._rules_cache = rules
        return rules
    
    def rules_to_english(self) -> List[str]:
        """
        Convert extracted rules to human-readable English sentences.
        
        Returns:
            List of English rule descriptions
        """
        rules = self.extract_rules()
        english_rules = []
        
        for rule in rules:
            conditions = []
            for cond in rule['conditions']:
                op_text = "is at most" if cond['operator'] == '<=' else "is greater than"
                conditions.append(f"{cond['feature']} {op_text} {cond['threshold']:.4f}")
            
            if conditions:
                condition_text = " AND ".join(conditions)
                rule_text = (
                    f"IF {condition_text}\n"
                    f"   THEN predict: {rule['prediction']} "
                    f"(confidence: {rule['confidence']:.1%}, samples: {rule['samples']})"
                )
            else:
                rule_text = f"DEFAULT: predict {rule['prediction']}"
            
            english_rules.append(rule_text)
        
        return english_rules
    
    def get_sklearn_text_export(self) -> str:
        """Get the standard sklearn text export of the tree."""
        if self.tree is None:
            raise ValueError("Tree not trained. Call train() first.")
        
        return export_text(
            self.tree, 
            feature_names=self.feature_names,
            class_names=self.class_names
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances as a dictionary."""
        if self.tree is None:
            raise ValueError("Tree not trained. Call train() first.")
        
        importances = dict(zip(self.feature_names, self.tree.feature_importances_))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    def to_rule_graph(self) -> 'RuleGraph':
        """
        Convert the decision tree to a RuleGraph.
        
        Returns:
            RuleGraph representation of the tree
        """
        from .rule_graph import RuleGraph, RuleNode, RuleEdge, NodeType, EdgeType, ThresholdConfig
        
        if self.tree is None:
            raise ValueError("Tree not trained. Call train() first.")
        
        graph = RuleGraph(
            graph_id=f"tree_{self.tree_id}",
            name=f"Decision Tree: {self.name}",
            description=f"Converted from decision tree trained on {self.training_samples} samples"
        )
        
        tree_ = self.tree.tree_
        
        def create_nodes(node_id: int, parent_id: Optional[str] = None, 
                         edge_type: EdgeType = None) -> str:
            """Recursively create nodes from tree structure."""
            current_node_id = f"node_{node_id}"
            
            # Check if leaf
            if tree_.children_left[node_id] == tree_.children_right[node_id]:
                class_counts = tree_.value[node_id][0]
                predicted_class = np.argmax(class_counts)
                confidence = class_counts[predicted_class] / class_counts.sum()
                
                class_name = self.class_names[predicted_class] if predicted_class < len(self.class_names) else str(predicted_class)
                is_anomaly = class_name.lower() in ['anomaly', 'true', '1', 'yes']
                
                node = RuleNode(
                    node_id=current_node_id,
                    node_type=NodeType.ACTION,
                    name=f"Leaf: {class_name}",
                    description=f"Predict {class_name} with {confidence:.1%} confidence",
                    condition=f"Prediction: {class_name}",
                    action="flag_anomaly" if is_anomaly else "normal",
                    severity="high" if is_anomaly else "low"
                )
                node.metadata.true_positive_rate = confidence
                graph.add_node(node)
            else:
                # Internal node
                feature_idx = tree_.feature[node_id]
                threshold = tree_.threshold[node_id]
                feature_name = self.feature_names[feature_idx]
                
                node = RuleNode(
                    node_id=current_node_id,
                    node_type=NodeType.THRESHOLD,
                    name=f"Check {feature_name}",
                    description=f"Split on {feature_name} at {threshold:.4f}",
                    condition=f"{feature_name} <= {threshold:.4f}",
                    field_name=feature_name,
                    operator="<=",
                    threshold=ThresholdConfig(base=threshold)
                )
                graph.add_node(node, is_root=(node_id == 0))
                
                # Create children
                left_id = create_nodes(tree_.children_left[node_id])
                right_id = create_nodes(tree_.children_right[node_id])
                
                # Add edges
                graph.add_edge(current_node_id, left_id, EdgeType.CHAIN)  # <= goes left
                graph.add_edge(current_node_id, right_id, EdgeType.ELSE)  # > goes right
            
            return current_node_id
        
        create_nodes(0)
        return graph
    
    def explain_prediction(self, X: pd.DataFrame, sample_idx: int = 0) -> str:
        """
        Provide a detailed explanation for a specific prediction.
        
        Args:
            X: Feature DataFrame
            sample_idx: Index of sample to explain
        
        Returns:
            Human-readable explanation string
        """
        if self.tree is None:
            raise ValueError("Tree not trained. Call train() first.")
        
        X_clean = X.fillna(0)
        prediction = self.predict(X_clean.iloc[[sample_idx]])[0]
        proba = self.predict_proba(X_clean.iloc[[sample_idx]])[0]
        path = self.get_decision_path(X_clean.iloc[[sample_idx]])[0]
        
        explanation = [
            f"Prediction: {prediction}",
            f"Confidence: {max(proba):.1%}",
            "",
            "Decision Path:",
        ]
        
        for i, step in enumerate(path, 1):
            explanation.append(f"  {i}. {step}")
        
        explanation.extend([
            "",
            "Top Contributing Features:"
        ])
        
        # Add feature importance info
        importances = self.get_feature_importance()
        for feature, importance in list(importances.items())[:5]:
            value = X_clean.iloc[sample_idx][feature]
            explanation.append(f"  - {feature}: {value:.4f} (importance: {importance:.3f})")
        
        return "\n".join(explanation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tree metadata and configuration."""
        return {
            'tree_id': self.tree_id,
            'name': self.name,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'trained_at': self.trained_at.isoformat() if self.trained_at else None,
            'training_samples': self.training_samples,
            'training_accuracy': self.training_accuracy,
            'cv_scores': self.cv_scores,
            'feature_importances': self.get_feature_importance() if self.tree else {},
            'rules': self.extract_rules() if self.tree else []
        }
    
    def save(self, filepath: str) -> None:
        """Save tree to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterpretableTreeAgent':
        """Create instance from dictionary (without trained model)."""
        agent = cls(
            tree_id=data['tree_id'],
            name=data['name'],
            max_depth=data['max_depth'],
            min_samples_leaf=data['min_samples_leaf'],
            random_state=data['random_state']
        )
        agent.feature_names = data.get('feature_names', [])
        agent.class_names = data.get('class_names', [])
        agent.trained_at = datetime.fromisoformat(data['trained_at']) if data.get('trained_at') else None
        agent.training_samples = data.get('training_samples', 0)
        agent.training_accuracy = data.get('training_accuracy', 0.0)
        agent.cv_scores = data.get('cv_scores', [])
        return agent

