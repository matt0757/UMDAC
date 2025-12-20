"""
Rule Evolution Agent
====================

Enables the system to learn and adapt rules over time through
threshold adaptation, rule mutation, and performance-based optimization.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import uuid
import numpy as np

from ..core.models import (
    Feedback, RuleScore, Mutation, MutationType, FeedbackType
)
from ..core.rule_graph import RuleGraph, RuleNode, ThresholdConfig
from ..core.knowledge_base import KnowledgeBase


@dataclass
class MutationCandidate:
    """A candidate mutation with expected impact."""
    mutation: Mutation
    expected_precision_change: float
    expected_recall_change: float
    risk_score: float  # 0-1, higher = riskier


class RuleEvolutionAgent:
    """
    Agent responsible for evolving rules based on feedback.
    
    Implements RL-like behavior:
    - Reward: Reduced false positives
    - Penalty: Missed anomalies (false negatives)
    
    Features:
    - Threshold adaptation
    - Rule mutation (add/remove/modify conditions)
    - Tree retraining triggers
    - Rollback capability
    """
    
    def __init__(self, knowledge_base: KnowledgeBase = None,
                 min_samples_for_evolution: int = 20,
                 max_fp_rate_threshold: float = 0.20,
                 min_recall_threshold: float = 0.70):
        """
        Initialize the rule evolution agent.
        
        Args:
            knowledge_base: Shared knowledge base
            min_samples_for_evolution: Minimum feedback samples before evolving
            max_fp_rate_threshold: Maximum acceptable false positive rate
            min_recall_threshold: Minimum acceptable recall
        """
        self.kb = knowledge_base
        
        self.config = {
            'min_samples': min_samples_for_evolution,
            'max_fp_rate': max_fp_rate_threshold,
            'min_recall': min_recall_threshold,
            'threshold_adjustment_rate': 0.10,  # ±10% per adjustment
            'confidence_threshold_for_apply': 0.7,
            'max_mutations_per_cycle': 3,
            'cooldown_days': 7  # Days between mutations to same rule
        }
        
        # Track mutation history
        self.mutation_history: Dict[str, List[Mutation]] = {}
        self.rule_cooldowns: Dict[str, datetime] = {}
    
    def evaluate_rule(self, rule_id: str, last_n_days: int = 30) -> RuleScore:
        """
        Calculate performance metrics for a specific rule.
        
        Args:
            rule_id: The rule identifier
            last_n_days: Evaluation period
        
        Returns:
            RuleScore with precision, recall, F1
        """
        if not self.kb:
            return self._empty_score(rule_id)
        
        # Get flags for this rule
        flags = self.kb.get_flags_by_rule(rule_id, last_n_days)
        
        if not flags:
            return self._empty_score(rule_id)
        
        # Get feedback for these flags
        flag_ids = {f.flag_id for f in flags}
        feedbacks = []
        for flag in flags:
            if flag.feedback:
                feedbacks.append(flag.feedback)
        
        # Also get from database
        db_feedbacks = self.kb.get_feedback_for_rule(rule_id, last_n_days)
        for fb in db_feedbacks:
            if fb.flag_id in flag_ids:
                feedbacks.append(fb)
        
        # Calculate metrics
        tp = sum(1 for fb in feedbacks if fb.feedback_type == FeedbackType.TRUE_POSITIVE)
        fp = sum(1 for fb in feedbacks if fb.feedback_type == FeedbackType.FALSE_POSITIVE)
        fn = sum(1 for fb in feedbacks if fb.feedback_type == FeedbackType.FALSE_NEGATIVE)
        
        total_activations = len(flags)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        score = RuleScore(
            rule_id=rule_id,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            total_activations=total_activations,
            evaluation_period_days=last_n_days
        )
        
        # Save to knowledge base
        if self.kb:
            self.kb.save_rule_score(score)
        
        return score
    
    def _empty_score(self, rule_id: str) -> RuleScore:
        """Create an empty score for rules with no data."""
        return RuleScore(
            rule_id=rule_id,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            total_activations=0,
            evaluation_period_days=30
        )
    
    def propose_mutation(self, rule_id: str, 
                          rule_graph: RuleGraph = None) -> List[MutationCandidate]:
        """
        Generate candidate modifications for a rule.
        
        Args:
            rule_id: The rule to mutate
            rule_graph: The rule graph containing the rule
        
        Returns:
            List of MutationCandidate objects
        """
        candidates = []
        
        # Check cooldown
        if rule_id in self.rule_cooldowns:
            cooldown_end = self.rule_cooldowns[rule_id]
            if datetime.now() < cooldown_end:
                return []
        
        # Get current performance
        score = self.evaluate_rule(rule_id)
        
        if score.total_activations < self.config['min_samples']:
            return []
        
        # Determine mutation strategy based on performance
        fp_rate = score.false_positives / score.total_activations if score.total_activations > 0 else 0
        
        # High false positive rate -> increase threshold
        if fp_rate > self.config['max_fp_rate']:
            candidates.extend(self._propose_threshold_increase(rule_id, score, rule_graph))
        
        # Low recall -> decrease threshold
        if score.recall < self.config['min_recall']:
            candidates.extend(self._propose_threshold_decrease(rule_id, score, rule_graph))
        
        # Consider condition modifications
        if rule_graph:
            candidates.extend(self._propose_condition_changes(rule_id, score, rule_graph))
        
        # Sort by expected improvement and risk
        candidates.sort(key=lambda c: (c.expected_precision_change - c.risk_score), reverse=True)
        
        return candidates[:self.config['max_mutations_per_cycle']]
    
    def _propose_threshold_increase(self, rule_id: str, score: RuleScore,
                                     rule_graph: RuleGraph) -> List[MutationCandidate]:
        """Propose threshold increases to reduce false positives."""
        candidates = []
        
        if not rule_graph:
            return candidates
        
        # Find threshold nodes in the graph
        for node_id, node in rule_graph.nodes.items():
            if node.threshold and node_id.startswith(rule_id.split('_')[0]):
                current_threshold = node.threshold.effective_value
                adjustment = self.config['threshold_adjustment_rate']
                new_threshold = current_threshold * (1 + adjustment)
                
                # Check bounds
                if node.threshold.max_value and new_threshold > node.threshold.max_value:
                    continue
                
                mutation = Mutation(
                    mutation_id=str(uuid.uuid4())[:8],
                    rule_id=rule_id,
                    mutation_type=MutationType.THRESHOLD_INCREASE,
                    original_value=current_threshold,
                    new_value=new_threshold,
                    justification=f"High false positive rate ({score.false_positives}/{score.total_activations}). "
                                  f"Increasing threshold by {adjustment:.0%} to reduce false alarms.",
                    confidence=min(0.9, 0.5 + score.false_positives * 0.02),
                    performance_before=score
                )
                
                candidate = MutationCandidate(
                    mutation=mutation,
                    expected_precision_change=0.1,  # Expected improvement
                    expected_recall_change=-0.05,   # Slight recall decrease
                    risk_score=0.3
                )
                candidates.append(candidate)
        
        return candidates
    
    def _propose_threshold_decrease(self, rule_id: str, score: RuleScore,
                                     rule_graph: RuleGraph) -> List[MutationCandidate]:
        """Propose threshold decreases to improve recall."""
        candidates = []
        
        if not rule_graph:
            return candidates
        
        for node_id, node in rule_graph.nodes.items():
            if node.threshold and node_id.startswith(rule_id.split('_')[0]):
                current_threshold = node.threshold.effective_value
                adjustment = self.config['threshold_adjustment_rate']
                new_threshold = current_threshold * (1 - adjustment)
                
                # Check bounds
                if node.threshold.min_value and new_threshold < node.threshold.min_value:
                    continue
                
                mutation = Mutation(
                    mutation_id=str(uuid.uuid4())[:8],
                    rule_id=rule_id,
                    mutation_type=MutationType.THRESHOLD_DECREASE,
                    original_value=current_threshold,
                    new_value=new_threshold,
                    justification=f"Low recall ({score.recall:.1%}). "
                                  f"Decreasing threshold by {adjustment:.0%} to catch more anomalies.",
                    confidence=min(0.9, 0.5 + score.false_negatives * 0.02),
                    performance_before=score
                )
                
                candidate = MutationCandidate(
                    mutation=mutation,
                    expected_precision_change=-0.05,  # Slight precision decrease
                    expected_recall_change=0.15,      # Expected recall improvement
                    risk_score=0.4  # Slightly higher risk
                )
                candidates.append(candidate)
        
        return candidates
    
    def _propose_condition_changes(self, rule_id: str, score: RuleScore,
                                    rule_graph: RuleGraph) -> List[MutationCandidate]:
        """Propose adding or removing conditions."""
        candidates = []
        
        # This is more complex - for now, just log the opportunity
        # In a full implementation, we'd analyze which conditions contribute
        # to false positives and propose removing them
        
        return candidates
    
    def apply_mutation(self, mutation: Mutation, rule_graph: RuleGraph,
                       confidence_override: float = None) -> bool:
        """
        Apply a mutation if confidence exceeds threshold.
        
        Args:
            mutation: The mutation to apply
            rule_graph: The rule graph to modify
            confidence_override: Override the confidence threshold
        
        Returns:
            True if mutation was applied
        """
        threshold = confidence_override or self.config['confidence_threshold_for_apply']
        
        if mutation.confidence < threshold:
            return False
        
        # Find the node to modify
        for node_id, node in rule_graph.nodes.items():
            if not node.threshold:
                continue
            
            if mutation.mutation_type == MutationType.THRESHOLD_INCREASE:
                if abs(node.threshold.effective_value - mutation.original_value) < 0.001:
                    node.threshold.adjusted = mutation.new_value
                    break
            
            elif mutation.mutation_type == MutationType.THRESHOLD_DECREASE:
                if abs(node.threshold.effective_value - mutation.original_value) < 0.001:
                    node.threshold.adjusted = mutation.new_value
                    break
        
        # Record mutation
        mutation.applied = True
        mutation.applied_at = datetime.now()
        
        # Add to history
        if mutation.rule_id not in self.mutation_history:
            self.mutation_history[mutation.rule_id] = []
        self.mutation_history[mutation.rule_id].append(mutation)
        
        # Set cooldown
        self.rule_cooldowns[mutation.rule_id] = datetime.now() + timedelta(
            days=self.config['cooldown_days']
        )
        
        # Save to knowledge base
        if self.kb:
            self.kb.save_mutation(mutation)
            self.kb.save_rule_graph(rule_graph, change_description=mutation.justification)
        
        return True
    
    def rollback(self, mutation_id: str, rule_graph: RuleGraph) -> bool:
        """
        Revert a previously applied mutation.
        
        Args:
            mutation_id: The mutation to rollback
            rule_graph: The rule graph to revert
        
        Returns:
            True if rollback was successful
        """
        # Find the mutation
        mutation = None
        for rule_id, mutations in self.mutation_history.items():
            for m in mutations:
                if m.mutation_id == mutation_id:
                    mutation = m
                    break
        
        if not mutation or not mutation.applied or not mutation.rollback_available:
            return False
        
        # Find the node and revert
        for node_id, node in rule_graph.nodes.items():
            if not node.threshold:
                continue
            
            if abs(node.threshold.adjusted or 0 - mutation.new_value) < 0.001:
                node.threshold.adjusted = mutation.original_value
                break
        
        # Mark as rolled back
        mutation.rollback_available = False
        
        # Save to knowledge base
        if self.kb:
            self.kb.save_mutation(mutation)
            self.kb.save_rule_graph(rule_graph, change_description=f"Rollback: {mutation.mutation_id}")
        
        return True
    
    def evaluate_all_rules(self, rule_graphs: Dict[str, RuleGraph],
                            last_n_days: int = 30) -> Dict[str, RuleScore]:
        """
        Evaluate all rules and return scores.
        
        Args:
            rule_graphs: Dictionary of rule graphs
            last_n_days: Evaluation period
        
        Returns:
            Dictionary mapping rule_id to RuleScore
        """
        scores = {}
        
        for graph_id, graph in rule_graphs.items():
            score = self.evaluate_rule(graph_id, last_n_days)
            scores[graph_id] = score
            
            # Also evaluate individual nodes with thresholds
            for node_id, node in graph.nodes.items():
                if node.threshold:
                    node_score = self.evaluate_rule(f"{graph_id}_{node_id}", last_n_days)
                    scores[f"{graph_id}_{node_id}"] = node_score
        
        return scores
    
    def run_evolution_cycle(self, rule_graphs: Dict[str, RuleGraph]) -> Dict[str, Any]:
        """
        Run a complete evolution cycle on all rules.
        
        Args:
            rule_graphs: Dictionary of rule graphs to evolve
        
        Returns:
            Summary of evolution actions taken
        """
        results = {
            'evaluated_rules': 0,
            'mutations_proposed': 0,
            'mutations_applied': 0,
            'details': []
        }
        
        for graph_id, graph in rule_graphs.items():
            # Evaluate performance
            score = self.evaluate_rule(graph_id)
            results['evaluated_rules'] += 1
            
            if score.total_activations < self.config['min_samples']:
                continue
            
            # Propose mutations
            candidates = self.propose_mutation(graph_id, graph)
            results['mutations_proposed'] += len(candidates)
            
            # Apply best mutation if available
            for candidate in candidates:
                if self.apply_mutation(candidate.mutation, graph):
                    results['mutations_applied'] += 1
                    results['details'].append({
                        'rule_id': graph_id,
                        'mutation_type': candidate.mutation.mutation_type.value,
                        'justification': candidate.mutation.justification
                    })
                    break  # One mutation per rule per cycle
        
        return results
    
    def get_underperforming_rules(self, rule_graphs: Dict[str, RuleGraph],
                                   last_n_days: int = 30) -> List[Tuple[str, RuleScore]]:
        """
        Identify rules with poor performance.
        
        Returns:
            List of (rule_id, score) tuples for underperforming rules
        """
        underperforming = []
        
        scores = self.evaluate_all_rules(rule_graphs, last_n_days)
        
        for rule_id, score in scores.items():
            if score.total_activations < self.config['min_samples']:
                continue
            
            fp_rate = score.false_positives / score.total_activations if score.total_activations > 0 else 0
            
            if fp_rate > self.config['max_fp_rate'] or score.recall < self.config['min_recall']:
                underperforming.append((rule_id, score))
        
        # Sort by F1 score (worst first)
        underperforming.sort(key=lambda x: x[1].f1_score)
        
        return underperforming
    
    def suggest_rule_pruning(self, rule_graphs: Dict[str, RuleGraph],
                              min_f1_score: float = 0.3) -> List[str]:
        """
        Suggest rules that should be pruned due to consistently poor performance.
        
        Args:
            rule_graphs: Dictionary of rule graphs
            min_f1_score: Minimum acceptable F1 score
        
        Returns:
            List of rule IDs that should be considered for removal
        """
        candidates = []
        
        scores = self.evaluate_all_rules(rule_graphs, last_n_days=60)  # Longer window
        
        for rule_id, score in scores.items():
            if score.total_activations < self.config['min_samples'] * 2:
                continue  # Need more data
            
            if score.f1_score < min_f1_score:
                candidates.append(rule_id)
        
        return candidates

