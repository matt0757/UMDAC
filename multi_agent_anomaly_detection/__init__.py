"""
Multi-Agent Anomaly Detection System
=====================================

A comprehensive, interpretable anomaly detection framework for cash flow data
that combines multiple specialized detection agents with rule graphs, decision
trees, and external context integration.

Key Components:
- Core: RuleGraph, KnowledgeBase, InterpretableTree
- Agents: Pattern, Statistical, Rule, Temporal, Category, ExternalContext
- Evolution: Rule adaptation and learning from feedback
- Coordination: Meta-coordinator for agent orchestration

Author: UMDAC Team
Version: 1.0.0
"""

__version__ = "1.0.0"

from .core.rule_graph import RuleGraph, RuleNode
from .core.knowledge_base import KnowledgeBase
from .core.interpretable_tree import InterpretableTreeAgent
from .core.models import AnomalyFlag, Feedback, RuleScore, Mutation

from .agents.base_agent import BaseDetectorAgent
from .agents.pattern_agent import PatternAgent
from .agents.statistical_agent import StatisticalAgent
from .agents.rule_agent import RuleAgent
from .agents.temporal_agent import TemporalAgent
from .agents.category_agent import CategoryAgent

from .evolution.rule_evolution import RuleEvolutionAgent
from .evolution.feedback import FeedbackCollector, PerformanceTracker

from .coordination.meta_coordinator import MetaCoordinator

__all__ = [
    # Core
    'RuleGraph', 'RuleNode', 'KnowledgeBase', 'InterpretableTreeAgent',
    'AnomalyFlag', 'Feedback', 'RuleScore', 'Mutation',
    # Agents
    'BaseDetectorAgent', 'PatternAgent', 'StatisticalAgent',
    'RuleAgent', 'TemporalAgent', 'CategoryAgent',
    # Evolution
    'RuleEvolutionAgent', 'FeedbackCollector', 'PerformanceTracker',
    # Coordination
    'MetaCoordinator',
]

