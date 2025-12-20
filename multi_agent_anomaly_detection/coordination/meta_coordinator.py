"""
Meta-Coordinator Agent
======================

Orchestrates all detection agents, resolves conflicts, and produces
final ensemble verdicts with aggregated explanations.
"""

from typing import Dict, List, Optional, Any, Type
from datetime import datetime
from dataclasses import dataclass
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from ..core.models import (
    AnomalyFlag, DetectionContext, AgentResult, EnsembleVerdict,
    Severity, AnomalyType
)
from ..core.knowledge_base import KnowledgeBase
from ..agents.base_agent import BaseDetectorAgent
from ..agents.statistical_agent import StatisticalAgent
from ..agents.pattern_agent import PatternAgent
from ..agents.rule_agent import RuleAgent
from ..agents.temporal_agent import TemporalAgent
from ..agents.category_agent import CategoryAgent
from ..evolution.rule_evolution import RuleEvolutionAgent
from ..evolution.feedback import FeedbackCollector, PerformanceTracker


@dataclass
class ConflictResolution:
    """Result of conflict resolution between agents."""
    is_anomaly: bool
    final_confidence: float
    severity: Severity
    agreeing_agents: List[str]
    disagreeing_agents: List[str]
    resolution_method: str
    explanation: str


class MetaCoordinator:
    """
    Central coordinator for all detection agents.
    
    Responsibilities:
    - Agent scheduling and lifecycle management
    - Conflict resolution when agents disagree
    - Ensemble flagging with confidence weighting
    - Explanation aggregation
    - External context integration
    """
    
    # Conflict resolution matrix thresholds
    CONFLICT_MATRIX = {
        ('HIGH', 'HIGH'): {'is_anomaly': True, 'confidence': 0.95, 'action': 'FLAG'},
        ('HIGH', 'MEDIUM'): {'is_anomaly': True, 'confidence': 0.85, 'action': 'FLAG'},
        ('HIGH', 'LOW'): {'is_anomaly': True, 'confidence': 0.60, 'action': 'REVIEW'},
        ('MEDIUM', 'MEDIUM'): {'is_anomaly': True, 'confidence': 0.75, 'action': 'FLAG'},
        ('MEDIUM', 'LOW'): {'is_anomaly': True, 'confidence': 0.55, 'action': 'REVIEW'},
        ('LOW', 'LOW'): {'is_anomaly': False, 'confidence': 0.90, 'action': 'NORMAL'},
    }
    
    def __init__(self, knowledge_base: KnowledgeBase = None,
                 enable_parallel: bool = True,
                 max_workers: int = 4):
        """
        Initialize the meta-coordinator.
        
        Args:
            knowledge_base: Shared knowledge base
            enable_parallel: Whether to run agents in parallel
            max_workers: Maximum parallel workers
        """
        self.kb = knowledge_base or KnowledgeBase(use_json_fallback=True)
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        
        # Initialize agents
        self.agents: Dict[str, BaseDetectorAgent] = {}
        
        # Evolution and feedback
        self.evolution_agent = RuleEvolutionAgent(knowledge_base=self.kb)
        self.feedback_collector = FeedbackCollector(knowledge_base=self.kb)
        self.performance_tracker = PerformanceTracker(knowledge_base=self.kb)
        
        # Configuration
        self.config = {
            'min_agents_for_ensemble': 2,
            'confidence_weight_by_performance': True,
            'default_context_modifier': 1.0
        }
        
        # Initialize standard agents
        self._init_agents()
    
    def _init_agents(self) -> None:
        """Initialize all detector agents."""
        # Statistical agent
        self.agents['statistical'] = StatisticalAgent(
            knowledge_base=self.kb,
            name="Statistical Detector"
        )
        
        # Pattern agent
        self.agents['pattern'] = PatternAgent(
            knowledge_base=self.kb,
            name="Pattern Detector"
        )
        
        # Rule agent
        self.agents['rule'] = RuleAgent(
            knowledge_base=self.kb,
            name="Business Rule Detector"
        )
        
        # Temporal agent
        self.agents['temporal'] = TemporalAgent(
            knowledge_base=self.kb,
            name="Temporal Detector"
        )
        
        # Category agent
        self.agents['category'] = CategoryAgent(
            knowledge_base=self.kb,
            name="Category Detector"
        )
    
    def add_agent(self, agent: BaseDetectorAgent) -> None:
        """Add a custom agent to the coordinator."""
        self.agents[agent.agent_id] = agent
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the coordinator."""
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def get_context(self, entity: str = None, 
                    timestamp: datetime = None) -> DetectionContext:
        """
        Get detection context.
        
        Args:
            entity: Entity being analyzed
            timestamp: Timestamp of the data
        
        Returns:
            DetectionContext with default signals
        """
        return DetectionContext(
            entity=entity or 'unknown',
            timestamp=timestamp or datetime.now(),
            threshold_modifier=self.config['default_context_modifier']
        )
    
    def run_detection(self, data: pd.DataFrame, 
                       entity: str = None,
                       include_agents: List[str] = None,
                       exclude_agents: List[str] = None) -> EnsembleVerdict:
        """
        Run all agents on the data and produce ensemble verdict.
        
        Args:
            data: Transaction or feature data
            entity: Entity being analyzed
            include_agents: Only run these agents (optional)
            exclude_agents: Skip these agents (optional)
        
        Returns:
            EnsembleVerdict with final decision and explanations
        """
        start_time = datetime.now()
        
        # Determine entity
        if entity is None and 'Entity' in data.columns:
            entities = data['Entity'].unique()
            if len(entities) == 1:
                entity = entities[0]
        
        # Get context
        context = self.get_context(entity, datetime.now())
        
        # Determine which agents to run
        agents_to_run = self._select_agents(include_agents, exclude_agents)
        
        # Run agents
        if self.enable_parallel and len(agents_to_run) > 1:
            agent_results = self._run_parallel(agents_to_run, data, context)
        else:
            agent_results = self._run_sequential(agents_to_run, data, context)
        
        # Resolve conflicts and create ensemble verdict
        verdict = self._create_ensemble_verdict(agent_results, entity, context)
        
        # Save to knowledge base
        self._save_results(verdict)
        
        return verdict
    
    def _select_agents(self, include: List[str] = None,
                        exclude: List[str] = None) -> Dict[str, BaseDetectorAgent]:
        """Select agents to run based on include/exclude lists."""
        agents = {}
        
        for agent_id, agent in self.agents.items():
            # Skip external context agent (it provides context, not detection)
            if agent_id == 'external':
                continue
            
            if not agent.is_active:
                continue
            
            if include and agent_id not in include:
                continue
            
            if exclude and agent_id in exclude:
                continue
            
            agents[agent_id] = agent
        
        return agents
    
    def _run_parallel(self, agents: Dict[str, BaseDetectorAgent],
                       data: pd.DataFrame,
                       context: DetectionContext) -> List[AgentResult]:
        """Run agents in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(agent.run, data, context): agent_id
                for agent_id, agent in agents.items()
            }
            
            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Agent {agent_id} failed: {e}")
        
        return results
    
    def _run_sequential(self, agents: Dict[str, BaseDetectorAgent],
                         data: pd.DataFrame,
                         context: DetectionContext) -> List[AgentResult]:
        """Run agents sequentially."""
        results = []
        
        for agent_id, agent in agents.items():
            try:
                result = agent.run(data, context)
                results.append(result)
            except Exception as e:
                print(f"Agent {agent_id} failed: {e}")
        
        return results
    
    def _create_ensemble_verdict(self, agent_results: List[AgentResult],
                                   entity: str,
                                   context: DetectionContext) -> EnsembleVerdict:
        """Create ensemble verdict from agent results."""
        # Collect all flags
        all_flags = []
        for result in agent_results:
            all_flags.extend(result.flags)
        
        if not all_flags:
            # No anomalies detected by any agent
            return EnsembleVerdict(
                verdict_id=str(uuid.uuid4())[:8],
                entity=entity or 'unknown',
                timestamp=datetime.now(),
                is_anomaly=False,
                final_confidence=0.90,
                severity=Severity.LOW,
                agent_results=agent_results,
                agreeing_agents=[r.agent_id for r in agent_results],
                disagreeing_agents=[],
                primary_flags=[],
                secondary_flags=[],
                explanation="No anomalies detected by any agent.",
                decision_factors=["All agents report normal behavior"],
                recommended_action="NORMAL - No action required"
            )
        
        # Classify agents by their detection
        detecting_agents = []
        non_detecting_agents = []
        
        for result in agent_results:
            if result.flags:
                detecting_agents.append(result.agent_id)
            else:
                non_detecting_agents.append(result.agent_id)
        
        # Resolve conflicts
        resolution = self._resolve_conflicts(agent_results, context)
        
        # Sort flags by severity and confidence
        sorted_flags = sorted(
            all_flags,
            key=lambda f: (
                {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(f.severity.value, 0),
                f.confidence
            ),
            reverse=True
        )
        
        # Primary flags = high-severity or high-confidence flags (critical, high severity OR confidence >= 0.85)
        # Secondary flags = everything else
        primary_flags = [
            f for f in sorted_flags 
            if f.severity.value in ('critical', 'high') or f.confidence >= 0.85
        ]
        secondary_flags = [f for f in sorted_flags if f not in primary_flags]
        
        # Generate explanation
        explanation = self._generate_explanation(primary_flags, resolution, context)
        
        # Decision factors
        decision_factors = [
            f"{len(detecting_agents)} of {len(agent_results)} agents detected anomalies",
            f"Highest severity: {primary_flags[0].severity.value if primary_flags else 'none'}",
            f"Market stress: {context.market_stress}"
        ]
        
        # Recommended action
        if resolution.is_anomaly:
            if resolution.final_confidence >= 0.85:
                action = "INVESTIGATE - High-confidence anomaly requires immediate review"
            elif resolution.final_confidence >= 0.6:
                action = "REVIEW - Moderate-confidence finding requires verification"
            else:
                action = "MONITOR - Low-confidence flag, continue monitoring"
        else:
            action = "NORMAL - No action required"
        
        return EnsembleVerdict(
            verdict_id=str(uuid.uuid4())[:8],
            entity=entity or 'unknown',
            timestamp=datetime.now(),
            is_anomaly=resolution.is_anomaly,
            final_confidence=resolution.final_confidence,
            severity=resolution.severity,
            agent_results=agent_results,
            agreeing_agents=resolution.agreeing_agents,
            disagreeing_agents=resolution.disagreeing_agents,
            primary_flags=primary_flags,
            secondary_flags=secondary_flags,
            explanation=explanation,
            decision_factors=decision_factors,
            recommended_action=action
        )
    
    def _resolve_conflicts(self, agent_results: List[AgentResult],
                            context: DetectionContext) -> ConflictResolution:
        """Resolve conflicts between agent results."""
        # Classify detection levels
        levels = {'HIGH': [], 'MEDIUM': [], 'LOW': [], 'NONE': []}
        
        severity_order = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
        
        for result in agent_results:
            if not result.flags:
                levels['NONE'].append(result.agent_id)
            else:
                # Get highest severity from this agent
                max_severity = max(
                    result.flags, key=lambda f: severity_order.get(f.severity, 0)
                ).severity
                if max_severity in [Severity.CRITICAL, Severity.HIGH]:
                    levels['HIGH'].append(result.agent_id)
                elif max_severity == Severity.MEDIUM:
                    levels['MEDIUM'].append(result.agent_id)
                else:
                    levels['LOW'].append(result.agent_id)
        
        # Determine overall level
        if levels['HIGH']:
            primary_level = 'HIGH'
        elif levels['MEDIUM']:
            primary_level = 'MEDIUM'
        elif levels['LOW']:
            primary_level = 'LOW'
        else:
            primary_level = 'NONE'
        
        # Check for consensus
        detecting_agents = levels['HIGH'] + levels['MEDIUM'] + levels['LOW']
        total_agents = len(agent_results)
        
        # Calculate dynamic confidence based on multiple factors
        if len(detecting_agents) == 0:
            # No agents detected anything
            is_anomaly = False
            confidence = 0.95  # High confidence it's normal
        else:
            is_anomaly = True
            
            # Base confidence from agent agreement ratio
            agreement_ratio = len(detecting_agents) / total_agents if total_agents > 0 else 0
            
            # Calculate weighted confidence from individual agent results
            total_flag_confidence = 0
            total_flags = 0
            severity_weights = {'critical': 1.0, 'high': 0.85, 'medium': 0.65, 'low': 0.4}
            
            for result in agent_results:
                for flag in result.flags:
                    weight = severity_weights.get(flag.severity.value, 0.5)
                    total_flag_confidence += flag.confidence * weight
                    total_flags += 1
            
            avg_flag_confidence = total_flag_confidence / total_flags if total_flags > 0 else 0.5
            
            # Combine factors:
            # - 40% from agent agreement ratio
            # - 40% from average flag confidence (weighted by severity)
            # - 20% from severity level bonus
            severity_bonus = {'HIGH': 0.15, 'MEDIUM': 0.08, 'LOW': 0.0, 'NONE': -0.1}
            
            confidence = (
                0.40 * agreement_ratio +
                0.40 * avg_flag_confidence +
                0.20 + severity_bonus.get(primary_level, 0)
            )
            
            # Clamp confidence to valid range
            confidence = max(0.30, min(0.98, confidence))
        
        # Determine severity
        if primary_level == 'HIGH':
            severity = Severity.HIGH
        elif primary_level == 'MEDIUM':
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW
        
        return ConflictResolution(
            is_anomaly=is_anomaly,
            final_confidence=confidence,
            severity=severity,
            agreeing_agents=detecting_agents if is_anomaly else levels['NONE'],
            disagreeing_agents=levels['NONE'] if is_anomaly else detecting_agents,
            resolution_method='consensus' if len(detecting_agents) >= 2 else 'single_agent',
            explanation=f"{len(detecting_agents)} agents detected anomalies"
        )
    
    def _generate_explanation(self, primary_flags: List[AnomalyFlag],
                               resolution: ConflictResolution,
                               context: DetectionContext) -> str:
        """Generate aggregated explanation."""
        if not primary_flags:
            return "No significant anomalies detected across all agents."
        
        lines = [
            f"Ensemble Analysis: {resolution.resolution_method.replace('_', ' ').title()}",
            f"Final Confidence: {resolution.final_confidence:.1%}",
            "",
            "Primary Findings:"
        ]
        
        for i, flag in enumerate(primary_flags, 1):
            lines.append(f"  {i}. [{flag.agent_id}] {flag.description}")
            lines.append(f"     Severity: {flag.severity.value}, Confidence: {flag.confidence:.1%}")
        
        if context.market_stress != 'NORMAL':
            lines.append("")
            lines.append(f"External Context: Market stress is {context.market_stress}")
            lines.append(f"  Threshold adjustment applied: {context.threshold_modifier:.0%}")
        
        return "\n".join(lines)
    
    def _save_results(self, verdict: EnsembleVerdict) -> None:
        """Save results to knowledge base."""
        if not self.kb:
            return
        
        # Save all primary flags
        for flag in verdict.primary_flags:
            self.kb.save_anomaly_flag(flag)
        
        # Save secondary flags
        for flag in verdict.secondary_flags:
            self.kb.save_anomaly_flag(flag)
    
    def train_agents(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all trainable agents on historical data.
        
        Args:
            training_data: Historical data with features
        
        Returns:
            Training results per agent
        """
        results = {}
        
        # Train pattern agent
        if 'pattern' in self.agents:
            pattern_agent = self.agents['pattern']
            results['pattern'] = pattern_agent.train_patterns(training_data)
        
        # Train category agent
        if 'category' in self.agents:
            category_agent = self.agents['category']
            results['category'] = category_agent.train_category_models(training_data)
        
        return results
    
    def run_evolution_cycle(self) -> Dict[str, Any]:
        """Run evolution cycle on all agent rules."""
        all_graphs = {}
        
        for agent_id, agent in self.agents.items():
            for graph_id, graph in agent.rule_graphs.items():
                all_graphs[f"{agent_id}_{graph_id}"] = graph
        
        return self.evolution_agent.run_evolution_cycle(all_graphs)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        status = {
            'total_agents': len(self.agents),
            'active_agents': sum(1 for a in self.agents.values() if a.is_active),
            'agents': {},
            'knowledge_base': self.kb.get_statistics() if self.kb else {},
            'performance': self.performance_tracker.get_summary_report()
        }
        
        for agent_id, agent in self.agents.items():
            status['agents'][agent_id] = {
                'type': agent.agent_type,
                'name': agent.name,
                'is_active': agent.is_active,
                'confidence': agent.get_confidence(),
                'total_detections': agent.total_detections,
                'rule_graphs': len(agent.rule_graphs)
            }
        
        return status
    
    def explain_verdict(self, verdict: EnsembleVerdict) -> str:
        """Generate detailed explanation for a verdict."""
        lines = [
            "=" * 60,
            "ANOMALY DETECTION REPORT",
            "=" * 60,
            "",
            f"Entity: {verdict.entity}",
            f"Time: {verdict.timestamp}",
            f"Verdict: {'ANOMALY DETECTED' if verdict.is_anomaly else 'NORMAL'}",
            f"Confidence: {verdict.final_confidence:.1%}",
            f"Severity: {verdict.severity.value.upper()}",
            "",
            "-" * 60,
            "AGENT ANALYSIS",
            "-" * 60,
        ]
        
        for result in verdict.agent_results:
            status = "DETECTED" if result.flags else "NORMAL"
            lines.append(f"  [{result.agent_type}] {status} ({len(result.flags)} flags)")
        
        lines.extend([
            "",
            "-" * 60,
            "PRIMARY FINDINGS",
            "-" * 60,
        ])
        
        for flag in verdict.primary_flags:
            lines.append(f"\n  > {flag.description}")
            # Show when this occurred
            if flag.timestamp:
                lines.append(f"    When: {flag.timestamp.strftime('%Y-%m-%d') if hasattr(flag.timestamp, 'strftime') else flag.timestamp}")
            lines.append(f"    Agent: {flag.agent_id}")
            lines.append(f"    Severity: {flag.severity.value}")
            lines.append(f"    Rule: {flag.rule_id}")
            
            # Show key contributing factors
            if flag.contributing_factors:
                key_factors = ['week_num', 'Week_Num', 'week_start', 'transaction_id', 'category', 
                               'current_value', 'previous_value', 'amount_column']
                relevant = {k: v for k, v in flag.contributing_factors.items() 
                            if k in key_factors and v is not None}
                if relevant:
                    lines.append("    Details:")
                    for k, v in relevant.items():
                        if isinstance(v, float):
                            lines.append(f"      {k}: {v:,.2f}")
                        else:
                            lines.append(f"      {k}: {v}")
            
            if flag.decision_path:
                lines.append("    Decision Path:")
                for step in flag.decision_path[:3]:
                    lines.append(f"      -> {step}")
        
        lines.extend([
            "",
            "-" * 60,
            "RECOMMENDED ACTION",
            "-" * 60,
            f"  {verdict.recommended_action}",
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)

