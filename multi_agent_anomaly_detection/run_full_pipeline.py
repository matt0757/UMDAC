"""
================================================================================
END-TO-END MULTI-AGENT ANOMALY DETECTION PIPELINE WITH INTERACTIVE DASHBOARD
================================================================================

AstraZeneca DATATHON 2025 - Anomaly Detection System
Author: Multi-Agent Detection Team
Version: 1.0.0
Last Updated: December 2025

================================================================================
EXECUTIVE SUMMARY
================================================================================

This pipeline provides an end-to-end solution for automated anomaly detection
in cash flow data using a multi-agent ensemble approach. It combines data
processing, multi-agent detection, conflict resolution, and interactive
visualization into a single executable script.

================================================================================
PIPELINE ARCHITECTURE
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐            │
│  │ DataLoader   │───►│ MetaCoordinator  │───►│ AgentEnsemble   │            │
│  │              │    │                  │    │                 │            │
│  │ • Load CSV   │    │ • Orchestrate    │    │ • Statistical   │            │
│  │ • Validate   │    │ • Context        │    │ • Pattern       │            │
│  │ • Prepare    │    │ • Resolve        │    │ • Rule-based    │            │
│  └──────────────┘    └──────────────────┘    │ • Temporal      │            │
│                                              │ • Category      │            │
│                                              │ • External      │            │
│                                              └────────┬────────┘            │
│                                                       │                     │
│                                                       ▼                     │
│                                          ┌────────────────────────┐         │
│                                          │ InteractiveDashboard   │         │
│                                          │ Builder                │         │
│                                          │                        │         │
│                                          │ • Plotly.js charts     │         │
│                                          │ • HTML/CSS generation  │         │
│                                          │ • AstraZeneca theme    │         │
│                                          └────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
USAGE INSTRUCTIONS
================================================================================

BASIC USAGE:
    python run_full_pipeline.py

This will:
1. Load all entity data from processed_data folder
2. Run multi-agent anomaly detection on each entity
3. Generate an interactive HTML dashboard

COMMAND LINE OPTIONS:
    --data-dir PATH     : Path to processed data directory
    --output-dir PATH   : Path to output directory
    --entities LIST     : Comma-separated list of entities to analyze

================================================================================
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import json
import argparse
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_anomaly_detection.coordination.meta_coordinator import MetaCoordinator
from multi_agent_anomaly_detection.core.knowledge_base import KnowledgeBase
from multi_agent_anomaly_detection.core.models import (
    AnomalyFlag, EnsembleVerdict, DetectionContext, AgentResult,
    Severity, AnomalyType
)
from multi_agent_anomaly_detection.utils.helpers import prepare_detection_data


# =============================================================================
# ASTRAZENECA COLOR PALETTE
# =============================================================================
class AZColors:
    """AstraZeneca brand color palette."""
    MULBERRY = "#830051"      # Primary purple/magenta
    LIME = "#C4D600"          # Bright green
    NAVY = "#003865"          # Dark blue
    GRAPHITE = "#3F4444"      # Dark gray
    LIGHT_BLUE = "#68D2DF"    # Teal/cyan
    MAGENTA = "#D0006F"       # Bright pink
    PURPLE = "#3C1053"        # Deep purple
    GOLD = "#F0AB00"          # Orange/gold
    WHITE = "#FFFFFF"
    LIGHT_GRAY = "#F8F9FA"
    
    # Severity color mapping
    SEVERITY = {
        'critical': "#D0006F",  # Magenta
        'high': "#830051",       # Mulberry
        'medium': "#F0AB00",     # Gold
        'low': "#68D2DF"         # Light blue
    }
    
    # Agent colors
    AGENTS = {
        'statistical': "#830051",
        'pattern': "#3C1053",
        'rule': "#003865",
        'temporal': "#68D2DF",
        'category': "#F0AB00"
    }


# =============================================================================
# DATA PROCESSING
# =============================================================================
@dataclass
class EntityData:
    """Container for entity-specific data and results."""
    entity_id: str
    data: pd.DataFrame
    verdict: Optional[EnsembleVerdict] = None
    summary: Dict[str, Any] = field(default_factory=dict)


class DataProcessor:
    """Handles data loading and preparation."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize data processor.
        
        Args:
            data_dir: Path to processed data directory
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            # Default to 1_cashflow_forecast/processed_data
            self.data_dir = Path(__file__).parent.parent / "1_cashflow_forecast" / "processed_data"
    
    def load_all_entities(self) -> Dict[str, EntityData]:
        """Load data for all entities."""
        entities = {}
        
        # Find all weekly_*.csv files
        weekly_files = list(self.data_dir.glob("weekly_*.csv"))
        
        for file in weekly_files:
            # Extract entity ID from filename
            entity_id = file.stem.replace("weekly_", "")
            if entity_id == "entity_features":
                continue  # Skip aggregated file
            
            try:
                df = pd.read_csv(file)
                
                # Parse dates
                if 'Week_Start' in df.columns:
                    df['Week_Start'] = pd.to_datetime(df['Week_Start'])
                
                entities[entity_id] = EntityData(
                    entity_id=entity_id,
                    data=df
                )
                print(f"  ✓ Loaded {entity_id}: {len(df)} records")
            except Exception as e:
                print(f"  ✗ Failed to load {entity_id}: {e}")
        
        # Also try loading clean_transactions.csv for overall view
        clean_txn_path = self.data_dir / "clean_transactions.csv"
        if clean_txn_path.exists():
            try:
                self.transactions_df = pd.read_csv(clean_txn_path, low_memory=False)
                print(f"  ✓ Loaded transactions: {len(self.transactions_df)} records")
            except Exception as e:
                self.transactions_df = None
                print(f"  ✗ Failed to load transactions: {e}")
        
        return entities
    
    def load_entity(self, entity_id: str) -> Optional[EntityData]:
        """Load data for a specific entity."""
        file_path = self.data_dir / f"weekly_{entity_id}.csv"
        
        if not file_path.exists():
            print(f"  ✗ File not found: {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path)
            if 'Week_Start' in df.columns:
                df['Week_Start'] = pd.to_datetime(df['Week_Start'])
            
            return EntityData(entity_id=entity_id, data=df)
        except Exception as e:
            print(f"  ✗ Failed to load {entity_id}: {e}")
            return None


# =============================================================================
# ANOMALY DETECTION ENGINE
# =============================================================================
class AnomalyDetectionEngine:
    """Main engine for running anomaly detection."""
    
    def __init__(self):
        """
        Initialize the detection engine.
        """
        # Initialize knowledge base
        kb_path = Path(__file__).parent / 'data' / 'knowledge_base.db'
        self.kb = KnowledgeBase(str(kb_path))
        
        # Initialize meta-coordinator
        self.coordinator = MetaCoordinator(
            knowledge_base=self.kb,
            enable_parallel=True,
            max_workers=4
        )
        
        # Get system status
        self.system_status = self.coordinator.get_system_status()
    
    def train_agents(self, entities: Dict[str, 'EntityData']) -> Dict[str, Any]:
        """
        Train pattern-based agents on historical entity data.
        
        Args:
            entities: Dictionary of EntityData objects
        
        Returns:
            Training results per entity
        """
        # Combine all entity data for training
        all_data = []
        for entity_id, entity_data in entities.items():
            df = entity_data.data.copy()
            if 'Entity' not in df.columns:
                df['Entity'] = entity_id
            all_data.append(df)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return self.coordinator.train_agents(combined_data)
        
        return {}
    
    def run_detection(self, entity_data: EntityData) -> EnsembleVerdict:
        """
        Run anomaly detection on entity data.
        
        Args:
            entity_data: EntityData object with data
        
        Returns:
            EnsembleVerdict with detection results
        """
        # Prepare data
        data = prepare_detection_data(entity_data.data)
        
        # Run detection
        verdict = self.coordinator.run_detection(
            data=data,
            entity=entity_data.entity_id
        )
        
        return verdict
    
    def analyze_all_entities(self, entities: Dict[str, EntityData], 
                             progress_callback=None) -> Dict[str, EntityData]:
        """
        Run detection on all entities.
        
        Args:
            entities: Dictionary of EntityData objects
            progress_callback: Optional callback(entity_id, progress_pct)
        
        Returns:
            Updated entities dictionary with verdicts
        """
        total = len(entities)
        
        for i, (entity_id, entity_data) in enumerate(entities.items()):
            print(f"\n  Analyzing {entity_id}...")
            
            try:
                verdict = self.run_detection(entity_data)
                entity_data.verdict = verdict
                
                # Generate summary
                entity_data.summary = self._generate_entity_summary(entity_data, verdict)
                
                status = "⚠️ ANOMALY" if verdict.is_anomaly else "✓ NORMAL"
                print(f"    {status} - Confidence: {verdict.final_confidence:.1%}")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                entity_data.verdict = None
            
            if progress_callback:
                progress_callback(entity_id, (i + 1) / total * 100)
        
        return entities
    
    def _generate_entity_summary(self, entity_data: EntityData, 
                                  verdict: EnsembleVerdict) -> Dict[str, Any]:
        """Generate summary statistics for an entity."""
        df = entity_data.data
        
        summary = {
            'entity_id': entity_data.entity_id,
            'total_weeks': len(df),
            'date_range': {
                'start': df['Week_Start'].min().strftime('%Y-%m-%d') if 'Week_Start' in df.columns else 'N/A',
                'end': df['Week_Start'].max().strftime('%Y-%m-%d') if 'Week_Start' in df.columns else 'N/A'
            },
            'is_anomaly': verdict.is_anomaly,
            'confidence': verdict.final_confidence,
            'severity': verdict.severity.value,
            'num_flags': len(verdict.primary_flags) + len(verdict.secondary_flags),
            'primary_flags': len(verdict.primary_flags),
            'agent_results': {},
            'recommended_action': verdict.recommended_action,
        }
        
        # Financial metrics - check for both naming conventions
        net_col = 'Total_Net' if 'Total_Net' in df.columns else ('Net' if 'Net' in df.columns else None)
        inflow_col = 'Total_Inflow' if 'Total_Inflow' in df.columns else ('Inflow' if 'Inflow' in df.columns else None)
        outflow_col = 'Total_Outflow' if 'Total_Outflow' in df.columns else ('Outflow' if 'Outflow' in df.columns else None)
        
        if net_col:
            summary['financial'] = {
                'total_net': float(df[net_col].sum()),
                'avg_net': float(df[net_col].mean()),
                'std_net': float(df[net_col].std()),
                'min_net': float(df[net_col].min()),
                'max_net': float(df[net_col].max()),
            }
        
        if inflow_col and outflow_col:
            summary['financial']['total_inflow'] = float(df[inflow_col].sum())
            summary['financial']['total_outflow'] = float(df[outflow_col].sum())
        
        # Agent breakdown
        for result in verdict.agent_results:
            summary['agent_results'][result.agent_type] = {
                'flags': len(result.flags),
                'confidence': result.confidence,
                'execution_time_ms': result.execution_time_ms
            }
        
        return summary


# =============================================================================
# DASHBOARD GENERATOR
# =============================================================================
class DashboardGenerator:
    """Generates interactive HTML dashboard."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize dashboard generator.
        
        Args:
            output_dir: Output directory for dashboard
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent / "outputs" / "dashboards"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, entities: Dict[str, EntityData], 
                 system_status: Dict[str, Any]) -> str:
        """
        Generate interactive dashboard.
        
        Args:
            entities: Dictionary of EntityData with verdicts
            system_status: System status from coordinator
        
        Returns:
            Path to generated dashboard
        """
        # Calculate overall statistics
        overall_stats = self._calculate_overall_stats(entities)
        
        # Generate HTML
        html = self._generate_html(entities, overall_stats, system_status)
        
        # Save dashboard
        output_path = self.output_dir / "anomaly_detection_dashboard.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(output_path)
    
    def _calculate_overall_stats(self, entities: Dict[str, EntityData]) -> Dict[str, Any]:
        """Calculate overall statistics across all entities."""
        total_entities = len(entities)
        entities_with_anomalies = sum(
            1 for e in entities.values() 
            if e.verdict and e.verdict.is_anomaly
        )
        
        total_flags = sum(
            len(e.verdict.primary_flags) + len(e.verdict.secondary_flags)
            for e in entities.values() if e.verdict
        )
        
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        agent_flag_counts = {}
        
        for entity in entities.values():
            if not entity.verdict:
                continue
            
            for flag in entity.verdict.primary_flags + entity.verdict.secondary_flags:
                severity_counts[flag.severity.value] += 1
                
                agent_type = flag.agent_id.split('_')[0] if '_' in flag.agent_id else flag.agent_id
                agent_flag_counts[agent_type] = agent_flag_counts.get(agent_type, 0) + 1
        
        return {
            'total_entities': total_entities,
            'entities_with_anomalies': entities_with_anomalies,
            'anomaly_rate': entities_with_anomalies / total_entities if total_entities > 0 else 0,
            'total_flags': total_flags,
            'severity_counts': severity_counts,
            'agent_flag_counts': agent_flag_counts,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _generate_html(self, entities: Dict[str, EntityData],
                       overall_stats: Dict[str, Any],
                       system_status: Dict[str, Any]) -> str:
        """Generate the complete HTML dashboard."""
        
        # Build entity sections
        entity_sections = ""
        for entity_id, entity_data in sorted(entities.items()):
            if entity_data.verdict:
                entity_sections += self._generate_entity_section(entity_data)
        
        # Build agent overview
        agent_overview = self._generate_agent_overview(system_status)
        
        # Build severity chart data
        severity_data = self._generate_severity_chart_data(overall_stats)
        
        # Build agent chart data
        agent_chart_data = self._generate_agent_chart_data(overall_stats)
        
        # Build entity navigation
        nav_links = self._generate_nav_links(entities)
        
        # Generate timeline data
        timeline_data = self._generate_timeline_data(entities)
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Agent Anomaly Detection Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --az-mulberry: {AZColors.MULBERRY};
            --az-lime: {AZColors.LIME};
            --az-navy: {AZColors.NAVY};
            --az-graphite: {AZColors.GRAPHITE};
            --az-light-blue: {AZColors.LIGHT_BLUE};
            --az-magenta: {AZColors.MAGENTA};
            --az-purple: {AZColors.PURPLE};
            --az-gold: {AZColors.GOLD};
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            color: var(--az-graphite);
            line-height: 1.6;
        }}

        .dashboard-header {{
            background: linear-gradient(135deg, var(--az-mulberry) 0%, var(--az-purple) 100%);
            color: white;
            padding: 2rem 3rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}

        .header-content {{
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .dashboard-title {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}

        .dashboard-subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 300;
        }}

        .header-stats {{
            display: flex;
            gap: 2rem;
        }}

        .header-stat {{
            text-align: center;
            padding: 0.5rem 1.5rem;
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
        }}

        .header-stat-value {{
            font-size: 2rem;
            font-weight: 700;
        }}

        .header-stat-label {{
            font-size: 0.85rem;
            opacity: 0.8;
        }}

        .nav-bar {{
            background: white;
            padding: 1rem 3rem;
            border-bottom: 3px solid var(--az-mulberry);
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .nav-content {{
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: center;
        }}

        .nav-label {{
            font-weight: 600;
            color: var(--az-mulberry);
            margin-right: 1rem;
        }}

        .nav-link {{
            padding: 0.5rem 1.2rem;
            background: #f1f3f4;
            border-radius: 25px;
            text-decoration: none;
            color: var(--az-graphite);
            font-weight: 500;
            transition: all 0.3s ease;
            cursor: pointer;
        }}

        .nav-link:hover {{
            background: var(--az-mulberry);
            color: white;
            transform: translateY(-2px);
        }}

        .nav-link.anomaly {{
            background: rgba(208, 0, 111, 0.1);
            border: 2px solid var(--az-magenta);
        }}

        .nav-link.anomaly:hover {{
            background: var(--az-magenta);
        }}

        .main-content {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
        }}

        /* Executive Summary */
        .executive-summary {{
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}

        .summary-title {{
            font-size: 1.5rem;
            color: var(--az-mulberry);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
        }}

        .summary-card {{
            text-align: center;
            padding: 1.5rem;
            border-radius: 12px;
            background: linear-gradient(145deg, #f8f9fa, #ffffff);
            border: 1px solid #e9ecef;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .summary-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}

        .summary-value {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}

        .summary-label {{
            font-size: 0.9rem;
            color: #6c757d;
            font-weight: 500;
        }}

        .severity-critical {{ color: var(--az-magenta); }}
        .severity-high {{ color: var(--az-mulberry); }}
        .severity-medium {{ color: var(--az-gold); }}
        .severity-low {{ color: var(--az-light-blue); }}

        /* Charts Section */
        .charts-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .chart-container {{
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}

        .chart-container.full-width {{
            grid-column: 1 / -1;
        }}

        .chart-container h3 {{
            font-size: 1.1rem;
            color: var(--az-navy);
            margin-bottom: 1rem;
        }}

        .chart {{
            width: 100%;
            min-height: 300px;
        }}

        /* Agent Overview */
        .agent-section {{
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}

        .agent-section h2 {{
            font-size: 1.5rem;
            color: var(--az-mulberry);
            margin-bottom: 1.5rem;
        }}

        .agent-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }}

        .agent-card {{
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid var(--az-mulberry);
            background: linear-gradient(145deg, #f8f9fa, #ffffff);
            transition: transform 0.3s ease;
        }}

        .agent-card:hover {{
            transform: translateX(5px);
        }}

        .agent-card h4 {{
            color: var(--az-navy);
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .agent-card p {{
            font-size: 0.9rem;
            color: #6c757d;
            margin-bottom: 0.75rem;
        }}

        .agent-stats {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }}

        .agent-stat {{
            font-size: 0.85rem;
            padding: 0.25rem 0.75rem;
            background: rgba(131, 0, 81, 0.1);
            border-radius: 15px;
            color: var(--az-mulberry);
        }}

        /* Entity Section */
        .entity-section {{
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}

        .entity-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .entity-header h2 {{
            font-size: 1.75rem;
            color: var(--az-graphite);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .entity-badge {{
            background: var(--az-mulberry);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 8px;
            font-size: 1.2rem;
        }}

        .status-badge {{
            padding: 0.5rem 1.25rem;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.9rem;
        }}

        .status-anomaly {{
            background: linear-gradient(135deg, var(--az-magenta), var(--az-mulberry));
            color: white;
        }}

        .status-normal {{
            background: linear-gradient(135deg, var(--az-lime), #98c93c);
            color: var(--az-graphite);
        }}

        /* KPI Cards */
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}

        .kpi-card {{
            background: linear-gradient(145deg, #ffffff, #f8f9fa);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            border: 1px solid #e9ecef;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}

        .kpi-icon {{
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }}

        .kpi-value {{
            font-size: 1.4rem;
            font-weight: 700;
        }}

        .kpi-label {{
            font-size: 0.8rem;
            color: #6c757d;
            margin-top: 0.25rem;
        }}

        /* Flags List */
        .flags-section {{
            margin-top: 1.5rem;
        }}

        .flags-section h3 {{
            font-size: 1.1rem;
            color: var(--az-navy);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e9ecef;
        }}

        .flag-card {{
            padding: 1rem 1.25rem;
            border-radius: 10px;
            margin-bottom: 0.75rem;
            border-left: 4px solid var(--az-mulberry);
            background: #f8f9fa;
            transition: transform 0.2s ease;
        }}

        .flag-card:hover {{
            transform: translateX(5px);
        }}

        .flag-card.severity-critical {{ border-left-color: var(--az-magenta); }}
        .flag-card.severity-high {{ border-left-color: var(--az-mulberry); }}
        .flag-card.severity-medium {{ border-left-color: var(--az-gold); }}
        .flag-card.severity-low {{ border-left-color: var(--az-light-blue); }}

        .flag-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.5rem;
        }}

        .flag-title {{
            font-weight: 600;
            color: var(--az-graphite);
        }}

        .flag-severity {{
            font-size: 0.75rem;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .flag-severity.critical {{
            background: var(--az-magenta);
            color: white;
        }}

        .flag-severity.high {{
            background: var(--az-mulberry);
            color: white;
        }}

        .flag-severity.medium {{
            background: var(--az-gold);
            color: var(--az-graphite);
        }}

        .flag-severity.low {{
            background: var(--az-light-blue);
            color: var(--az-graphite);
        }}

        .flag-description {{
            font-size: 0.9rem;
            color: #555;
            margin-bottom: 0.5rem;
        }}

        .flag-meta {{
            display: flex;
            gap: 1.5rem;
            font-size: 0.8rem;
            color: #6c757d;
        }}

        .flag-meta span {{
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }}

        /* Recommendation Box */
        .recommendation-box {{
            background: linear-gradient(135deg, rgba(131, 0, 81, 0.05), rgba(60, 16, 83, 0.05));
            border: 1px solid rgba(131, 0, 81, 0.2);
            border-radius: 12px;
            padding: 1.25rem;
            margin-top: 1.5rem;
        }}

        .recommendation-box h4 {{
            color: var(--az-mulberry);
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }}

        .recommendation-box p {{
            color: var(--az-graphite);
            font-size: 0.95rem;
        }}

        /* Footer */
        .dashboard-footer {{
            text-align: center;
            padding: 2rem;
            color: #6c757d;
            font-size: 0.85rem;
        }}

        .dashboard-footer a {{
            color: var(--az-mulberry);
            text-decoration: none;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .dashboard-header {{
                padding: 1.5rem;
            }}
            
            .dashboard-title {{
                font-size: 1.75rem;
            }}
            
            .header-stats {{
                flex-direction: column;
                gap: 0.75rem;
            }}
            
            .nav-bar {{
                padding: 0.75rem 1rem;
            }}
            
            .main-content {{
                padding: 1rem;
            }}
            
            .charts-row {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <!-- Header -->
    <header class="dashboard-header">
        <div class="header-content">
            <div>
                <h1 class="dashboard-title">🔍 Multi-Agent Anomaly Detection</h1>
                <p class="dashboard-subtitle">AstraZeneca Cash Flow Analysis | Generated: {overall_stats['generated_at']}</p>
            </div>
            <div class="header-stats">
                <div class="header-stat">
                    <div class="header-stat-value">{overall_stats['total_entities']}</div>
                    <div class="header-stat-label">Entities Analyzed</div>
                </div>
                <div class="header-stat">
                    <div class="header-stat-value">{overall_stats['entities_with_anomalies']}</div>
                    <div class="header-stat-label">With Anomalies</div>
                </div>
                <div class="header-stat">
                    <div class="header-stat-value">{overall_stats['total_flags']}</div>
                    <div class="header-stat-label">Total Flags</div>
                </div>
            </div>
        </div>
    </header>

    <!-- Navigation -->
    <nav class="nav-bar">
        <div class="nav-content">
            <span class="nav-label">🏢 Entities:</span>
            {nav_links}
        </div>
    </nav>

    <!-- Main Content -->
    <main class="main-content">
        <!-- Executive Summary -->
        <section class="executive-summary">
            <h2 class="summary-title">📊 Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-value" style="color: var(--az-mulberry);">{overall_stats['anomaly_rate']:.1%}</div>
                    <div class="summary-label">Anomaly Rate</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value severity-critical">{overall_stats['severity_counts']['critical']}</div>
                    <div class="summary-label">Critical Flags</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value severity-high">{overall_stats['severity_counts']['high']}</div>
                    <div class="summary-label">High Severity</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value severity-medium">{overall_stats['severity_counts']['medium']}</div>
                    <div class="summary-label">Medium Severity</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value severity-low">{overall_stats['severity_counts']['low']}</div>
                    <div class="summary-label">Low Severity</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value" style="color: var(--az-navy);">{system_status.get('active_agents', 6)}</div>
                    <div class="summary-label">Active Agents</div>
                </div>
            </div>
        </section>

        <!-- Charts Row -->
        <div class="charts-row">
            <div class="chart-container">
                <h3>🎯 Severity Distribution</h3>
                <div id="severityChart" class="chart"></div>
            </div>
            <div class="chart-container">
                <h3>🤖 Flags by Agent</h3>
                <div id="agentChart" class="chart"></div>
            </div>
        </div>

        <!-- Detection Timeline -->
        <div class="charts-row">
            <div class="chart-container full-width">
                <h3>📈 Entity Detection Overview</h3>
                <div id="timelineChart" class="chart" style="min-height: 400px;"></div>
            </div>
        </div>

        <!-- Agent Overview -->
        {agent_overview}

        <!-- Entity Sections -->
        {entity_sections}

    </main>

    <!-- Footer -->
    <footer class="dashboard-footer">
        <p>Multi-Agent Anomaly Detection System | AstraZeneca Datathon 2025</p>
        <p>Powered by 6 specialized detection agents with ensemble decision-making</p>
    </footer>

    <!-- Plotly Charts -->
    <script>
        // AstraZeneca Color Palette
        const AZ_COLORS = {{
            mulberry: '{AZColors.MULBERRY}',
            lime: '{AZColors.LIME}',
            navy: '{AZColors.NAVY}',
            graphite: '{AZColors.GRAPHITE}',
            lightBlue: '{AZColors.LIGHT_BLUE}',
            magenta: '{AZColors.MAGENTA}',
            purple: '{AZColors.PURPLE}',
            gold: '{AZColors.GOLD}'
        }};

        // Severity Chart
        {severity_data}

        // Agent Chart
        {agent_chart_data}

        // Timeline Chart
        {timeline_data}
    </script>
</body>
</html>'''
        
        return html
    
    def _generate_nav_links(self, entities: Dict[str, EntityData]) -> str:
        """Generate navigation links for entities."""
        links = []
        
        for entity_id in sorted(entities.keys()):
            entity = entities[entity_id]
            is_anomaly = entity.verdict and entity.verdict.is_anomaly
            
            css_class = "nav-link anomaly" if is_anomaly else "nav-link"
            icon = "⚠️" if is_anomaly else "✓"
            
            links.append(
                f'<a href="#{entity_id}" class="{css_class}">{icon} {entity_id}</a>'
            )
        
        return "\n            ".join(links)
    
    def _generate_agent_overview(self, system_status: Dict[str, Any]) -> str:
        """Generate agent overview section."""
        agents_html = ""
        
        agent_info = {
            'statistical': {
                'icon': '📊',
                'name': 'Statistical Agent',
                'description': 'Detects outliers using Z-scores, IQR, and rolling statistics'
            },
            'pattern': {
                'icon': '🔄',
                'name': 'Pattern Agent',
                'description': 'Identifies violations in seasonal and temporal patterns'
            },
            'rule': {
                'icon': '📋',
                'name': 'Rule Agent',
                'description': 'Enforces business rules for transactions and amounts'
            },
            'temporal': {
                'icon': '⏰',
                'name': 'Temporal Agent',
                'description': 'Analyzes week-over-week and month-over-month changes'
            },
            'category': {
                'icon': '🏷️',
                'name': 'Category Agent',
                'description': 'Monitors category-specific patterns (AP, AR, Payroll, etc.)'
            }
        }
        
        agents_data = system_status.get('agents', {})
        
        for agent_type, info in agent_info.items():
            agent_status = agents_data.get(agent_type, {})
            confidence = agent_status.get('confidence', 0.5)
            detections = agent_status.get('total_detections', 0)
            rule_graphs = agent_status.get('rule_graphs', 0)
            
            color = AZColors.AGENTS.get(agent_type, AZColors.MULBERRY)
            
            agents_html += f'''
            <div class="agent-card" style="border-left-color: {color};">
                <h4>{info['icon']} {info['name']}</h4>
                <p>{info['description']}</p>
                <div class="agent-stats">
                    <span class="agent-stat">Confidence: {confidence:.0%}</span>
                    <span class="agent-stat">{detections} detections</span>
                    <span class="agent-stat">{rule_graphs} rule graphs</span>
                </div>
            </div>'''
        
        return f'''
        <section class="agent-section">
            <h2>🤖 Detection Agents</h2>
            <div class="agent-grid">
                {agents_html}
            </div>
        </section>'''
    
    def _generate_entity_section(self, entity_data: EntityData) -> str:
        """Generate HTML section for a single entity."""
        verdict = entity_data.verdict
        summary = entity_data.summary
        
        status_class = "status-anomaly" if verdict.is_anomaly else "status-normal"
        status_text = "⚠️ ANOMALY DETECTED" if verdict.is_anomaly else "✓ NORMAL"
        
        # Generate flag cards
        flags_html = ""
        all_flags = verdict.primary_flags + verdict.secondary_flags
        
        if all_flags:
            for flag in all_flags[:10]:  # Limit to 10 flags
                flags_html += f'''
                <div class="flag-card severity-{flag.severity.value}">
                    <div class="flag-header">
                        <span class="flag-title">{flag.metric_name}</span>
                        <span class="flag-severity {flag.severity.value}">{flag.severity.value}</span>
                    </div>
                    <div class="flag-description">{flag.description}</div>
                    <div class="flag-meta">
                        <span>🤖 {flag.agent_id}</span>
                        <span>📊 Confidence: {flag.confidence:.1%}</span>
                        <span>📍 Value: {flag.metric_value:,.2f}</span>
                    </div>
                </div>'''
        else:
            flags_html = '<p style="color: #6c757d; padding: 1rem;">No anomalies detected by any agent.</p>'
        
        # Financial metrics
        financial = summary.get('financial', {})
        total_net = financial.get('total_net', 0)
        avg_net = financial.get('avg_net', 0)
        total_inflow = financial.get('total_inflow', 0)
        total_outflow = financial.get('total_outflow', 0)
        
        # Agent results KPIs
        agent_results = summary.get('agent_results', {})
        agents_detecting = sum(1 for a in agent_results.values() if a.get('flags', 0) > 0)
        
        return f'''
        <section class="entity-section" id="{entity_data.entity_id}">
            <div class="entity-header">
                <h2>
                    <span class="entity-badge">{entity_data.entity_id}</span>
                    Cash Flow Anomaly Analysis
                </h2>
                <span class="{status_class}">{status_text}</span>
            </div>
            
            <div class="kpi-row">
                <div class="kpi-card">
                    <div class="kpi-icon">🎯</div>
                    <div class="kpi-value" style="color: var(--az-mulberry);">{verdict.final_confidence:.1%}</div>
                    <div class="kpi-label">Confidence</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon">🚨</div>
                    <div class="kpi-value">{len(verdict.primary_flags)}</div>
                    <div class="kpi-label">Primary Flags</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon">🤖</div>
                    <div class="kpi-value">{agents_detecting}/{len(agent_results)}</div>
                    <div class="kpi-label">Agents Detecting</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon">📅</div>
                    <div class="kpi-value">{summary.get('total_weeks', 0)}</div>
                    <div class="kpi-label">Weeks Analyzed</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon">💰</div>
                    <div class="kpi-value">${total_net/1e6:.1f}M</div>
                    <div class="kpi-label">Total Net Flow</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon">📈</div>
                    <div class="kpi-value">${total_inflow/1e6:.1f}M</div>
                    <div class="kpi-label">Total Inflow</div>
                </div>
            </div>
            
            <div class="flags-section">
                <h3>🚩 Detected Anomaly Flags</h3>
                {flags_html}
            </div>
            
            <div class="recommendation-box">
                <h4>📋 Recommended Action</h4>
                <p>{verdict.recommended_action}</p>
            </div>
        </section>'''
    
    def _generate_severity_chart_data(self, overall_stats: Dict[str, Any]) -> str:
        """Generate Plotly data for severity chart."""
        counts = overall_stats['severity_counts']
        
        return f'''
        Plotly.newPlot('severityChart', [{{
            values: [{counts['critical']}, {counts['high']}, {counts['medium']}, {counts['low']}],
            labels: ['Critical', 'High', 'Medium', 'Low'],
            type: 'pie',
            hole: 0.5,
            marker: {{
                colors: ['{AZColors.MAGENTA}', '{AZColors.MULBERRY}', '{AZColors.GOLD}', '{AZColors.LIGHT_BLUE}']
            }},
            textinfo: 'label+value',
            textposition: 'outside'
        }}], {{
            showlegend: true,
            legend: {{ orientation: 'h', y: -0.1 }},
            margin: {{ t: 20, b: 40, l: 20, r: 20 }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }}, {{ responsive: true }});
        '''
    
    def _generate_agent_chart_data(self, overall_stats: Dict[str, Any]) -> str:
        """Generate Plotly data for agent chart."""
        agent_counts = overall_stats['agent_flag_counts']
        
        agents = list(agent_counts.keys()) if agent_counts else ['No Data']
        counts = list(agent_counts.values()) if agent_counts else [0]
        colors = [AZColors.AGENTS.get(a, AZColors.MULBERRY) for a in agents]
        
        return f'''
        Plotly.newPlot('agentChart', [{{
            x: {json.dumps(agents)},
            y: {json.dumps(counts)},
            type: 'bar',
            marker: {{
                color: {json.dumps(colors)},
                line: {{ color: '{AZColors.GRAPHITE}', width: 1 }}
            }}
        }}], {{
            margin: {{ t: 20, b: 60, l: 50, r: 20 }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            yaxis: {{ title: 'Number of Flags', gridcolor: '#e9ecef' }},
            xaxis: {{ tickangle: -30 }}
        }}, {{ responsive: true }});
        '''
    
    def _generate_timeline_data(self, entities: Dict[str, EntityData]) -> str:
        """Generate Plotly data for entity timeline chart."""
        entity_ids = []
        confidences = []
        flag_counts = []
        colors = []
        statuses = []
        
        for entity_id in sorted(entities.keys()):
            entity = entities[entity_id]
            if entity.verdict:
                entity_ids.append(entity_id)
                confidences.append(entity.verdict.final_confidence * 100)
                flag_counts.append(len(entity.verdict.primary_flags) + len(entity.verdict.secondary_flags))
                
                if entity.verdict.is_anomaly:
                    colors.append(AZColors.MAGENTA)
                    statuses.append('Anomaly')
                else:
                    colors.append(AZColors.LIME)
                    statuses.append('Normal')
        
        return f'''
        Plotly.newPlot('timelineChart', [
            {{
                x: {json.dumps(entity_ids)},
                y: {json.dumps(confidences)},
                type: 'bar',
                name: 'Confidence %',
                marker: {{ color: {json.dumps(colors)}, opacity: 0.8 }},
                text: {json.dumps(statuses)},
                hovertemplate: '<b>%{{x}}</b><br>Confidence: %{{y:.1f}}%<br>Status: %{{text}}<extra></extra>'
            }},
            {{
                x: {json.dumps(entity_ids)},
                y: {json.dumps(flag_counts)},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Flag Count',
                yaxis: 'y2',
                line: {{ color: '{AZColors.NAVY}', width: 3 }},
                marker: {{ size: 10, color: '{AZColors.NAVY}' }}
            }}
        ], {{
            margin: {{ t: 30, b: 80, l: 60, r: 60 }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            yaxis: {{ 
                title: 'Confidence %', 
                gridcolor: '#e9ecef',
                range: [0, 100]
            }},
            yaxis2: {{
                title: 'Number of Flags',
                overlaying: 'y',
                side: 'right',
                gridcolor: 'rgba(0,0,0,0)'
            }},
            xaxis: {{ tickangle: -45 }},
            legend: {{ orientation: 'h', y: 1.15 }},
            barmode: 'group'
        }}, {{ responsive: true }});
        '''


# =============================================================================
# MAIN PIPELINE
# =============================================================================
class AnomalyDetectionPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, data_dir: str = None, output_dir: str = None):
        """
        Initialize the pipeline.
        
        Args:
            data_dir: Path to processed data directory
            output_dir: Path to output directory
        """
        self.data_processor = DataProcessor(data_dir)
        self.detection_engine = AnomalyDetectionEngine()
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent / "outputs"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "dashboards").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        self.dashboard_generator = DashboardGenerator(
            str(self.output_dir / "dashboards")
        )
    
    def run(self, entity_ids: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            entity_ids: Optional list of specific entities to analyze
        
        Returns:
            Pipeline results dictionary
        """
        start_time = datetime.now()
        
        print("=" * 70)
        print("MULTI-AGENT ANOMALY DETECTION PIPELINE")
        print("=" * 70)
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Step 1: Load Data
        print("📂 STEP 1: Loading Data")
        print("-" * 40)
        
        if entity_ids:
            entities = {}
            for eid in entity_ids:
                entity = self.data_processor.load_entity(eid)
                if entity:
                    entities[eid] = entity
        else:
            entities = self.data_processor.load_all_entities()
        
        if not entities:
            print("❌ No entities loaded. Check data directory.")
            return {'success': False, 'error': 'No data loaded'}
        
        print(f"\n✓ Loaded {len(entities)} entities")
        print()
        
        # Step 2: Train Pattern Agents
        print("🧠 STEP 2: Training Pattern Agents")
        print("-" * 40)
        
        training_results = self.detection_engine.train_agents(entities)
        # Count trained entities from nested pattern results
        pattern_results = training_results.get('pattern', {})
        trained_count = sum(
            1 for v in pattern_results.values() 
            if isinstance(v, dict) and v.get('status') == 'trained'
        )
        print(f"  ✓ Trained patterns for {trained_count}/{len(entities)} entities")
        print()
        
        # Step 3: Run Detection
        print("🔍 STEP 3: Running Multi-Agent Detection")
        print("-" * 40)
        
        entities = self.detection_engine.analyze_all_entities(entities)
        
        # Count results
        anomaly_count = sum(
            1 for e in entities.values() 
            if e.verdict and e.verdict.is_anomaly
        )
        
        print(f"\n✓ Detection complete: {anomaly_count}/{len(entities)} entities with anomalies")
        print()
        
        # Step 4: Generate Dashboard
        print("📊 STEP 4: Generating Interactive Dashboard")
        print("-" * 40)
        
        dashboard_path = self.dashboard_generator.generate(
            entities, 
            self.detection_engine.system_status
        )
        
        print(f"✓ Dashboard saved: {dashboard_path}")
        print()
        
        # Step 5: Save Reports
        print("📝 STEP 5: Saving Reports")
        print("-" * 40)
        
        report_paths = self._save_reports(entities)
        
        for path in report_paths:
            print(f"  ✓ {path}")
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print()
        print("=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Duration: {duration:.1f} seconds")
        print(f"Entities Analyzed: {len(entities)}")
        print(f"Anomalies Detected: {anomaly_count}")
        print(f"Dashboard: {dashboard_path}")
        print()
        
        return {
            'success': True,
            'entities_analyzed': len(entities),
            'anomalies_detected': anomaly_count,
            'dashboard_path': dashboard_path,
            'report_paths': report_paths,
            'duration_seconds': duration
        }
    
    def _save_reports(self, entities: Dict[str, EntityData]) -> List[str]:
        """Save individual entity reports."""
        report_paths = []
        reports_dir = self.output_dir / "reports"
        
        # Save summary JSON
        summary_data = {
            'generated_at': datetime.now().isoformat(),
            'entities': {}
        }
        
        for entity_id, entity in entities.items():
            if entity.verdict:
                summary_data['entities'][entity_id] = {
                    'is_anomaly': entity.verdict.is_anomaly,
                    'confidence': entity.verdict.final_confidence,
                    'severity': entity.verdict.severity.value,
                    'num_flags': len(entity.verdict.primary_flags) + len(entity.verdict.secondary_flags),
                    'recommended_action': entity.verdict.recommended_action
                }
        
        summary_path = reports_dir / "detection_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        report_paths.append(str(summary_path))
        
        # Save detailed verdicts
        verdicts_data = {}
        for entity_id, entity in entities.items():
            if entity.verdict:
                verdicts_data[entity_id] = entity.verdict.to_dict()
        
        verdicts_path = reports_dir / "detailed_verdicts.json"
        with open(verdicts_path, 'w') as f:
            json.dump(verdicts_data, f, indent=2, default=str)
        report_paths.append(str(verdicts_path))
        
        return report_paths


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Anomaly Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_full_pipeline.py
    python run_full_pipeline.py --entities ID10,MY10,TH10
    python run_full_pipeline.py --data-dir ./custom_data --output-dir ./custom_output
        """
    )
    
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Path to output directory')
    parser.add_argument('--entities', type=str, default=None,
                        help='Comma-separated list of entity IDs to analyze')
    
    args = parser.parse_args()
    
    # Parse entity list
    entity_ids = None
    if args.entities:
        entity_ids = [e.strip() for e in args.entities.split(',')]
    
    # Create and run pipeline
    pipeline = AnomalyDetectionPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    result = pipeline.run(entity_ids=entity_ids)
    
    if result['success']:
        print("🎉 Pipeline completed successfully!")
        print(f"\n📊 Open the dashboard: {result['dashboard_path']}")
    else:
        print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
