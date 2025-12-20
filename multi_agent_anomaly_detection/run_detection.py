"""
Main Entry Point for Multi-Agent Anomaly Detection
===================================================

This script provides the main interface for running the anomaly detection
system on cash flow data.

Usage:
    python run_detection.py --data path/to/data.csv --entity ID10
    python run_detection.py --train --data path/to/training_data.csv
    python run_detection.py --status
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_anomaly_detection.coordination.meta_coordinator import MetaCoordinator
from multi_agent_anomaly_detection.core.knowledge_base import KnowledgeBase
from multi_agent_anomaly_detection.utils.helpers import prepare_detection_data, generate_report


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from file."""
    path = Path(data_path)
    
    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix == '.xlsx':
        return pd.read_excel(path)
    elif path.suffix == '.json':
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def run_detection(args):
    """Run anomaly detection on data."""
    print("=" * 60)
    print("MULTI-AGENT ANOMALY DETECTION SYSTEM")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize knowledge base
    kb_path = Path(__file__).parent / 'data' / 'knowledge_base.db'
    kb = KnowledgeBase(str(kb_path))
    
    # Initialize coordinator
    print("Initializing detection agents...")
    coordinator = MetaCoordinator(knowledge_base=kb)
    
    # Get system status
    status = coordinator.get_system_status()
    print(f"Active agents: {status['active_agents']}")
    print()
    
    # Load data
    print(f"Loading data from: {args.data}")
    raw_data = load_data(args.data)
    print(f"Loaded {len(raw_data)} records")
    
    # Prepare data
    data = prepare_detection_data(raw_data)
    
    # Determine entity
    entity = args.entity
    if entity is None and 'Entity' in data.columns:
        entities = data['Entity'].unique()
        if len(entities) == 1:
            entity = entities[0]
        else:
            print(f"Multiple entities found: {list(entities)}")
            print("Please specify --entity or run on single-entity data")
            return
    
    print(f"Analyzing entity: {entity}")
    print()
    
    # Run detection
    print("Running detection agents...")
    verdict = coordinator.run_detection(data, entity=entity)
    
    # Generate report
    print()
    report = coordinator.explain_verdict(verdict)
    print(report)
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        
        if args.format == 'json':
            with open(output_path, 'w') as f:
                json.dump(verdict.to_dict(), f, indent=2, default=str)
        elif args.format == 'html':
            html_report = generate_report(verdict, format='html')
            with open(output_path, 'w') as f:
                f.write(html_report)
        else:
            with open(output_path, 'w') as f:
                f.write(report)
        
        print(f"\nResults saved to: {output_path}")
    
    # Return verdict for programmatic use
    return verdict


def run_training(args):
    """Train agents on historical data."""
    print("=" * 60)
    print("AGENT TRAINING MODE")
    print("=" * 60)
    print()
    
    # Initialize
    kb_path = Path(__file__).parent / 'data' / 'knowledge_base.db'
    kb = KnowledgeBase(str(kb_path))
    coordinator = MetaCoordinator(knowledge_base=kb)
    
    # Load training data
    print(f"Loading training data from: {args.data}")
    data = load_data(args.data)
    data = prepare_detection_data(data)
    print(f"Training samples: {len(data)}")
    print()
    
    # Train agents
    print("Training agents...")
    results = coordinator.train_agents(data)
    
    # Print results
    for agent_type, result in results.items():
        print(f"\n{agent_type.upper()} Agent:")
        if isinstance(result, dict):
            for entity, entity_result in result.items():
                if isinstance(entity_result, dict):
                    status = entity_result.get('status', 'unknown')
                    samples = entity_result.get('samples', 0)
                    print(f"  {entity}: {status} ({samples} samples)")
    
    print("\nTraining complete!")


def show_status(args):
    """Show system status."""
    print("=" * 60)
    print("SYSTEM STATUS")
    print("=" * 60)
    print()
    
    kb_path = Path(__file__).parent / 'data' / 'knowledge_base.db'
    kb = KnowledgeBase(str(kb_path))
    coordinator = MetaCoordinator(knowledge_base=kb)
    
    status = coordinator.get_system_status()
    
    print(f"Total Agents: {status['total_agents']}")
    print(f"Active Agents: {status['active_agents']}")
    print()
    
    print("Agent Details:")
    print("-" * 40)
    for agent_id, agent_status in status['agents'].items():
        print(f"  {agent_status['name']}")
        print(f"    Type: {agent_status['type']}")
        print(f"    Active: {agent_status['is_active']}")
        print(f"    Confidence: {agent_status['confidence']:.1%}")
        print(f"    Detections: {agent_status['total_detections']}")
        print(f"    Rule Graphs: {agent_status['rule_graphs']}")
        print()
    
    print("Knowledge Base:")
    print("-" * 40)
    kb_stats = status.get('knowledge_base', {})
    for key, value in kb_stats.items():
        print(f"  {key}: {value}")


def run_evolution(args):
    """Run rule evolution cycle."""
    print("=" * 60)
    print("RULE EVOLUTION CYCLE")
    print("=" * 60)
    print()
    
    kb_path = Path(__file__).parent / 'data' / 'knowledge_base.db'
    kb = KnowledgeBase(str(kb_path))
    coordinator = MetaCoordinator(knowledge_base=kb)
    
    print("Running evolution cycle...")
    results = coordinator.run_evolution_cycle()
    
    print(f"\nRules evaluated: {results['evaluated_rules']}")
    print(f"Mutations proposed: {results['mutations_proposed']}")
    print(f"Mutations applied: {results['mutations_applied']}")
    
    if results['details']:
        print("\nApplied Mutations:")
        for detail in results['details']:
            print(f"  - {detail['rule_id']}: {detail['mutation_type']}")
            print(f"    Reason: {detail['justification']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Anomaly Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_detection.py --data weekly_features.csv --entity ID10
  python run_detection.py --data data.csv --output report.html --format html
  python run_detection.py --train --data historical_data.csv
  python run_detection.py --status
  python run_detection.py --evolve
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--train', action='store_true',
                           help='Train agents on historical data')
    mode_group.add_argument('--status', action='store_true',
                           help='Show system status')
    mode_group.add_argument('--evolve', action='store_true',
                           help='Run rule evolution cycle')
    
    # Data options
    parser.add_argument('--data', '-d', type=str,
                       help='Path to input data file (CSV, XLSX, or JSON)')
    parser.add_argument('--entity', '-e', type=str,
                       help='Entity to analyze')
    
    # Output options
    parser.add_argument('--output', '-o', type=str,
                       help='Path to output file')
    parser.add_argument('--format', '-f', type=str,
                       choices=['text', 'json', 'html', 'markdown'],
                       default='text',
                       help='Output format (default: text)')
    
    args = parser.parse_args()
    
    # Route to appropriate function
    if args.status:
        show_status(args)
    elif args.train:
        if not args.data:
            parser.error("--train requires --data")
        run_training(args)
    elif args.evolve:
        run_evolution(args)
    else:
        if not args.data:
            parser.error("Please specify --data or use --status/--train/--evolve")
        run_detection(args)


if __name__ == "__main__":
    main()

