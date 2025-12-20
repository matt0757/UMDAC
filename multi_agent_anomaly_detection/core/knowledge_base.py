"""
Knowledge Base Implementation
=============================

Central store for all rule graphs, decision trees, agent configurations,
and detection context. Supports SQLite and JSON-based storage with
versioning and audit trails.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import uuid
import numpy as np

from .rule_graph import RuleGraph
from .models import AnomalyFlag, Feedback, RuleScore, Mutation


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class KnowledgeBase:
    """
    Central knowledge store for the multi-agent anomaly detection system.
    
    Features:
    - SQLite storage with JSON columns for flexible schema
    - Versioned rule graphs with audit trails
    - Agent configuration storage
    - Feedback and performance tracking
    - Caching for frequently accessed data
    """
    
    def __init__(self, db_path: str = None, use_json_fallback: bool = False):
        """
        Initialize the knowledge base.
        
        Args:
            db_path: Path to SQLite database. If None, uses in-memory database.
            use_json_fallback: If True, use JSON files instead of SQLite.
        """
        self.use_json = use_json_fallback
        
        if use_json_fallback:
            self.data_dir = Path(db_path) if db_path else Path("multi_agent_anomaly_detection/data")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._init_json_storage()
        else:
            self.db_path = db_path or ":memory:"
            if db_path:
                os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._init_database()
        
        # Cache for frequently accessed data
        self._cache: Dict[str, Any] = {}
    
    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        cursor = self.conn.cursor()
        
        # Rule graphs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rule_graphs (
                graph_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                graph_data TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                is_active INTEGER DEFAULT 1,
                entity TEXT,
                agent_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Rule graph history for versioning
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rule_graph_history (
                history_id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                graph_data TEXT NOT NULL,
                change_description TEXT,
                changed_by TEXT DEFAULT 'system',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (graph_id) REFERENCES rule_graphs(graph_id)
            )
        """)
        
        # Decision trees
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decision_trees (
                tree_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity TEXT,
                agent_type TEXT,
                tree_data TEXT NOT NULL,
                feature_names TEXT,
                performance_metrics TEXT,
                version INTEGER DEFAULT 1,
                is_active INTEGER DEFAULT 1,
                trained_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Anomaly flags
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_flags (
                flag_id TEXT PRIMARY KEY,
                transaction_id TEXT,
                entity TEXT NOT NULL,
                timestamp TEXT,
                anomaly_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL,
                agent_id TEXT,
                rule_id TEXT,
                flag_data TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Feedback
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                flag_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                reviewer TEXT,
                comments TEXT,
                correct_label TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (flag_id) REFERENCES anomaly_flags(flag_id)
            )
        """)
        
        # Rule scores / performance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rule_scores (
                score_id TEXT PRIMARY KEY,
                rule_id TEXT NOT NULL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                true_positives INTEGER,
                false_positives INTEGER,
                false_negatives INTEGER,
                total_activations INTEGER,
                evaluation_period_days INTEGER,
                evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Mutations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mutations (
                mutation_id TEXT PRIMARY KEY,
                rule_id TEXT NOT NULL,
                mutation_type TEXT NOT NULL,
                original_value TEXT,
                new_value TEXT,
                justification TEXT,
                confidence REAL,
                applied INTEGER DEFAULT 0,
                applied_at TEXT,
                rollback_available INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Agent configurations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_configs (
                agent_id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                name TEXT,
                config_data TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                priority INTEGER DEFAULT 50,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # External context cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS external_context (
                context_id TEXT PRIMARY KEY,
                context_type TEXT NOT NULL,
                context_data TEXT NOT NULL,
                valid_until TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flags_entity ON anomaly_flags(entity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_flags_timestamp ON anomaly_flags(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_flag ON feedback(flag_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scores_rule ON rule_scores(rule_id)")
        
        self.conn.commit()
    
    def _init_json_storage(self) -> None:
        """Initialize JSON-based storage structure."""
        subdirs = ['rule_graphs', 'decision_trees', 'anomaly_flags', 
                   'feedback', 'mutations', 'agent_configs', 'external_context']
        for subdir in subdirs:
            (self.data_dir / subdir).mkdir(exist_ok=True)
        
        # Initialize index files
        for subdir in subdirs:
            index_file = self.data_dir / subdir / "_index.json"
            if not index_file.exists():
                with open(index_file, 'w') as f:
                    json.dump({}, f)
    
    # =========================================================================
    # Rule Graph Operations
    # =========================================================================
    
    def save_rule_graph(self, graph: RuleGraph, entity: str = None, 
                        agent_type: str = None, change_description: str = None) -> None:
        """Save or update a rule graph."""
        if self.use_json:
            self._save_rule_graph_json(graph, entity, agent_type)
        else:
            self._save_rule_graph_sql(graph, entity, agent_type, change_description)
        
        # Invalidate cache
        self._cache.pop(f"graph_{graph.graph_id}", None)
    
    def _save_rule_graph_sql(self, graph: RuleGraph, entity: str, 
                              agent_type: str, change_description: str) -> None:
        cursor = self.conn.cursor()
        
        # Check if exists
        cursor.execute("SELECT version FROM rule_graphs WHERE graph_id = ?", (graph.graph_id,))
        row = cursor.fetchone()
        
        if row:
            # Update existing - save old version to history first
            cursor.execute("""
                INSERT INTO rule_graph_history (history_id, graph_id, version, graph_data, change_description)
                SELECT ?, graph_id, version, graph_data, ?
                FROM rule_graphs WHERE graph_id = ?
            """, (str(uuid.uuid4())[:8], change_description or "Update", graph.graph_id))
            
            new_version = row['version'] + 1
            graph.version = new_version
            
            cursor.execute("""
                UPDATE rule_graphs 
                SET name = ?, description = ?, graph_data = ?, version = ?, 
                    entity = ?, agent_type = ?, updated_at = ?
                WHERE graph_id = ?
            """, (graph.name, graph.description, graph.to_json(), new_version,
                  entity, agent_type, datetime.now().isoformat(), graph.graph_id))
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO rule_graphs (graph_id, name, description, graph_data, entity, agent_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (graph.graph_id, graph.name, graph.description, graph.to_json(), entity, agent_type))
        
        self.conn.commit()
    
    def _save_rule_graph_json(self, graph: RuleGraph, entity: str, agent_type: str) -> None:
        filepath = self.data_dir / 'rule_graphs' / f"{graph.graph_id}.json"
        
        # Save with metadata
        data = {
            'graph': graph.to_dict(),
            'entity': entity,
            'agent_type': agent_type,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Update index
        self._update_json_index('rule_graphs', graph.graph_id, {
            'name': graph.name, 'entity': entity, 'agent_type': agent_type
        })
    
    def get_rule_graph(self, graph_id: str) -> Optional[RuleGraph]:
        """Retrieve a rule graph by ID."""
        cache_key = f"graph_{graph_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if self.use_json:
            graph = self._get_rule_graph_json(graph_id)
        else:
            graph = self._get_rule_graph_sql(graph_id)
        
        if graph:
            self._cache[cache_key] = graph
        return graph
    
    def _get_rule_graph_sql(self, graph_id: str) -> Optional[RuleGraph]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT graph_data FROM rule_graphs WHERE graph_id = ? AND is_active = 1", (graph_id,))
        row = cursor.fetchone()
        if row:
            return RuleGraph.from_json(row['graph_data'])
        return None
    
    def _get_rule_graph_json(self, graph_id: str) -> Optional[RuleGraph]:
        filepath = self.data_dir / 'rule_graphs' / f"{graph_id}.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                return RuleGraph.from_dict(data['graph'])
        return None
    
    def get_graphs_by_agent(self, agent_type: str) -> List[RuleGraph]:
        """Get all rule graphs for a specific agent type."""
        if self.use_json:
            index = self._get_json_index('rule_graphs')
            graphs = []
            for graph_id, meta in index.items():
                if meta.get('agent_type') == agent_type:
                    graph = self.get_rule_graph(graph_id)
                    if graph:
                        graphs.append(graph)
            return graphs
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT graph_data FROM rule_graphs 
                WHERE agent_type = ? AND is_active = 1
            """, (agent_type,))
            return [RuleGraph.from_json(row['graph_data']) for row in cursor.fetchall()]
    
    def get_graphs_by_entity(self, entity: str) -> List[RuleGraph]:
        """Get all rule graphs for a specific entity."""
        if self.use_json:
            index = self._get_json_index('rule_graphs')
            graphs = []
            for graph_id, meta in index.items():
                if meta.get('entity') == entity or meta.get('entity') is None:
                    graph = self.get_rule_graph(graph_id)
                    if graph:
                        graphs.append(graph)
            return graphs
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT graph_data FROM rule_graphs 
                WHERE (entity = ? OR entity IS NULL) AND is_active = 1
            """, (entity,))
            return [RuleGraph.from_json(row['graph_data']) for row in cursor.fetchall()]
    
    def list_rule_graphs(self) -> List[Dict[str, Any]]:
        """List all rule graphs with metadata."""
        if self.use_json:
            return list(self._get_json_index('rule_graphs').items())
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT graph_id, name, description, entity, agent_type, version, updated_at
                FROM rule_graphs WHERE is_active = 1
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # Anomaly Flag Operations
    # =========================================================================
    
    def save_anomaly_flag(self, flag: AnomalyFlag) -> None:
        """Save an anomaly flag."""
        if self.use_json:
            filepath = self.data_dir / 'anomaly_flags' / f"{flag.flag_id}.json"
            with open(filepath, 'w') as f:
                json.dump(flag.to_dict(), f, indent=2)
            self._update_json_index('anomaly_flags', flag.flag_id, {
                'entity': flag.entity, 'timestamp': flag.timestamp.isoformat()
            })
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO anomaly_flags 
                (flag_id, transaction_id, entity, timestamp, anomaly_type, severity, 
                 confidence, agent_id, rule_id, flag_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (flag.flag_id, flag.transaction_id, flag.entity, 
                  flag.timestamp.isoformat() if flag.timestamp else None,
                  flag.anomaly_type.value, flag.severity.value, flag.confidence,
                  flag.agent_id, flag.rule_id, json.dumps(flag.to_dict(), cls=NumpyEncoder)))
            self.conn.commit()
    
    def get_anomaly_flag(self, flag_id: str) -> Optional[AnomalyFlag]:
        """Retrieve an anomaly flag by ID."""
        if self.use_json:
            filepath = self.data_dir / 'anomaly_flags' / f"{flag_id}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    return AnomalyFlag.from_dict(json.load(f))
            return None
        else:
            cursor = self.conn.cursor()
            cursor.execute("SELECT flag_data FROM anomaly_flags WHERE flag_id = ?", (flag_id,))
            row = cursor.fetchone()
            if row:
                return AnomalyFlag.from_dict(json.loads(row['flag_data']))
            return None
    
    def get_flags_by_entity(self, entity: str, limit: int = 100) -> List[AnomalyFlag]:
        """Get recent anomaly flags for an entity."""
        if self.use_json:
            index = self._get_json_index('anomaly_flags')
            flags = []
            for flag_id, meta in index.items():
                if meta.get('entity') == entity:
                    flag = self.get_anomaly_flag(flag_id)
                    if flag:
                        flags.append(flag)
            return sorted(flags, key=lambda f: f.timestamp or datetime.min, reverse=True)[:limit]
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT flag_data FROM anomaly_flags 
                WHERE entity = ? ORDER BY timestamp DESC LIMIT ?
            """, (entity, limit))
            return [AnomalyFlag.from_dict(json.loads(row['flag_data'])) for row in cursor.fetchall()]
    
    def get_flags_by_rule(self, rule_id: str, days: int = 30) -> List[AnomalyFlag]:
        """Get flags triggered by a specific rule in the last N days."""
        if self.use_json:
            # Simplified for JSON - load all and filter
            index = self._get_json_index('anomaly_flags')
            flags = []
            cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
            for flag_id in index.keys():
                flag = self.get_anomaly_flag(flag_id)
                if flag and flag.rule_id == rule_id:
                    if flag.created_at.timestamp() > cutoff:
                        flags.append(flag)
            return flags
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT flag_data FROM anomaly_flags 
                WHERE rule_id = ? AND created_at >= datetime('now', ?)
            """, (rule_id, f'-{days} days'))
            return [AnomalyFlag.from_dict(json.loads(row['flag_data'])) for row in cursor.fetchall()]
    
    # =========================================================================
    # Feedback Operations
    # =========================================================================
    
    def save_feedback(self, feedback: Feedback) -> None:
        """Save feedback on an anomaly flag."""
        if self.use_json:
            filepath = self.data_dir / 'feedback' / f"{feedback.feedback_id}.json"
            with open(filepath, 'w') as f:
                json.dump(feedback.to_dict(), f, indent=2)
            self._update_json_index('feedback', feedback.feedback_id, {
                'flag_id': feedback.flag_id, 'feedback_type': feedback.feedback_type.value
            })
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO feedback 
                (feedback_id, flag_id, feedback_type, reviewer, comments, correct_label)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (feedback.feedback_id, feedback.flag_id, feedback.feedback_type.value,
                  feedback.reviewer, feedback.comments, feedback.correct_label))
            self.conn.commit()
    
    def get_feedback_for_rule(self, rule_id: str, days: int = 30) -> List[Feedback]:
        """Get all feedback for flags triggered by a rule."""
        # First get flags for the rule
        flags = self.get_flags_by_rule(rule_id, days)
        flag_ids = {f.flag_id for f in flags}
        
        if self.use_json:
            index = self._get_json_index('feedback')
            feedbacks = []
            for feedback_id, meta in index.items():
                if meta.get('flag_id') in flag_ids:
                    filepath = self.data_dir / 'feedback' / f"{feedback_id}.json"
                    if filepath.exists():
                        with open(filepath, 'r') as f:
                            feedbacks.append(Feedback.from_dict(json.load(f)))
            return feedbacks
        else:
            if not flag_ids:
                return []
            cursor = self.conn.cursor()
            placeholders = ','.join(['?' for _ in flag_ids])
            cursor.execute(f"""
                SELECT * FROM feedback WHERE flag_id IN ({placeholders})
            """, list(flag_ids))
            return [Feedback.from_dict(dict(row)) for row in cursor.fetchall()]
    
    # =========================================================================
    # Rule Score Operations
    # =========================================================================
    
    def save_rule_score(self, score: RuleScore) -> None:
        """Save rule performance score."""
        if self.use_json:
            filepath = self.data_dir / 'feedback' / f"score_{score.rule_id}_{datetime.now().strftime('%Y%m%d')}.json"
            with open(filepath, 'w') as f:
                json.dump(score.to_dict(), f, indent=2)
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO rule_scores 
                (score_id, rule_id, precision, recall, f1_score, true_positives,
                 false_positives, false_negatives, total_activations, evaluation_period_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (str(uuid.uuid4())[:8], score.rule_id, score.precision, score.recall,
                  score.f1_score, score.true_positives, score.false_positives,
                  score.false_negatives, score.total_activations, score.evaluation_period_days))
            self.conn.commit()
    
    def get_latest_rule_score(self, rule_id: str) -> Optional[RuleScore]:
        """Get the most recent performance score for a rule."""
        if self.use_json:
            # Find latest score file for rule
            score_files = list((self.data_dir / 'feedback').glob(f"score_{rule_id}_*.json"))
            if not score_files:
                return None
            latest = max(score_files, key=lambda p: p.name)
            with open(latest, 'r') as f:
                return RuleScore.from_dict(json.load(f))
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM rule_scores WHERE rule_id = ? 
                ORDER BY evaluated_at DESC LIMIT 1
            """, (rule_id,))
            row = cursor.fetchone()
            if row:
                return RuleScore.from_dict(dict(row))
            return None
    
    # =========================================================================
    # Mutation Operations
    # =========================================================================
    
    def save_mutation(self, mutation: Mutation) -> None:
        """Save a rule mutation."""
        if self.use_json:
            filepath = self.data_dir / 'mutations' / f"{mutation.mutation_id}.json"
            with open(filepath, 'w') as f:
                json.dump(mutation.to_dict(), f, indent=2)
            self._update_json_index('mutations', mutation.mutation_id, {
                'rule_id': mutation.rule_id, 'applied': mutation.applied
            })
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO mutations 
                (mutation_id, rule_id, mutation_type, original_value, new_value,
                 justification, confidence, applied, applied_at, rollback_available)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (mutation.mutation_id, mutation.rule_id, mutation.mutation_type.value,
                  json.dumps(mutation.original_value, cls=NumpyEncoder), json.dumps(mutation.new_value, cls=NumpyEncoder),
                  mutation.justification, mutation.confidence, int(mutation.applied),
                  mutation.applied_at.isoformat() if mutation.applied_at else None,
                  int(mutation.rollback_available)))
            self.conn.commit()
    
    def get_mutations_for_rule(self, rule_id: str, applied_only: bool = False) -> List[Mutation]:
        """Get all mutations for a rule."""
        if self.use_json:
            index = self._get_json_index('mutations')
            mutations = []
            for mutation_id, meta in index.items():
                if meta.get('rule_id') == rule_id:
                    if applied_only and not meta.get('applied'):
                        continue
                    filepath = self.data_dir / 'mutations' / f"{mutation_id}.json"
                    if filepath.exists():
                        with open(filepath, 'r') as f:
                            mutations.append(Mutation.from_dict(json.load(f)))
            return mutations
        else:
            cursor = self.conn.cursor()
            query = "SELECT * FROM mutations WHERE rule_id = ?"
            if applied_only:
                query += " AND applied = 1"
            cursor.execute(query, (rule_id,))
            return [Mutation.from_dict({
                **dict(row),
                'original_value': json.loads(row['original_value']),
                'new_value': json.loads(row['new_value'])
            }) for row in cursor.fetchall()]
    
    # =========================================================================
    # Agent Configuration Operations
    # =========================================================================
    
    def save_agent_config(self, agent_id: str, agent_type: str, 
                          config: Dict[str, Any], name: str = None) -> None:
        """Save agent configuration."""
        if self.use_json:
            filepath = self.data_dir / 'agent_configs' / f"{agent_id}.json"
            with open(filepath, 'w') as f:
                json.dump({
                    'agent_id': agent_id,
                    'agent_type': agent_type,
                    'name': name,
                    'config': config,
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO agent_configs 
                (agent_id, agent_type, name, config_data, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (agent_id, agent_type, name, json.dumps(config, cls=NumpyEncoder), datetime.now().isoformat()))
            self.conn.commit()
    
    def get_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent configuration."""
        if self.use_json:
            filepath = self.data_dir / 'agent_configs' / f"{agent_id}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    return json.load(f)
            return None
        else:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM agent_configs WHERE agent_id = ?", (agent_id,))
            row = cursor.fetchone()
            if row:
                return {**dict(row), 'config': json.loads(row['config_data'])}
            return None
    
    def get_active_agents(self) -> List[Dict[str, Any]]:
        """Get all active agent configurations."""
        if self.use_json:
            configs = []
            config_dir = self.data_dir / 'agent_configs'
            for filepath in config_dir.glob("*.json"):
                if filepath.name.startswith("_"):
                    continue
                with open(filepath, 'r') as f:
                    configs.append(json.load(f))
            return configs
        else:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM agent_configs WHERE is_active = 1 ORDER BY priority")
            return [{**dict(row), 'config': json.loads(row['config_data'])} for row in cursor.fetchall()]
    
    # =========================================================================
    # External Context Operations
    # =========================================================================
    
    def save_external_context(self, context_type: str, context_data: Dict[str, Any],
                               valid_hours: int = 24) -> None:
        """Save external context data."""
        context_id = f"{context_type}_{datetime.now().strftime('%Y%m%d%H')}"
        valid_until = datetime.now().timestamp() + (valid_hours * 3600)
        
        if self.use_json:
            filepath = self.data_dir / 'external_context' / f"{context_id}.json"
            with open(filepath, 'w') as f:
                json.dump({
                    'context_id': context_id,
                    'context_type': context_type,
                    'context_data': context_data,
                    'valid_until': datetime.fromtimestamp(valid_until).isoformat(),
                    'created_at': datetime.now().isoformat()
                }, f, indent=2)
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO external_context 
                (context_id, context_type, context_data, valid_until)
                VALUES (?, ?, ?, ?)
            """, (context_id, context_type, json.dumps(context_data, cls=NumpyEncoder),
                  datetime.fromtimestamp(valid_until).isoformat()))
            self.conn.commit()
    
    def get_latest_context(self, context_type: str) -> Optional[Dict[str, Any]]:
        """Get the latest valid external context."""
        if self.use_json:
            context_dir = self.data_dir / 'external_context'
            files = list(context_dir.glob(f"{context_type}_*.json"))
            if not files:
                return None
            latest = max(files, key=lambda p: p.name)
            with open(latest, 'r') as f:
                data = json.load(f)
                # Check if still valid
                if datetime.fromisoformat(data['valid_until']) > datetime.now():
                    return data['context_data']
            return None
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT context_data FROM external_context 
                WHERE context_type = ? AND valid_until > ?
                ORDER BY created_at DESC LIMIT 1
            """, (context_type, datetime.now().isoformat()))
            row = cursor.fetchone()
            if row:
                return json.loads(row['context_data'])
            return None
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _update_json_index(self, collection: str, item_id: str, metadata: Dict[str, Any]) -> None:
        """Update the index file for a collection."""
        index_file = self.data_dir / collection / "_index.json"
        with open(index_file, 'r') as f:
            index = json.load(f)
        index[item_id] = {**metadata, 'updated_at': datetime.now().isoformat()}
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _get_json_index(self, collection: str) -> Dict[str, Any]:
        """Get the index for a collection."""
        index_file = self.data_dir / collection / "_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
    
    def close(self) -> None:
        """Close database connection."""
        if not self.use_json and hasattr(self, 'conn'):
            self.conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics from the knowledge base."""
        if self.use_json:
            return {
                'rule_graphs': len(self._get_json_index('rule_graphs')),
                'anomaly_flags': len(self._get_json_index('anomaly_flags')),
                'feedback_items': len(self._get_json_index('feedback')),
                'mutations': len(self._get_json_index('mutations')),
                'storage_type': 'json'
            }
        else:
            cursor = self.conn.cursor()
            stats = {'storage_type': 'sqlite'}
            for table in ['rule_graphs', 'anomaly_flags', 'feedback', 'mutations']:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            return stats

