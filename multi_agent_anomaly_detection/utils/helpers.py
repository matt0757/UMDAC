"""
Utility Functions
=================

Helper functions for the multi-agent anomaly detection system.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np


def calculate_zscore(value: float, mean: float, std: float) -> float:
    """
    Calculate Z-score for a value.
    
    Args:
        value: The value to calculate Z-score for
        mean: Population mean
        std: Population standard deviation
    
    Returns:
        Z-score (0 if std is 0)
    """
    if std == 0 or pd.isna(std):
        return 0.0
    return (value - mean) / std


def calculate_rolling_stats(series: pd.Series, window: int = 4) -> Dict[str, pd.Series]:
    """
    Calculate rolling statistics for a series.
    
    Args:
        series: Pandas series
        window: Rolling window size
    
    Returns:
        Dictionary with 'mean', 'std', 'min', 'max' series
    """
    return {
        'mean': series.rolling(window=window, min_periods=1).mean(),
        'std': series.rolling(window=window, min_periods=1).std(),
        'min': series.rolling(window=window, min_periods=1).min(),
        'max': series.rolling(window=window, min_periods=1).max()
    }


def parse_date(date_input: Union[str, datetime, pd.Timestamp, None]) -> Optional[datetime]:
    """
    Parse various date formats to datetime.
    
    Args:
        date_input: Date in various formats
    
    Returns:
        datetime object or None
    """
    if date_input is None:
        return None
    
    if isinstance(date_input, datetime):
        return date_input
    
    if isinstance(date_input, pd.Timestamp):
        return date_input.to_pydatetime()
    
    if isinstance(date_input, str):
        # Try various formats
        formats = [
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%m/%d/%Y',
            '%d/%m/%Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_input, fmt)
            except ValueError:
                continue
        
        # Try ISO format
        try:
            return datetime.fromisoformat(date_input)
        except ValueError:
            pass
    
    return None


def format_currency(value: float, currency: str = 'USD') -> str:
    """
    Format a number as currency.
    
    Args:
        value: The value to format
        currency: Currency code
    
    Returns:
        Formatted currency string
    """
    if pd.isna(value):
        return 'N/A'
    
    if currency == 'USD':
        return f"${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"


def generate_report(verdict, format: str = 'text') -> str:
    """
    Generate a formatted report from an ensemble verdict.
    
    Args:
        verdict: EnsembleVerdict object
        format: Output format ('text', 'markdown', 'html')
    
    Returns:
        Formatted report string
    """
    if format == 'markdown':
        return _generate_markdown_report(verdict)
    elif format == 'html':
        return _generate_html_report(verdict)
    else:
        return _generate_text_report(verdict)


def _generate_text_report(verdict) -> str:
    """Generate plain text report."""
    lines = [
        "=" * 60,
        "ANOMALY DETECTION REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Entity: {verdict.entity}",
        f"Analysis Time: {verdict.timestamp}",
        f"Verdict: {'ANOMALY DETECTED' if verdict.is_anomaly else 'NORMAL'}",
        f"Confidence: {verdict.final_confidence:.1%}",
        f"Severity: {verdict.severity.value.upper()}",
        "",
        "-" * 60,
        "PRIMARY FINDINGS",
        "-" * 60,
    ]
    
    for i, flag in enumerate(verdict.primary_flags, 1):
        lines.append(f"\n{i}. {flag.description}")
        lines.append(f"   Agent: {flag.agent_id}")
        lines.append(f"   Severity: {flag.severity.value}")
        lines.append(f"   Confidence: {flag.confidence:.1%}")
    
    if not verdict.primary_flags:
        lines.append("   No anomalies detected")
    
    lines.extend([
        "",
        "-" * 60,
        "AGENT RESULTS",
        "-" * 60,
    ])
    
    for result in verdict.agent_results:
        status = f"DETECTED ({len(result.flags)} flags)" if result.flags else "NORMAL"
        lines.append(f"   {result.agent_type}: {status}")
    
    lines.extend([
        "",
        "-" * 60,
        "RECOMMENDED ACTION",
        "-" * 60,
        f"   {verdict.recommended_action}",
        "",
        "=" * 60,
    ])
    
    return "\n".join(lines)


def _generate_markdown_report(verdict) -> str:
    """Generate markdown report."""
    lines = [
        "# Anomaly Detection Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Entity | {verdict.entity} |",
        f"| Verdict | {'🔴 ANOMALY' if verdict.is_anomaly else '🟢 NORMAL'} |",
        f"| Confidence | {verdict.final_confidence:.1%} |",
        f"| Severity | {verdict.severity.value.upper()} |",
        "",
        "## Primary Findings",
        ""
    ]
    
    if verdict.primary_flags:
        for i, flag in enumerate(verdict.primary_flags, 1):
            lines.append(f"### {i}. {flag.description}")
            lines.append(f"- **Agent:** {flag.agent_id}")
            lines.append(f"- **Severity:** {flag.severity.value}")
            lines.append(f"- **Confidence:** {flag.confidence:.1%}")
            lines.append("")
    else:
        lines.append("*No anomalies detected*")
    
    lines.extend([
        "",
        "## Agent Results",
        "",
        "| Agent | Status | Flags |",
        "|-------|--------|-------|",
    ])
    
    for result in verdict.agent_results:
        status = "⚠️ Detected" if result.flags else "✅ Normal"
        lines.append(f"| {result.agent_type} | {status} | {len(result.flags)} |")
    
    lines.extend([
        "",
        "## Recommended Action",
        "",
        f"> {verdict.recommended_action}",
    ])
    
    return "\n".join(lines)


def _generate_html_report(verdict) -> str:
    """Generate HTML report."""
    status_class = "danger" if verdict.is_anomaly else "success"
    status_text = "ANOMALY DETECTED" if verdict.is_anomaly else "NORMAL"
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Anomaly Detection Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        .header {{ background: #1a1a2e; color: white; padding: 20px; border-radius: 8px; }}
        .status-{status_class} {{ background: {'#dc3545' if status_class == 'danger' else '#28a745'}; padding: 4px 12px; border-radius: 4px; }}
        .card {{ border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Anomaly Detection Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="card">
        <h2>Summary</h2>
        <table>
            <tr><th>Entity</th><td>{verdict.entity}</td></tr>
            <tr><th>Verdict</th><td><span class="status-{status_class}">{status_text}</span></td></tr>
            <tr><th>Confidence</th><td>{verdict.final_confidence:.1%}</td></tr>
            <tr><th>Severity</th><td>{verdict.severity.value.upper()}</td></tr>
        </table>
    </div>
    
    <div class="card">
        <h2>Primary Findings</h2>
"""
    
    if verdict.primary_flags:
        for i, flag in enumerate(verdict.primary_flags, 1):
            html += f"""
        <div style="margin: 10px 0; padding: 10px; background: #fff3cd; border-radius: 4px;">
            <strong>{i}. {flag.description}</strong>
            <p>Agent: {flag.agent_id} | Severity: {flag.severity.value} | Confidence: {flag.confidence:.1%}</p>
        </div>
"""
    else:
        html += "<p><em>No anomalies detected</em></p>"
    
    html += """
    </div>
    
    <div class="card">
        <h2>Recommended Action</h2>
        <p style="font-size: 1.2em; font-weight: bold;">""" + verdict.recommended_action + """</p>
    </div>
</body>
</html>
"""
    
    return html


def prepare_detection_data(raw_data: pd.DataFrame, 
                           entity_column: str = 'Name') -> pd.DataFrame:
    """
    Prepare raw transaction data for detection.
    
    Args:
        raw_data: Raw transaction DataFrame
        entity_column: Column containing entity identifier
    
    Returns:
        Prepared DataFrame with features
    """
    df = raw_data.copy()
    
    # Rename entity column if needed
    if entity_column in df.columns and entity_column != 'Entity':
        df['Entity'] = df[entity_column]
    
    # Parse dates
    date_cols = ['Pstng Date', 'Doc..Date', 'Week_Start']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Ensure numeric columns
    numeric_cols = ['Amount in USD', 'Total_Net', 'Total_Inflow', 'Total_Outflow']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

