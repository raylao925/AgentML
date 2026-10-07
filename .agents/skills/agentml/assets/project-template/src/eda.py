#!/usr/bin/env python3
"""
EDA - Exploratory Data Analysis (Standalone Script)

Converts the interactive eda.ipynb into a runnable Python script.
Accepts a project slug and optionally a config path, then loads data,
runs comprehensive EDA, and outputs a Markdown report.

Usage:
    python src/eda.py --project <project_slug>
    python src/eda.py --project <project_slug> --config configs/baseline.yaml --output memory/eda_report.md
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ============================================
# Parse Arguments
# ============================================
parser = argparse.ArgumentParser(description="Run EDA and generate report")
parser.add_argument("--project", type=str, required=True, help="Project slug (e.g., playground-series-s6e4)")
parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Config file path (relative to project root)")
parser.add_argument("--output", type=str, default=None, help="Output report path (relative to project root)")
parser.add_argument("--no-plots", action="store_true", help="Skip plot generation (for headless environments)")
args = parser.parse_args()

# ============================================
# 1) Locate Project Root
# ============================================
project_slug = args.project
_cwd = Path.cwd()

# Try to locate the project root
if _cwd.name == project_slug:
    PROJECT_ROOT = _cwd
elif _cwd.name == "src":
    PROJECT_ROOT = _cwd.parent
elif (_cwd / "projects" / project_slug).exists():
    PROJECT_ROOT = _cwd / "projects" / project_slug
elif (_cwd.parent / project_slug).exists():
    PROJECT_ROOT = _cwd.parent / project_slug
else:
    # Assume we're at AgentML root, project is under projects/
    PROJECT_ROOT = _cwd / "projects" / project_slug

PROJECT_ROOT = PROJECT_ROOT.resolve()
print(f"Project root: {PROJECT_ROOT}")

# ============================================
# 2) Load Config
# ============================================
config_path = PROJECT_ROOT / args.config
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"[OK] Config loaded from: {config_path}")
else:
    print(f"[WARN] Config not found: {config_path}")
    config = {}

data_cfg = config.get("data", {}) if config else {}
task_cfg = config.get("task", {}) if config else {}

# ============================================
# 3) Load Data
# ============================================
def _read_table(path: Path) -> pd.DataFrame:
    """Read a table file based on its extension."""
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    elif suf == ".gz":
        return pd.read_csv(path, compression="gzip")
    elif suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif suf == ".feather":
        return pd.read_feather(path)
    elif suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported format: {path}")

train_path = PROJECT_ROOT / data_cfg.get("train_path", "data/processed/train.parquet")
test_path_raw = data_cfg.get("test_path", "")
test_path = PROJECT_ROOT / test_path_raw if test_path_raw else None

print(f"Train path: {train_path}")
if train_path.exists():
    train_df = _read_table(train_path)
    print(f"[OK] Train data loaded: {train_df.shape}")
else:
    train_df = pd.DataFrame()
    print(f"[WARN] Train data not found at: {train_path}")

if test_path and test_path.exists():
    test_df = _read_table(test_path)
    print(f"[OK] Test data loaded: {test_df.shape}")
else:
    test_df = None
    print("Test data: none")

# ============================================
# 4) EDA Report Collector
# ============================================
report_sections = []

def add_section(title: str, content: str):
    """Add a section to the report."""
    report_sections.append((title, content))

def fmt_num(x: float, decimals: int = 4) -> str:
    """Format a number for display."""
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:.{decimals}f}"

# ============================================
# 5) Run EDA
# ============================================

# --- 5a) Data Overview ---
if not train_df.empty:
    n_rows, n_cols = train_df.shape
    target_col = task_cfg.get("target", "target")
    
    overview_lines = [
        f"- **Rows**: {n_rows:,}",
        f"- **Columns**: {n_cols}",
        f"- **Target column**: `{target_col}`",
        f"- **Target dtype**: {train_df[target_col].dtype if target_col in train_df.columns else 'N/A'}",
        f"",
        f"### Column List",
        f"```",
    ]
    for i, col in enumerate(train_df.columns):
        overview_lines.append(f"  {i:3d}. {col} ({train_df[col].dtype})")
    overview_lines.append("```")
    
    # Dtype summary
    dtype_counts = train_df.dtypes.value_counts()
    overview_lines.append(f"\n### Data Type Summary")
    for dtype, count in dtype_counts.items():
        overview_lines.append(f"- **{dtype}**: {count}")
    
    add_section("## 1. Data Overview", "\n".join(overview_lines))

# --- 5b) Missing Values ---
if not train_df.empty:
    missing = train_df.isnull().sum()
    missing_pct = (missing / len(train_df)) * 100
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values("missing_count", ascending=False)
    
    if not missing_df.empty:
        missing_lines = [
            f"**Overall missing rate**: {train_df.isnull().sum().mean() * 100:.2f}%",
            f"",
            f"| Column | Missing Count | Missing % |",
            f"|--------|--------------|-----------|",
        ]
        for col, row in missing_df.iterrows():
            missing_lines.append(f"| `{col}` | {fmt_num(row['missing_count'], 0)} | {row['missing_pct']:.2f}% |")
        add_section("## 2. Missing Values", "\n".join(missing_lines))
    else:
        add_section("## 2. Missing Values", "No missing values found.")

# --- 5c) Duplicates ---
if not train_df.empty:
    dup_count = train_df.duplicated().sum()
    dup_lines = [f"- **Duplicate rows**: {dup_count:,} ({dup_count/len(train_df)*100:.2f}%)"]
    
    id_cols = data_cfg.get("id_cols", [])
    if id_cols:
        id_dup = train_df.duplicated(subset=id_cols).sum()
        dup_lines.append(f"- **ID columns `{id_cols}` duplicates**: {id_dup:,} ({id_dup/len(train_df)*100:.2f}%)")
    
    add_section("## 3. Duplicates", "\n".join(dup_lines))

# --- 5d) Target Analysis ---
if not train_df.empty and target_col in train_df.columns:
    tv = train_df[target_col].dropna()
    
    target_lines = [
        f"- **Column**: `{target_col}`",
        f"- **Dtype**: {tv.dtype}",
        f"- **Non-null count**: {len(tv):,} / {len(train_df):,}",
    ]
    
    if tv.dtype in ("object", "category", "bool"):
        vc = tv.value_counts()
        target_lines.append(f"- **Unique classes**: {len(vc)}")
        target_lines.append(f"")
        target_lines.append(f"| Class | Count | Proportion |")
        target_lines.append(f"|-------|-------|-----------|")
        for val, cnt in vc.items():
            target_lines.append(f"| {val} | {cnt:,} | {cnt/len(tv)*100:.2f}% |")
        
        # Imbalance ratio
        if len(vc) >= 2:
            ratio = vc.iloc[0] / vc.iloc[1] if vc.iloc[1] > 0 else float("inf")
            target_lines.append(f"\n- **Imbalance ratio (major/minor)**: {ratio:.2f}")
    else:
        target_lines.append(f"- **Mean**: {tv.mean():.4f}")
        target_lines.append(f"- **Std**: {tv.std():.4f}")
        target_lines.append(f"- **Min**: {tv.min():.4f}")
        target_lines.append(f"- **25%**: {tv.quantile(0.25):.4f}")
        target_lines.append(f"- **50%**: {tv.quantile(0.50):.4f}")
        target_lines.append(f"- **75%**: {tv.quantile(0.75):.4f}")
        target_lines.append(f"- **Max**: {tv.max():.4f}")
        target_lines.append(f"- **Skewness**: {tv.skew():.4f}")
        target_lines.append(f"- **Kurtosis**: {tv.kurtosis():.4f}")
        target_lines.append(f"- **NaN count**: {tv.isnull().sum() if tv.isnull().sum() > 0 else 0}")
    
    add_section("## 4. Target Variable Analysis", "\n".join(target_lines))

# --- 5e) Numeric Features ---
if not train_df.empty:
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    
    if num_cols:
        num_lines = [
            f"**Total numeric features**: {len(num_cols)}",
            f"",
            f"| Column | Count | Mean | Std | Min | 25% | 50% | 75% | Max | Skew | Missing% |",
            f"|--------|-------|------|-----|-----|-----|-----|-----|-----|------|---------|",
        ]
        desc = train_df[num_cols].describe().T
        for col in num_cols:
            if col in desc.index:
                row = desc.loc[col]
                miss_pct = train_df[col].isnull().mean() * 100
                skew_val = train_df[col].skew()
                num_lines.append(
                    f"| `{col}` | {fmt_num(row['count'], 0)} | {row['mean']:.4f} | {row['std']:.4f} | "
                    f"{row['min']:.4f} | {row['25%']:.4f} | {row['50%']:.4f} | {row['75%']:.4f} | "
                    f"{row['max']:.4f} | {skew_val:.2f} | {miss_pct:.2f}% |"
                )
        add_section("## 5. Numeric Features", "\n".join(num_lines))
    else:
        add_section("## 5. Numeric Features", "No numeric features found.")

# --- 5f) Categorical Features ---
if not train_df.empty:
    cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    
    if cat_cols:
        cat_lines = [
            f"**Total categorical features**: {len(cat_cols)}",
            f"",
        ]
        for col in cat_cols:
            nunique = train_df[col].nunique()
            miss_pct = train_df[col].isnull().mean() * 100
            cat_lines.append(f"### `{col}`")
            cat_lines.append(f"- **Unique values**: {nunique:,}")
            cat_lines.append(f"- **Missing**: {miss_pct:.2f}%")
            if nunique <= 20:
                vc = train_df[col].value_counts()
                cat_lines.append(f"\n  | Value | Count | Proportion |")
                cat_lines.append(f"  |-------|-------|-----------|")
                for val, cnt in vc.items():
                    cat_lines.append(f"  | {val} | {cnt:,} | {cnt/len(train_df)*100:.2f}% |")
            cat_lines.append("")
        add_section("## 6. Categorical Features", "\n".join(cat_lines))
    else:
        add_section("## 6. Categorical Features", "No categorical features found.")

# --- 5g) Correlation Matrix ---
if not train_df.empty:
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        corr = train_df[num_cols].corr()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                val = corr.iloc[i, j]
                if abs(val) > 0.8:
                    high_corr_pairs.append((num_cols[i], num_cols[j], val))
        
        corr_lines = [
            f"**Total numeric features for correlation**: {len(num_cols)}",
            f"",
        ]
        if high_corr_pairs:
            corr_lines.append(f"### High Correlation Pairs (|r| > 0.8)")
            corr_lines.append(f"| Feature A | Feature B | Correlation |")
            corr_lines.append(f"|-----------|-----------|-------------|")
            for a, b, val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                corr_lines.append(f"| `{a}` | `{b}` | {val:.4f} |")
            corr_lines.append("")
        
        # Target correlation (if target is numeric)
        if target_col in num_cols:
            target_corr = corr[target_col].drop(target_col).sort_values(ascending=False)
            corr_lines.append(f"### Top Correlations with Target (`{target_col}`)")
            corr_lines.append(f"| Feature | Correlation |")
            corr_lines.append(f"|---------|-------------|")
            for feat, val in target_corr.head(10).items():
                corr_lines.append(f"| `{feat}` | {val:.4f} |")
            if len(target_corr) > 10:
                corr_lines.append(f"  ... and {len(target_corr) - 10} more features")
        
        add_section("## 7. Feature Correlations", "\n".join(corr_lines))
    else:
        add_section("## 7. Feature Correlations", f"Only {len(num_cols)} numeric feature(s); correlation requires at least 2.")

# --- 5h) Leakage Check ---
if not train_df.empty:
    leak_keywords = ["leak", "future", "target_", "post_", "after_",
                     "_result", "_outcome", "_status", "refund", "cancel_"]
    
    all_cols = list(train_df.columns)
    leak_candidates = []
    for col in all_cols:
        col_lower = col.lower()
        if any(kw in col_lower for kw in leak_keywords):
            leak_candidates.append(col)
    
    leak_lines = []
    if leak_candidates:
        leak_lines.append("**:warning: Potential Leak Columns Detected**")
        leak_lines.append(f"")
        leak_lines.append(f"| Column | Reason |")
        leak_lines.append(f"|--------|--------|")
        for col in leak_candidates:
            matched_kw = [kw for kw in leak_keywords if kw in col.lower()]
            leak_lines.append(f"| `{col}` | Name matches: {', '.join(matched_kw)} |")
    else:
        leak_lines.append("No obvious leak columns detected.")
    
    drop_cols = data_cfg.get("drop_cols", [])
    if drop_cols:
        leak_lines.append(f"\n**Columns marked for removal in config**: `{drop_cols}`")
    
    add_section("## 8. Data Leakage Check", "\n".join(leak_lines))

# --- 5i) CV Strategy Validation ---
if config:
    cv_cfg = config.get("cv", {})
    cv_lines = [
        f"- **CV Type**: {cv_cfg.get('cv_type', 'N/A')}",
        f"- **N Splits**: {cv_cfg.get('n_splits', 'N/A')}",
        f"- **Shuffle**: {cv_cfg.get('shuffle', 'N/A')}",
        f"- **Random State**: {cv_cfg.get('random_state', 'N/A')}",
    ]
    
    group_key = cv_cfg.get("group_key") or data_cfg.get("group_key")
    time_col = cv_cfg.get("time_col") or data_cfg.get("time_col")
    
    if group_key and not train_df.empty and group_key in train_df.columns:
        n_groups = train_df[group_key].nunique()
        dup_rate = (1 - n_groups / len(train_df)) * 100 if len(train_df) > 0 else 0
        cv_lines.append(f"\n- **Group key `{group_key}`**: {n_groups:,} unique groups")
        cv_lines.append(f"- **Group repeat rate**: {dup_rate:.1f}%")
        if dup_rate > 5:
            cv_lines.append(f"- **:warning: High group repeat rate -> GroupKFold recommended**")
    
    if time_col and not train_df.empty and time_col in train_df.columns:
        cv_lines.append(f"\n- **Time column `{time_col}`**: {train_df[time_col].min()} ~ {train_df[time_col].max()}")
        cv_lines.append(f"- **:warning: Time column present -> TimeSeriesSplit may be needed**")
    
    add_section("## 9. CV Strategy Validation", "\n".join(cv_lines))

# --- 5j) Feature Engineering Suggestions ---
if not train_df.empty:
    suggestions = []
    
    # Skewness-based suggestions
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    if num_cols:
        skewness = train_df[num_cols].skew().sort_values(ascending=False)
        high_skew = skewness[skewness.abs() > 1]
        if not high_skew.empty:
            suggestions.append("### Log/Box-Cox Transform Candidates (|skew| > 1)")
            for col, sk in high_skew.items():
                suggestions.append(f"- `{col}` (skewness: {sk:.2f})")
            suggestions.append("")
    
    # Cardinality-based suggestions
    cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    if cat_cols:
        high_card = [(c, train_df[c].nunique()) for c in cat_cols if train_df[c].nunique() > 20]
        if high_card:
            suggestions.append("### High-Cardinality Categorical (Target Encoding)")
            for col, card in high_card:
                suggestions.append(f"- `{col}` ({card} unique values)")
            suggestions.append("")
        
        low_card = [(c, train_df[c].nunique()) for c in cat_cols if 2 <= train_df[c].nunique() <= 20]
        if low_card:
            suggestions.append("### Low-Cardinality Categorical (One-Hot Encoding)")
            for col, card in low_card:
                suggestions.append(f"- `{col}` ({card} unique values)")
            suggestions.append("")
    
    suggestions.append("### General Suggestions")
    suggestions.append("- Create interaction features (cross features)")
    suggestions.append("- Try group aggregations (group by categorical, aggregate numeric)")
    suggestions.append("- Check for time-based features if datetime columns exist")
    suggestions.append("- Run featuretools for automated feature engineering")
    
    add_section("## 10. Feature Engineering Suggestions", "\n".join(suggestions))

# --- 5k) Summary ---
if not train_df.empty:
    missing_rate = train_df.isnull().sum().mean() * 100
    summary_lines = [
        f"| Aspect | Value |",
        f"|--------|-------|",
        f"| Dataset size | {len(train_df):,} rows x {len(train_df.columns)} cols |",
        f"| Target column | `{target_col}` |",
        f"| Overall missing rate | {missing_rate:.2f}% |",
        f"| Numeric features | {len(train_df.select_dtypes(include=[np.number]).columns) - (1 if target_col in train_df.select_dtypes(include=[np.number]).columns else 0)} |",
        f"| Categorical features | {len(train_df.select_dtypes(include=['object', 'category']).columns) - (1 if target_col in train_df.select_dtypes(include=['object', 'category']).columns else 0)} |",
        f"| Duplicate rows | {train_df.duplicated().sum():,} ({train_df.duplicated().sum()/len(train_df)*100:.2f}%) |",
    ]
    if target_col in train_df.columns:
        tv = train_df[target_col]
        if tv.dtype in ("object", "category", "bool"):
            vc = tv.value_counts()
            summary_lines.append(f"| Target classes | {len(vc)} |")
            summary_lines.append(f"| Majority class | {vc.iloc[0]/len(tv)*100:.1f}% |")
        else:
            summary_lines.append(f"| Target mean/std | {tv.mean():.4f} / {tv.std():.4f} |")
    
    add_section("## 11. Summary", "\n".join(summary_lines))

# ============================================
# 6) Write Report
# ============================================
report_lines = [
    f"# EDA Report — {project_slug}",
    f"",
    f"> **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"> **Data source**: `{train_path}`",
    f"> **Config**: `{config_path}`",
    f"",
    f"---",
    f"",
]

for title, content in report_sections:
    report_lines.append(title)
    report_lines.append("")
    report_lines.append(content)
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

report_text = "\n".join(report_lines)

# Determine output path
if args.output:
    output_path = PROJECT_ROOT / args.output
else:
    output_dir = PROJECT_ROOT / "memory"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eda_report.md"

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"\n[OK] EDA report written to: {output_path}")
print(f"[OK] EDA completed successfully.")