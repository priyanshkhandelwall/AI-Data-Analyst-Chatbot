import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io

def get_data_profile(df):
    """Generate a profile of the dataframe."""
    profile = {
        "rowCount": len(df),
        "columnCount": len(df.columns),
        "missingValues": df.isnull().sum().to_dict(),
        "duplicateRows": df.duplicated().sum(),
        "duplicateRowsPercent": (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0,
        "columns": []
    }
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        if "int" in col_type or "float" in col_type:
            type_name = "number"
        elif "datetime" in col_type:
            type_name = "date"
        elif "bool" in col_type:
            type_name = "boolean"
        else:
            type_name = "string"
            
        profile["columns"].append({
            "name": col,
            "type": type_name,
            "nullable": df[col].isnull().any()
        })
        
    return profile

def get_column_stats(df):
    """Compute detailed statistics for each column."""
    stats_list = []
    for col in df.columns:
        col_data = df[col]
        stat = {
            "name": col,
            "type": str(col_data.dtype),
            "count": int(col_data.count()),
            "missing": int(col_data.isnull().sum()),
            "missingPercent": (col_data.isnull().sum() / len(df)) * 100,
            "unique": int(col_data.nunique()),
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            stat.update({
                "mean": float(col_data.mean()) if not col_data.empty else None,
                "median": float(col_data.median()) if not col_data.empty else None,
                "std": float(col_data.std()) if not col_data.empty else None,
                "min": float(col_data.min()) if not col_data.empty else None,
                "max": float(col_data.max()) if not col_data.empty else None,
                "percentile25": float(col_data.quantile(0.25)) if not col_data.empty else None,
                "percentile75": float(col_data.quantile(0.75)) if not col_data.empty else None,
            })
        else:
            stat["mode"] = str(col_data.mode()[0]) if not col_data.mode().empty else None
            
        stats_list.append(stat)
    return stats_list

def get_correlations(df):
    """Compute correlations for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {}
    return numeric_df.corr().to_dict()

def detect_outliers(df, method="iqr"):
    """Detect outliers in numeric columns."""
    outliers = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        if col_data.empty:
            continue
            
        if method == "iqr":
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            count = outlier_mask.sum()
            threshold = iqr
        else: # zscore
            z_scores = np.abs(stats.zscore(col_data))
            count = (z_scores > 3).sum()
            threshold = 3
            
        outliers.append({
            "column": col,
            "method": method,
            "outlierCount": int(count),
            "outlierPercent": (count / len(col_data)) * 100,
            "threshold": float(threshold)
        })
    return outliers
