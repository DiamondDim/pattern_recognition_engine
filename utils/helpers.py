"""
Helper functions for data processing and analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame,
                       required_columns: Optional[List[str]] = None) -> bool:
    """Validate DataFrame structure and content.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if DataFrame is valid, False otherwise
    """
    if df is None:
        logger.error("DataFrame is None")
        return False

    if df.empty:
        logger.warning("DataFrame is empty")
        return False

    # Check for required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"DataFrame contains {nan_count} NaN values")
        # Still return True, just log warning

    # Check for infinite values
    if np.any(np.isinf(df.select_dtypes(include=[np.number]))):
        logger.warning("DataFrame contains infinite values")

    return True


def calculate_returns(prices: pd.Series,
                      periods: int = 1,
                      log_returns: bool = False) -> pd.Series:
    """Calculate returns from price series.

    Args:
        prices: Price series
        periods: Number of periods for return calculation
        log_returns: If True, calculate log returns

    Returns:
        Returns series
    """
    if log_returns:
        returns = np.log(prices / prices.shift(periods))
    else:
        returns = prices.pct_change(periods)

    returns.name = f'returns_{periods}'
    return returns


def normalize_data(data: pd.DataFrame,
                   method: str = 'minmax',
                   columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Normalize data using specified method.

    Args:
        data: DataFrame to normalize
        method: Normalization method ('minmax', 'zscore', 'robust')
        columns: Columns to normalize (None for all numeric columns)

    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns

    data_normalized = data.copy()

    for column in columns:
        if column not in data.columns:
            continue

        if method == 'minmax':
            col_min = data[column].min()
            col_max = data[column].max()
            if col_max != col_min:
                data_normalized[column] = (data[column] - col_min) / (col_max - col_min)

        elif method == 'zscore':
            col_mean = data[column].mean()
            col_std = data[column].std()
            if col_std > 0:
                data_normalized[column] = (data[column] - col_mean) / col_std

        elif method == 'robust':
            col_median = data[column].median()
            col_iqr = data[column].quantile(0.75) - data[column].quantile(0.25)
            if col_iqr > 0:
                data_normalized[column] = (data[column] - col_median) / col_iqr

    return data_normalized


def create_lagged_features(data: pd.DataFrame,
                           columns: List[str],
                           lags: List[int]) -> pd.DataFrame:
    """Create lagged features for time series data.

    Args:
        data: DataFrame with time series data
        columns: Columns to create lags for
        lags: List of lag periods

    Returns:
        DataFrame with original and lagged features
    """
    data_lagged = data.copy()

    for column in columns:
        if column not in data.columns:
            continue

        for lag in lags:
            if lag > 0:
                lagged_col_name = f'{column}_lag_{lag}'
                data_lagged[lagged_col_name] = data[column].shift(lag)

    # Drop rows with NaN values from lagging
    if data_lagged.isna().any().any():
        initial_rows = len(data_lagged)
        data_lagged = data_lagged.dropna()
        dropped_rows = initial_rows - len(data_lagged)
        logger.debug(f"Dropped {dropped_rows} rows with NaN values from lagged features")

    return data_lagged


def split_train_test(data: pd.DataFrame,
                     target_column: str,
                     test_size: float = 0.2,
                     shuffle: bool = True,
                     random_state: int = 42) -> Tuple:
    """Split data into training and testing sets.

    Args:
        data: DataFrame to split
        target_column: Name of target column
        test_size: Proportion of data for testing
        shuffle: Whether to shuffle data before splitting
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if target_column not in data.columns:
        logger.error(f"Target column '{target_column}' not found in data")
        raise ValueError(f"Target column '{target_column}' not found")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
    )

    logger.info(f"Split data: {len(X_train)} train, {len(X_test)} test samples")

    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Calculate performance metrics for predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for ROC/AUC)

    Returns:
        Dictionary with calculated metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score
    )

    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['classification_report'] = report

    # ROC-AUC if probabilities provided
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except:
            metrics['roc_auc'] = None

    # Additional custom metrics
    metrics['error_rate'] = 1 - metrics['accuracy']

    return metrics


def resample_data(data: pd.DataFrame,
                  rule: str = '1D',
                  agg_method: str = 'ohlc') -> pd.DataFrame:
    """Resample time series data to different frequency.

    Args:
        data: DataFrame with datetime index
        rule: Resampling rule (e.g., '1H', '1D', '1W')
        agg_method: Aggregation method ('ohlc', 'mean', 'sum')

    Returns:
        Resampled DataFrame
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.error("Data must have datetime index for resampling")
        return data

    if agg_method == 'ohlc':
        # For OHLC data
        if 'open' in data.columns and 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            ohlc_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            resampled = data.resample(rule).agg(ohlc_dict).dropna()
        else:
            resampled = data.resample(rule).mean().dropna()
    elif agg_method == 'mean':
        resampled = data.resample(rule).mean().dropna()
    elif agg_method == 'sum':
        resampled = data.resample(rule).sum().dropna()
    else:
        resampled = data.resample(rule).mean().dropna()

    logger.debug(f"Resampled data from {len(data)} to {len(resampled)} rows")

    return resampled


def detect_outliers_iqr(data: pd.Series,
                        threshold: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method.

    Args:
        data: Input data series
        threshold: IQR multiplier threshold

    Returns:
        Boolean series indicating outliers
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    outliers = (data < lower_bound) | (data > upper_bound)

    return outliers


def calculate_correlation_matrix(data: pd.DataFrame,
                                 method: str = 'pearson') -> pd.DataFrame:
    """Calculate correlation matrix for DataFrame.

    Args:
        data: Input DataFrame
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Correlation matrix DataFrame
    """
    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.empty:
        logger.warning("No numeric columns for correlation calculation")
        return pd.DataFrame()

    corr_matrix = numeric_data.corr(method=method)

    return corr_matrix


def save_dataframe(df: pd.DataFrame,
                   filepath: str,
                   format: str = 'csv') -> bool:
    """Save DataFrame to file.

    Args:
        df: DataFrame to save
        filepath: Output file path
        format: File format ('csv', 'parquet', 'pickle')

    Returns:
        True if successful, False otherwise
    """
    try:
        if format == 'csv':
            df.to_csv(filepath, index=True)
        elif format == 'parquet':
            df.to_parquet(filepath, index=True)
        elif format == 'pickle':
            df.to_pickle(filepath)
        else:
            logger.error(f"Unsupported format: {format}")
            return False

        logger.info(f"DataFrame saved to {filepath} ({format})")
        return True

    except Exception as e:
        logger.error(f"Failed to save DataFrame to {filepath}: {e}")
        return False


def load_dataframe(filepath: str,
                   format: str = 'csv') -> pd.DataFrame:
    """Load DataFrame from file.

    Args:
        filepath: Input file path
        format: File format ('csv', 'parquet', 'pickle')

    Returns:
        Loaded DataFrame
    """
    try:
        if format == 'csv':
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif format == 'parquet':
            df = pd.read_parquet(filepath)
        elif format == 'pickle':
            df = pd.read_pickle(filepath)
        else:
            logger.error(f"Unsupported format: {format}")
            return pd.DataFrame()

        logger.info(f"DataFrame loaded from {filepath} ({format}), shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Failed to load DataFrame from {filepath}: {e}")
        return pd.DataFrame()

