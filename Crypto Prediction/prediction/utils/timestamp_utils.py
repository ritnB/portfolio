# utils/timestamp_utils.py
# Unified timestampz processing utilities

import pandas as pd
from datetime import datetime
from typing import Union

def safe_parse_timestampz(timestamp_input: Union[str, datetime, pd.Timestamp]) -> datetime:
    """
    Unified function to safely parse Supabase timestampz format
    
    Supported formats:
    - "2025-01-25T12:34:56.789Z" (ISO with Z)
    - "2025-01-25T12:34:56.789" (ISO without Z)
    - "2025-01-25T12:34:56+00:00" (ISO with timezone)
    - datetime object
    - pandas Timestamp
    
    Returns:
        datetime: UTC-based datetime object (timezone-naive)
    """
    
    # If already a datetime object
    if isinstance(timestamp_input, datetime):
        # If timezone-aware, convert to UTC then make naive
        if timestamp_input.tzinfo is not None:
            return timestamp_input.utctimetuple()
        return timestamp_input
    
    # If pandas Timestamp
    if isinstance(timestamp_input, pd.Timestamp):
        return timestamp_input.to_pydatetime()
    
    # If string
    if isinstance(timestamp_input, str):
        # Normalize timestampz format
        clean_timestamp = timestamp_input.strip()
        
        # Remove Z if it ends with Z (UTC indicator)
        if clean_timestamp.endswith('Z'):
            clean_timestamp = clean_timestamp[:-1]
        
        # Remove timezone information (+00:00, +09:00, etc.)
        if '+' in clean_timestamp:
            clean_timestamp = clean_timestamp.split('+')[0]
        elif clean_timestamp.count('-') > 2:  # Prevent YYYY-MM-DD-HH:... format
            # Check if the part after the last - is timezone
            parts = clean_timestamp.split('-')
            if len(parts) >= 4 and ':' in parts[-1]:
                clean_timestamp = '-'.join(parts[:-1])
        
        try:
            # Parse as ISO format
            return datetime.fromisoformat(clean_timestamp)
        except ValueError:
            try:
                # Try pandas' more flexible parsing
                parsed = pd.to_datetime(clean_timestamp, utc=True)
                return parsed.to_pydatetime().replace(tzinfo=None)
            except Exception as e:
                print(f"⚠️ Timestamp parsing failed: {timestamp_input} -> {e}")
                # Last attempt: return current time
                return datetime.utcnow()
    
    # Unsupported format
    print(f"⚠️ Unsupported timestamp format: {type(timestamp_input)} {timestamp_input}")
    return datetime.utcnow()

def safe_parse_timestamp_series(timestamp_series: pd.Series) -> pd.Series:
    """
    Safely parse all timestamps in a pandas Series
    
    Args:
        timestamp_series: pandas Series containing timestamps
        
    Returns:
        pd.Series: parsed datetime Series
    """
    try:
        # If already datetime type
        if pd.api.types.is_datetime64_any_dtype(timestamp_series):
            return timestamp_series
        
        # Apply safe parsing to each value
        parsed_timestamps = timestamp_series.apply(safe_parse_timestampz)
        
        # Convert to datetime
        return pd.to_datetime(parsed_timestamps, errors='coerce')
        
    except Exception as e:
        print(f"⚠️ Timestamp series parsing error: {e}")
        # Force conversion on error
        return pd.to_datetime(timestamp_series, errors='coerce')

def normalize_timestamp_for_query(timestamp: Union[str, datetime]) -> str:
    """
    Normalize timestamp string for Supabase query
    
    Args:
        timestamp: timestamp to normalize
        
    Returns:
        str: "YYYY-MM-DDTHH:MM:SS" format string
    """
    if isinstance(timestamp, str):
        parsed = safe_parse_timestampz(timestamp)
    elif isinstance(timestamp, datetime):
        parsed = timestamp
    else:
        parsed = datetime.utcnow()
    
    # Return in ISO format (remove microseconds)
    return parsed.strftime('%Y-%m-%dT%H:%M:%S')

def get_utc_now_iso() -> str:
    """Return current UTC time in ISO format"""
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')

def calculate_time_difference_hours(start: Union[str, datetime], end: Union[str, datetime]) -> float:
    """
    Calculate time difference between two timestamps in hours
    
    Args:
        start: start time
        end: end time
        
    Returns:
        float: time difference in hours
    """
    start_dt = safe_parse_timestampz(start)
    end_dt = safe_parse_timestampz(end)
    
    diff = end_dt - start_dt
    return diff.total_seconds() / 3600.0