import pandas as pd
from datetime import datetime
from typing import Union


def safe_parse_timestampz(timestamp_input: Union[str, datetime, pd.Timestamp]) -> datetime:
    """Parse Supabase timestampz safely and return a timezone-naive UTC datetime.

    Supported formats include:
    - "2025-01-25T12:34:56.789Z" (ISO with Z)
    - "2025-01-25T12:34:56.789" (ISO without Z)
    - "2025-01-25T12:34:56+00:00" (ISO with timezone)
    - datetime
    - pandas Timestamp
    """
    if isinstance(timestamp_input, datetime):
        if timestamp_input.tzinfo is not None:
            # Drop tzinfo and assume UTC
            return timestamp_input.replace(tzinfo=None)
        return timestamp_input

    if isinstance(timestamp_input, pd.Timestamp):
        return timestamp_input.to_pydatetime().replace(tzinfo=None)

    if isinstance(timestamp_input, str):
        clean_timestamp = timestamp_input.strip()
        if clean_timestamp.endswith('Z'):
            clean_timestamp = clean_timestamp[:-1]
        if '+' in clean_timestamp:
            clean_timestamp = clean_timestamp.split('+')[0]
        elif clean_timestamp.count('-') > 2:
            parts = clean_timestamp.split('-')
            if len(parts) >= 4 and ':' in parts[-1]:
                clean_timestamp = '-'.join(parts[:-1])
        try:
            return datetime.fromisoformat(clean_timestamp)
        except ValueError:
            try:
                parsed = pd.to_datetime(clean_timestamp, utc=True)
                return parsed.to_pydatetime().replace(tzinfo=None)
            except Exception as e:
                print(f"⚠️ Timestamp parse failed: {timestamp_input} -> {e}")
                return datetime.utcnow()

    print(f"⚠️ Unsupported timestamp type: {type(timestamp_input)} {timestamp_input}")
    return datetime.utcnow()


def safe_parse_timestamp_series(timestamp_series: pd.Series) -> pd.Series:
    """Safely parse all timestamps in a pandas Series into datetimes."""
    try:
        if pd.api.types.is_datetime64_any_dtype(timestamp_series):
            return timestamp_series
        parsed_timestamps = timestamp_series.apply(safe_parse_timestampz)
        return pd.to_datetime(parsed_timestamps, errors='coerce')
    except Exception as e:
        print(f"⚠️ Timestamp series parse error: {e}")
        return pd.to_datetime(timestamp_series, errors='coerce')


def normalize_timestamp_for_query(timestamp: Union[str, datetime]) -> str:
    """Normalize a timestamp to "YYYY-MM-DDTHH:MM:SS" string for queries."""
    if isinstance(timestamp, str):
        parsed = safe_parse_timestampz(timestamp)
    elif isinstance(timestamp, datetime):
        parsed = timestamp
    else:
        parsed = datetime.utcnow()
    return parsed.strftime('%Y-%m-%dT%H:%M:%S')


def get_utc_now_iso() -> str:
    """Return current UTC time in ISO format (no microseconds)."""
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')


def calculate_time_difference_hours(start: Union[str, datetime], end: Union[str, datetime]) -> float:
    """Compute the time difference in hours between two timestamps."""
    start_dt = safe_parse_timestampz(start)
    end_dt = safe_parse_timestampz(end)
    diff = end_dt - start_dt
    return diff.total_seconds() / 3600.0