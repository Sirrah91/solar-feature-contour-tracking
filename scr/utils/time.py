from datetime import datetime


def parse_datetime(
        date_str: str
) -> datetime | None:
    """
    Try parsing the date string with multiple formats and return the parsed datetime object.
    Returns None if parsing fails.
    """

    # List of possible date formats
    formats = [
        "%Y.%m.%d_%H:%M:%S.%f",  # e.g., 2025.05.06_12:30:45.123456
        "%Y.%m.%d_%H:%M:%S",  # e.g., 2025.05.06_12:30:45
        "%Y%m%d_%H%M%S",  # e.g., 20250506_123045
        "%Y-%m-%dT%H:%M:%S.%f",  # e.g., 2025-02-13T12:30:45.123456
        "%Y-%m-%dT%H:%M:%S",  # e.g., 2025-02-13T12:30:45
        "%Y-%m-%dT%H:%M",  # e.g., 2025-02-13T12:30
        "%Y-%m-%dT%H",  # e.g., 2025-02-13T12
        "%Y-%m-%d",  # e.g., 2025-02-13
        "%Y-%m-%d %H:%M:%S",  # e.g., 2025-02-13 12:30:45
        "%d/%m/%Y %H:%M:%S.%f",  # e.g., 13/02/2025 12:30:45.123456
        "%d/%m/%Y %H:%M:%S",  # e.g., 13/02/2025 12:30:45
        "%d/%m/%Y %H:%M",  # e.g., 13/02/2025 12:30
        "%d/%m/%Y %H",  # e.g., 13/02/2025 12
        "%d/%m/%Y",  # e.g., 13/02/2025
        "%d.%m.%Y",  # e.g., 13.02.2025
    ]

    for date_format in formats:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    return None
