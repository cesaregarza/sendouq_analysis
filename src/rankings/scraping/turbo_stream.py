"""
Turbo-stream decoder for Python.

This module decodes the turbo-stream format used by React Router's Single Fetch
feature, which is the new format used by Sendou.ink's .data endpoints.

The turbo-stream format is a compact JSON representation where:
- Objects have `_N` keys where N is an index into the flat array
- Negative integers represent special values (null, undefined, etc.)
- The first element is a header mapping route keys to their data positions

Reference: https://github.com/jacob-ebey/turbo-stream
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Special value constants for turbo-stream format
# Based on observed values in Sendou.ink responses
SPECIAL_VALUES: dict[int, Any] = {
    -1: None,  # null
    -2: True,  # true
    -3: False,  # false
    -4: float("nan"),  # NaN
    -5: None,  # null (alternate encoding)
    -6: float("inf"),  # Infinity
    -7: None,  # undefined (treated as None in Python)
    -8: float("-inf"),  # -Infinity
    -9: -0.0,  # negative zero
}


class TurboStreamDecoder:
    """Decoder for turbo-stream format data."""

    def __init__(self, data: list) -> None:
        """
        Initialize decoder with raw turbo-stream array data.

        Parameters
        ----------
        data : list
            The raw JSON array from a .data endpoint response
        """
        self.data = data
        self._cache: dict[int, Any] = {}

    def decode_value(self, idx: int) -> Any:
        """
        Recursively decode value at array index.

        Parameters
        ----------
        idx : int
            Array index to decode. Negative values are special constants.

        Returns
        -------
        Any
            The decoded Python value
        """
        # Check cache first
        if idx in self._cache:
            return self._cache[idx]

        # Handle special values (negative indices)
        if idx < 0:
            return SPECIAL_VALUES.get(idx)

        # Bounds check
        if idx >= len(self.data):
            logger.warning(f"Index {idx} out of bounds (len={len(self.data)})")
            return None

        raw = self.data[idx]

        if isinstance(raw, dict):
            # Object with _N keys referencing other positions
            result: dict[str, Any] = {}
            for key, val_idx in raw.items():
                if key.startswith("_"):
                    try:
                        key_idx = int(key[1:])
                        actual_key = self.decode_value(key_idx)
                        actual_val = self.decode_value(val_idx)
                        # Skip undefined keys
                        if actual_key is not None:
                            result[actual_key] = actual_val
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error decoding key {key}: {e}")
            self._cache[idx] = result
            return result

        elif isinstance(raw, list):
            # Array - decode each element
            result_list: list[Any] = []
            for item in raw:
                if isinstance(item, int):
                    result_list.append(self.decode_value(item))
                else:
                    # Direct value in array (shouldn't happen often)
                    result_list.append(item)
            self._cache[idx] = result_list
            return result_list

        else:
            # Direct value (string, number, boolean, null)
            return raw

    def decode(self) -> dict[str, Any]:
        """
        Decode the full turbo-stream structure.

        Returns
        -------
        dict
            Dictionary mapping route names to their decoded data
        """
        if not self.data:
            return {}

        # Header at position 0 maps route keys to values
        header = self.data[0]

        result: dict[str, Any] = {}
        if isinstance(header, dict):
            for key, val_idx in header.items():
                if key.startswith("_"):
                    try:
                        key_idx = int(key[1:])
                        actual_key = self.decode_value(key_idx)
                        actual_val = self.decode_value(val_idx)
                        if actual_key is not None:
                            result[actual_key] = actual_val
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error decoding route {key}: {e}")

        return result

    def get_tournament_data(self) -> dict[str, Any] | None:
        """
        Extract tournament data from decoded structure.

        This is a convenience method for the common case of fetching
        tournament data from Sendou.ink.

        Returns
        -------
        dict or None
            The tournament data dict, or None if not found
        """
        decoded = self.decode()

        # Try the tournament route key
        route_key = "features/tournament/routes/to.$id"
        if route_key in decoded:
            route_data = decoded[route_key]
            if isinstance(route_data, dict) and "data" in route_data:
                data = route_data["data"]
                # Handle case where data is a JSON string (new format as of late 2025)
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if isinstance(data, dict) and "tournament" in data:
                    return data

        # Fallback: try results route
        results_key = "features/tournament/routes/to.$id.results"
        if results_key in decoded:
            route_data = decoded[results_key]
            if isinstance(route_data, dict) and "data" in route_data:
                data = route_data["data"]
                # Handle case where data is a JSON string (new format as of late 2025)
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if isinstance(data, dict):
                    return data

        return None


def decode_turbo_stream(data: list | str | bytes) -> dict[str, Any]:
    """
    Decode turbo-stream data to Python dict.

    Parameters
    ----------
    data : list, str, or bytes
        Either the already-parsed JSON list, or raw JSON string/bytes

    Returns
    -------
    dict
        Dictionary mapping route names to their decoded data
    """
    if isinstance(data, (str, bytes)):
        data = json.loads(data)

    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data).__name__}")

    decoder = TurboStreamDecoder(data)
    return decoder.decode()


def extract_route_data(
    data: list | str | bytes,
    route_key: str,
) -> dict[str, Any] | None:
    """
    Extract data for a specific route from turbo-stream response.

    Parameters
    ----------
    data : list, str, or bytes
        Raw turbo-stream response data
    route_key : str
        The route key to extract (e.g., "features/sendouq-match/routes/q.match.$id")

    Returns
    -------
    dict or None
        The data for the specified route, or None if not found
    """
    if isinstance(data, (str, bytes)):
        data = json.loads(data)

    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data).__name__}")

    decoder = TurboStreamDecoder(data)
    decoded = decoder.decode()

    if route_key in decoded:
        route_data = decoded[route_key]
        if isinstance(route_data, dict) and "data" in route_data:
            return route_data["data"]
        return route_data

    return None


def extract_match_data(data: list | str | bytes) -> dict[str, Any] | None:
    """
    Extract SendouQ match data from turbo-stream response.

    Parameters
    ----------
    data : list, str, or bytes
        Raw turbo-stream response data

    Returns
    -------
    dict or None
        Match data with 'match' key, or None if not found

    Example
    -------
    >>> response = requests.get("https://sendou.ink/q/match/123.data")
    >>> match_data = extract_match_data(response.text)
    >>> if match_data:
    ...     match = match_data["match"]
    ...     print(match["id"], match["reportedAt"])
    """
    return extract_route_data(data, "features/sendouq-match/routes/q.match.$id")


def extract_tournament_data(data: list | str | bytes) -> dict[str, Any] | None:
    """
    Extract tournament data from turbo-stream response.

    This is the main function to use for scraping tournament data.

    Parameters
    ----------
    data : list, str, or bytes
        Raw turbo-stream response data

    Returns
    -------
    dict or None
        Tournament data in the expected format with 'tournament' key,
        or None if not found

    Example
    -------
    >>> response = requests.get("https://sendou.ink/to/123/results.data")
    >>> tournament_data = extract_tournament_data(response.text)
    >>> if tournament_data:
    ...     tournament = tournament_data["tournament"]
    ...     ctx = tournament["ctx"]
    ...     print(ctx["id"], ctx["name"])
    """
    if isinstance(data, (str, bytes)):
        data = json.loads(data)

    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data).__name__}")

    decoder = TurboStreamDecoder(data)
    return decoder.get_tournament_data()
