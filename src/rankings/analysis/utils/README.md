# Utils Module

The utils module provides utility functions for tournament ranking analysis, organized into focused submodules for better maintainability.

## Module Structure

### summaries.py
Functions for preparing comprehensive summary DataFrames:
- `prepare_player_summary()` - Create player rankings with statistics
- `prepare_team_summary()` - Create team rankings with statistics  
- `prepare_tournament_summary()` - Create tournament summaries with strength metrics

### formatting.py
Functions for formatting data for human-readable display:
- `format_top_rankings()` - Format rankings as a nice string for display

### comparisons.py
Functions for comparing rankings and analyzing relationships:
- `compare_rankings()` - Compare two ranking systems and show differences
- `get_head_to_head_record()` - Get head-to-head record between two entities

### matches.py
Functions for analyzing individual matches:
- `get_most_influential_matches()` - Find matches with highest impact on a player's ranking
- `get_player_match_history()` - Get complete match history with names and context

### names.py
Functions for adding human-readable names to DataFrames:
- `add_player_names()` - Add player usernames to DataFrames with user IDs
- `add_team_names()` - Add team names to DataFrames with team IDs
- `add_tournament_names()` - Add tournament names from raw JSON data
- `add_match_timestamps()` - Convert Unix timestamps to readable dates
- `create_match_summary_with_names()` - Create comprehensive match summaries

## Usage Examples

```python
from rankings.analysis.utils import (
    prepare_player_summary,
    get_most_influential_matches,
    add_tournament_names
)

# Prepare player summary
player_summary = prepare_player_summary(players_df, rankings_df)

# Get influential matches with names
influential = get_most_influential_matches(player_id=123, ...)
wins_with_names = add_tournament_names(influential["wins"], tournament_data)
```

## Import Structure

All functions are available through the main utils module:

```python
from rankings.analysis.utils import <function_name>
```

Or you can import from specific submodules:

```python
from rankings.analysis.utils.summaries import prepare_player_summary
from rankings.analysis.utils.names import add_player_names
```