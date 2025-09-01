import json
from pathlib import Path

from rankings.scraping.storage import save_tournament_batch


def test_save_tournament_batch_writes_json(tmp_path: Path):
    batch = [{"id": 1, "name": "T1"}, {"id": 2, "name": "T2"}]
    save_tournament_batch(batch, batch_idx=0, output_dir=str(tmp_path))
    p = tmp_path / "tournament_0.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert isinstance(data, list) and len(data) == 2
