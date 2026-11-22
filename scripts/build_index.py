import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingestion.contextual_indexer import build_index
print("indexed chunks:", build_index(reset=True))
