from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path(f"outputs_{str(datetime.now().strftime('%d-%m-%y_%Hh%Mm%Ss'))}")
