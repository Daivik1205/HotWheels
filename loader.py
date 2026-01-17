import pickle
from pathlib import Path


class MapLoader:
    def __init__(self):
        self.grids = {}
        self.meta = {}

    def load_dir(self, root="."):
        for p in Path(root).rglob("*.bin"):
            key = p.stem       # filename is map ID
            with open(p, "rb") as f:
                data = pickle.load(f)

            self.grids[key] = data["map_data"]
            self.meta[key]  = (data["width"], data["height"])

