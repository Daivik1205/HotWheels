import csv


class TeleportResolver:
    def __init__(self):
        self.links = {}

    def load(self, filename="teleport.csv"):
        with open(filename) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                tid = row[0]
                self.links[tid] = [x for x in row[1:] if x]

    def edges(self):
        out = []
        for tid, ds in self.links.items():
            for a in ds:
                for b in ds:
                    if a != b:
                        out.append((a, b, tid))
        return out

