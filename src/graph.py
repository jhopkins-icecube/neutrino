from dataclasses import dataclass

@dataclass
class DOMAttributes():
    zenith: float
    azimuth: float
    time: float
    track: float



class DOMGraph():

    def __init__(self) -> None:
        self.node = None
        self.edge = None

    def add_node(self, x, y, z, zenith, azimuth, time, track):
        self.node.append([x, y, z])
    