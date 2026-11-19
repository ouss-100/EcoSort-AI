from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Zone:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, cx: int, cy: int) -> bool:
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class ZoneManager:
    def __init__(self) -> None:
        self._zones: List[Zone] = []

    def add_zone(self, zone: Zone) -> None:
        self._zones.append(zone)

    def zone_for_centroid(self, cx: int, cy: int) -> Optional[str]:
        for z in self._zones:
            if z.contains(cx, cy):
                return z.name
        return None

    def zone_counts(self, detections) -> Dict[str, int]:
        counts: Dict[str, int] = {z.name: 0 for z in self._zones}
        for det in detections:
            cx = (det.bbox_xyxy[0] + det.bbox_xyxy[2]) // 2
            cy = (det.bbox_xyxy[1] + det.bbox_xyxy[3]) // 2
            zone = self.zone_for_centroid(cx, cy)
            if zone:
                counts[zone] = counts.get(zone, 0) + 1
        return counts

