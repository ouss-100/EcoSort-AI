"""
Possession / ownership tracking utilities.

Placeholder for future extension: maps detected waste items
to spatial zones (e.g. conveyor belt sectors, bin locations).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Zone:
    """A named rectangular zone in frame coordinates."""
    name: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, cx: int, cy: int) -> bool:
        """Return True if centroid (cx, cy) falls inside this zone."""
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class ZoneManager:
    """
    Maps detection bounding boxes to named zones.

    Usage
    -----
    zm = ZoneManager()
    zm.add_zone(Zone("recycling", 0, 0, 320, 480))
    zm.add_zone(Zone("landfill",  320, 0, 640, 480))

    for det in result.detections:
        cx = (det.bbox_xyxy[0] + det.bbox_xyxy[2]) // 2
        cy = (det.bbox_xyxy[1] + det.bbox_xyxy[3]) // 2
        zone = zm.zone_for_centroid(cx, cy)
        print(f"{det.class_name} → {zone}")
    """

    def __init__(self) -> None:
        self._zones: List[Zone] = []

    def add_zone(self, zone: Zone) -> None:
        self._zones.append(zone)

    def zone_for_centroid(self, cx: int, cy: int) -> Optional[str]:
        """Return the name of the first zone containing (cx, cy), or None."""
        for z in self._zones:
            if z.contains(cx, cy):
                return z.name
        return None

    def zone_counts(self, detections) -> Dict[str, int]:
        """
        Return a dict mapping zone_name -> detection count.

        Parameters
        ----------
        detections : iterable of Detection objects
        """
        counts: Dict[str, int] = {z.name: 0 for z in self._zones}
        for det in detections:
            cx = (det.bbox_xyxy[0] + det.bbox_xyxy[2]) // 2
            cy = (det.bbox_xyxy[1] + det.bbox_xyxy[3]) // 2
            zone = self.zone_for_centroid(cx, cy)
            if zone:
                counts[zone] = counts.get(zone, 0) + 1
        return counts

