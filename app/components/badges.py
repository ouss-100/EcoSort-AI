"""Category badge HTML helpers — self-contained."""
from typing import List, Set, Union

from core.detection.labels import CLASS_COLORS_HEX


def category_badges_html(detected_classes: Union[List[str], Set[str]]) -> str:
    """Return an HTML string of coloured category badges."""
    badges = []
    for c in sorted(detected_classes):
        color = CLASS_COLORS_HEX.get(c, "#9ca3af")
        badges.append(
            f'<span style="'
            f'display:inline-flex;align-items:center;gap:0.4rem;'
            f'background:#111827;border:1px solid #1f2937;'
            f'padding:0.3rem 0.65rem;border-radius:6px;'
            f'font-size:0.82rem;font-weight:600;color:#f3f4f6;'
            f'margin-right:0.4rem;margin-bottom:0.4rem;">'
            f'<span style="width:8px;height:8px;border-radius:50%;background:{color};"></span>'
            f'{c.capitalize()}</span>'
        )
    return f'<div style="margin:0.5rem 0 0.75rem 0;">{"".join(badges)}</div>'
