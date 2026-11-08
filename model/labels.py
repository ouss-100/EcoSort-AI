CLASSES = [
    "biodegradable",
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]

NUM_CLASSES = len(CLASSES)
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}
ID_TO_CLASS = {i: cls for i, cls in enumerate(CLASSES)}

CLASS_COLORS_HEX = {
    "biodegradable": "#27ae60",  # green
    "cardboard":     "#e67e22",  # orange
    "glass":         "#3498db",  # blue
    "metal":         "#95a5a6",  # silver
    "paper":         "#f1c40f",  # yellow
    "plastic":       "#e74c3c",  # red
}

CLASS_COLORS_BGR = {
    "biodegradable": (96,  174, 39),
    "cardboard":     (34,  126, 230),
    "glass":         (219, 152, 52),
    "metal":         (166, 165, 149),
    "paper":         (15,  196, 241),
    "plastic":       (60,  76,  231),
}


def get_color_hex(class_name: str) -> str:
    return CLASS_COLORS_HEX.get(class_name, "#95a5a6")


def get_color_bgr(class_name: str) -> tuple:
    return CLASS_COLORS_BGR.get(class_name, (166, 165, 149))
