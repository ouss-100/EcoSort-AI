from model.labels import (
    CLASS_COLORS_BGR,
    CLASS_COLORS_HEX,
    CLASSES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    NUM_CLASSES,
    get_color_bgr,
    get_color_hex,
)


def test_num_classes():
    assert NUM_CLASSES == 6


def test_classes_list():
    expected = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]
    assert CLASSES == expected


def test_class_to_id_and_back():
    for i, cls in enumerate(CLASSES):
        assert CLASS_TO_ID[cls] == i
        assert ID_TO_CLASS[i] == cls


def test_hex_colors_format():
    for cls, hex_color in CLASS_COLORS_HEX.items():
        assert hex_color.startswith("#"), f"{cls} hex color missing '#'"
        assert len(hex_color) == 7, f"{cls} hex color wrong length"


def test_bgr_colors_type():
    for cls, bgr in CLASS_COLORS_BGR.items():
        assert isinstance(bgr, tuple), f"{cls} BGR not a tuple"
        assert len(bgr) == 3, f"{cls} BGR should have 3 channels"
        for v in bgr:
            assert 0 <= v <= 255, f"{cls} BGR value {v} out of range"


def test_get_color_hex_known():
    assert get_color_hex("plastic") == "#e74c3c"
    assert get_color_hex("glass") == "#3498db"


def test_get_color_hex_unknown():
    result = get_color_hex("unknown_class")
    assert isinstance(result, str)
    assert result.startswith("#")


def test_get_color_bgr_known():
    bgr = get_color_bgr("biodegradable")
    assert isinstance(bgr, tuple)
    assert len(bgr) == 3


def test_get_color_bgr_unknown():
    bgr = get_color_bgr("not_a_class")
    assert isinstance(bgr, tuple)
    assert len(bgr) == 3

