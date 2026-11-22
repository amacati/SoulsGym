import numpy as np
import pytest

from soulsgym.core.game_window import GameWindow


def test_resolution_override_is_instance_local():
    default_resolutions = GameWindow._resolve_game_resolutions("EldenRing", None)
    overridden_resolutions = GameWindow._resolve_game_resolutions("EldenRing", (1024, 576))
    subsequent_default_resolutions = GameWindow._resolve_game_resolutions("EldenRing", None)

    assert default_resolutions["EldenRing"] == (800, 450)
    assert overridden_resolutions["EldenRing"] == (1024, 576)
    assert subsequent_default_resolutions == default_resolutions
    assert GameWindow.game_resolution["EldenRing"] == (800, 450)


@pytest.mark.parametrize(
    "game_resolution",
    [
        [800, 450],
        (800,),
        (800, 0),
        (800, -1),
        (800.0, 450),
        (True, 450),
    ],
)
def test_resolution_override_requires_two_positive_integers(game_resolution):
    with pytest.raises(ValueError, match="two positive integers"):
        GameWindow._resolve_game_resolutions("EldenRing", game_resolution)


def _window_with_raw_img(img):
    class _Window(GameWindow):
        raw_img = property(lambda self: img)

    window = object.__new__(_Window)
    window.game_id = "EldenRing"
    window.game_resolution = GameWindow._resolve_game_resolutions("EldenRing", None)
    return window


def _image_with_black_padding(shape, top=0, bottom=0, left=0, right=0):
    img = np.zeros((*shape, 3), dtype=np.uint8)
    img[top : shape[0] - bottom, left : shape[1] - right] = 1
    return img


@pytest.mark.parametrize("from_start, expected", [(True, 2), (False, 3)])
def test_edge_padding_length_counts_only_contiguous_edge_padding(from_start, expected):
    mask = np.array([True, True, False, True, False, True, True, True])

    assert GameWindow._edge_padding_length(mask, from_start) == expected


@pytest.mark.parametrize(
    "img_shape, padding, expected_crop",
    [
        ((481, 808), {}, ([30, 480], [4, 804])),
        ((481, 809), {"left": 3, "right": 2}, ([30, 480], [5, 805])),
        ((488, 804), {"top": 4, "bottom": 3}, ([34, 484], [2, 802])),
        ((490, 810), {"top": 5, "bottom": 3, "left": 2, "right": 4}, ([36, 486], [4, 804])),
    ],
)
def test_image_crop_removes_black_padding_on_any_axis(img_shape, padding, expected_crop):
    window = _window_with_raw_img(_image_with_black_padding(img_shape, **padding))

    crop_heights, crop_widths = window._determine_image_crop()

    assert (crop_heights.tolist(), crop_widths.tolist()) == expected_crop