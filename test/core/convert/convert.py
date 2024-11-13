import math
import unittest

import numpy as np

from alfspy.core.convert import change_pixel_origin, adjacent_angle
from alfspy.core.convert.data import PixelOrigin


class TestConvert(unittest.TestCase):

    def test_change_pixel_origin(self) -> None:
        for width, height in ((0, 0), (99, 99), (99, 199), (200, 99)):
            width_half = width / 2
            height_half = height / 2
            for in_args, output in (
                    ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.TopLeft), (0, 0)),
                    ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.TopCenter), (-width_half, 0)),
                    ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.TopRight), (width, 0)),
                    ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.CenterLeft), (0, height_half)),
                    ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.Center), (-width_half, height_half)),
                    ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.CenterRight), (width, height_half)),
                    ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.BottomLeft), (0, height)),
                    ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.BottomCenter), (-width_half, height)),
                    ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.BottomRight), (width, height)),
            ):
                self.assertEqual(change_pixel_origin(*in_args), output)

    def test_undistort_coords(self) -> None:
        pass # TODO

    def test_get_cos_angle(self) -> None:
        pi_fourth = math.pi / 4
        pi_eighth = math.pi / 8
        sin_ratio = np.sin(pi_eighth) / np.sin(pi_fourth)
        for side_length in (0.001, 0.1, 1, 10):
            o_side_length = side_length * sin_ratio
            for in_args, output in (
                    ((pi_fourth, o_side_length, side_length), pi_eighth),
                    ((pi_eighth, side_length, o_side_length), pi_fourth),
            ):
                self.assertAlmostEqual(adjacent_angle(*in_args), output)

    def test_cast_ray(self) -> None:
        pass # TODO

    def test_world_to_pixel_coord(self) -> None:
        pass # TODO

    def test_pixel_to_world_coord(self) -> None:
        pass # TODO
