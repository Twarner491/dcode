"""Image to path converter - ported from JS ImageConverter."""

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import Config
from .turtle import Turtle


@dataclass
class ConverterOptions:
    step_size: float = 2.0
    angle: float = 45.0
    passes: int = 4
    amplitude: float = 5.0
    box_size: float = 8.0
    cutoff: int = 128
    turns: int = 5000
    to_corners: bool = False


CONVERTERS = {
    "spiral": {"name": "Spiral", "defaults": {"step_size": 2.0, "to_corners": False}},
    "crosshatch": {"name": "Crosshatch", "defaults": {"angle": 45, "step_size": 2.0, "passes": 4}},
    "pulse": {"name": "Pulse Lines", "defaults": {"step_size": 3.0, "amplitude": 5.0}},
    "squares": {"name": "Concentric Squares", "defaults": {"box_size": 8.0, "cutoff": 128}},
    "wander": {"name": "Random Walk", "defaults": {"step_size": 1.0, "turns": 5000}},
}

# Permutations for dataset generation
ALGORITHM_PERMUTATIONS = {
    "spiral": [
        {"step_size": 1.5, "to_corners": False},
        {"step_size": 2.5, "to_corners": False},
        {"step_size": 3.5, "to_corners": True},
    ],
    "crosshatch": [
        {"angle": 30, "step_size": 2.0, "passes": 3},
        {"angle": 45, "step_size": 1.5, "passes": 4},
        {"angle": 60, "step_size": 2.5, "passes": 5},
    ],
    "pulse": [
        {"step_size": 2.0, "amplitude": 4.0},
        {"step_size": 3.0, "amplitude": 6.0},
        {"step_size": 4.0, "amplitude": 8.0},
    ],
    "squares": [
        {"box_size": 6.0, "cutoff": 100},
        {"box_size": 8.0, "cutoff": 128},
        {"box_size": 10.0, "cutoff": 150},
    ],
    "wander": [
        {"step_size": 0.8, "turns": 8000},
        {"step_size": 1.2, "turns": 5000},
    ],
}


class ImageConverter:
    """Converts raster images to vector paths."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.work_area = self.config.work_area

    def convert(
        self, image: Image.Image, algorithm: str, options: dict | None = None
    ) -> Turtle:
        """Convert image to turtle paths."""
        opts = ConverterOptions(**(options or {}))

        # Calculate target size within work area
        work_w = self.work_area.right - self.work_area.left
        work_h = self.work_area.top - self.work_area.bottom

        img_aspect = image.width / image.height
        work_aspect = work_w / work_h

        if img_aspect > work_aspect:
            new_w = int(work_w)
            new_h = int(new_w / img_aspect)
        else:
            new_h = int(work_h)
            new_w = int(new_h * img_aspect)

        # Resize and convert to grayscale
        image = image.convert("L").resize((new_w, new_h), Image.Resampling.LANCZOS)
        gray = np.array(image, dtype=np.uint8)

        offset_x = -new_w / 2
        offset_y = -new_h / 2

        method = getattr(self, f"_convert_{algorithm}", None)
        if method is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        turtle = method(gray, new_w, new_h, offset_x, offset_y, opts)
        turtle.finalize()
        return turtle

    def _sample(self, gray: np.ndarray, w: int, h: int, x: float, y: float, ox: float, oy: float) -> int:
        ix = int(x - ox)
        iy = int(h - 1 - (y - oy))
        if 0 <= ix < w and 0 <= iy < h:
            return int(gray[iy, ix])
        return 255

    def _convert_spiral(self, gray, w, h, ox, oy, opts: ConverterOptions) -> Turtle:
        turtle = Turtle()
        step = opts.step_size
        cx, cy = ox + w / 2, oy + h / 2

        if opts.to_corners:
            max_r = math.sqrt((w / 2) ** 2 + (h / 2) ** 2)
        else:
            max_r = min(w, h) / 2

        r = max_r
        while r > step:
            circumference = 2 * math.pi * r
            steps = max(1, int(circumference / step))

            for i in range(steps):
                p = i / steps
                angle = 2 * math.pi * p
                r1 = r - step * p

                fx = math.cos(angle) * r1
                fy = math.sin(angle) * r1
                x, y = cx + fx, cy + fy

                ix = int(fx + w / 2)
                iy = int(h / 2 - fy)

                if 0 <= ix < w and 0 <= iy < h:
                    brightness = gray[iy, ix]
                    level = 128 + 64 * math.sin(angle * 4)
                    if brightness < level:
                        turtle.pen_down()
                    else:
                        turtle.pen_up_cmd()
                else:
                    turtle.pen_up_cmd()

                turtle.move_to(x, y)

            r -= step

        return turtle

    def _convert_crosshatch(self, gray, w, h, ox, oy, opts: ConverterOptions) -> Turtle:
        turtle = Turtle()
        step = opts.step_size
        passes = opts.passes
        base_angle = opts.angle

        max_len = math.sqrt(w**2 + h**2)

        for pass_num in range(passes):
            angle = math.radians(base_angle + 180 * pass_num / passes)
            dx, dy = math.cos(angle), math.sin(angle)
            level = 255 * (1 + pass_num) / (passes + 1)

            a = -max_len
            while a < max_len:
                px, py = dx * a, dy * a
                x0 = px - dy * max_len + ox + w / 2
                y0 = py + dx * max_len + oy + h / 2
                x1 = px + dy * max_len + ox + w / 2
                y1 = py - dx * max_len + oy + h / 2

                self._convert_along_line(turtle, gray, w, h, x0, y0, x1, y1, step, level, ox, oy)
                a += step

        return turtle

    def _convert_along_line(self, turtle, gray, w, h, x0, y0, x1, y1, step, cutoff, ox, oy):
        dx, dy = x1 - x0, y1 - y0
        dist = math.sqrt(dx**2 + dy**2)
        if dist < step:
            return

        steps = int(dist / step)
        for i in range(steps + 1):
            t = i / steps
            x, y = x0 + dx * t, y0 + dy * t
            brightness = self._sample(gray, w, h, x, y, ox, oy)

            if brightness < cutoff:
                if turtle.pen_up:
                    turtle.jump_to(x, y)
                else:
                    turtle.move_to(x, y)
            else:
                if not turtle.pen_up:
                    turtle.pen_up_cmd()
                turtle.position.x, turtle.position.y = x, y

    def _convert_pulse(self, gray, w, h, ox, oy, opts: ConverterOptions) -> Turtle:
        turtle = Turtle()
        step = opts.step_size
        max_amp = opts.amplitude

        row = 0
        y = oy
        while y < oy + h:
            if row % 2 == 0:
                x_range = range(int(ox), int(ox + w))
            else:
                x_range = range(int(ox + w), int(ox), -1)

            first = True
            for x in x_range:
                ix = int(x - ox)
                iy = int(h - 1 - (y - oy))

                if 0 <= ix < w and 0 <= iy < h:
                    brightness = gray[iy, ix]
                    amplitude = max_amp * (255 - brightness) / 255
                    wave = math.sin(x * 0.5) * amplitude
                    py = y + wave

                    if first:
                        turtle.jump_to(x, py)
                        first = False
                    else:
                        turtle.move_to(x, py)

            y += step
            row += 1

        return turtle

    def _convert_squares(self, gray, w, h, ox, oy, opts: ConverterOptions) -> Turtle:
        turtle = Turtle()
        box = opts.box_size
        cutoff = opts.cutoff
        half = box / 2

        row = 0
        y = oy + half
        while y < oy + h - half:
            if row % 2 == 0:
                x_iter = list(np.arange(ox + half, ox + w - half, box))
            else:
                x_iter = list(np.arange(ox + w - half, ox + half, -box))

            for x in x_iter:
                brightness = self._sample(gray, w, h, x, y, ox, oy)
                if brightness < cutoff:
                    size = half * (cutoff - brightness) / cutoff
                    if size > 0.5:
                        turtle.jump_to(x - size, y - size)
                        turtle.move_to(x + size, y - size)
                        turtle.move_to(x + size, y + size)
                        turtle.move_to(x - size, y + size)
                        turtle.move_to(x - size, y - size)

            y += box
            row += 1

        return turtle

    def _convert_wander(self, gray, w, h, ox, oy, opts: ConverterOptions) -> Turtle:
        turtle = Turtle()
        step = opts.step_size
        max_turns = opts.turns

        x, y = ox + w / 2, oy + h / 2
        angle = random.random() * 2 * math.pi

        turtle.jump_to(x, y)

        for _ in range(max_turns):
            brightness = self._sample(gray, w, h, x, y, ox, oy)
            turn = (brightness / 255.0) * math.pi / 2
            angle += (random.random() - 0.5) * turn

            nx = x + math.cos(angle) * step
            ny = y + math.sin(angle) * step

            if nx < ox or nx > ox + w:
                angle = math.pi - angle
                nx = x + math.cos(angle) * step
            if ny < oy or ny > oy + h:
                angle = -angle
                ny = y + math.sin(angle) * step

            x, y = nx, ny

            if self._sample(gray, w, h, x, y, ox, oy) < 200:
                if turtle.pen_up:
                    turtle.jump_to(x, y)
                else:
                    turtle.move_to(x, y)
            else:
                turtle.pen_up_cmd()
                turtle.position.x, turtle.position.y = x, y

        return turtle

