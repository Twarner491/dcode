"""Image to path converter - ported from JS ImageConverter."""

import math
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
    box_size: float = 8.0
    cutoff: int = 128
    threshold: int = 128  # For trace_outline
    density: float = 2.0  # For trace_outline fill spacing
    to_corners: bool = False
    fill: bool = False  # For trace_outline: fill interior


CONVERTERS = {
    "spiral": {"name": "Spiral", "defaults": {"step_size": 2.0, "to_corners": False}},
    "crosshatch": {"name": "Crosshatch", "defaults": {"angle": 45, "step_size": 2.0, "passes": 4}},
    "squares": {"name": "Concentric Squares", "defaults": {"box_size": 8.0, "cutoff": 128}},
    "trace": {"name": "Trace Outline", "defaults": {"threshold": 128, "density": 2.0, "fill": False}},
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
    "squares": [
        {"box_size": 6.0, "cutoff": 100},
        {"box_size": 8.0, "cutoff": 128},
        {"box_size": 10.0, "cutoff": 150},
    ],
    "trace": [
        {"threshold": 100, "density": 1.5, "fill": False},
        {"threshold": 128, "density": 2.0, "fill": False},
        {"threshold": 150, "density": 2.5, "fill": True},
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

    def _convert_trace(self, gray, w, h, ox, oy, opts: ConverterOptions) -> Turtle:
        """Trace outline converter.
        
        Thresholds image to binary, then traces edges using scan-line approach.
        Each contiguous run of edge pixels becomes a line segment.
        
        For fills, scans the full binary mask (not just edges) with spacing
        controlled by density parameter.
        """
        turtle = Turtle()
        threshold = opts.threshold
        density = opts.density
        do_fill = opts.fill
        
        # Threshold to binary: dark pixels = shape (1), light = background (0)
        binary = (gray < threshold).astype(np.uint8)
        
        if do_fill:
            # Fill mode: scan entire shape
            mask = binary
        else:
            # Edge mode: find edge pixels
            # Edge = dark pixel with at least one 4-connected light neighbor
            mask = np.zeros_like(binary)
            for iy in range(h):
                for ix in range(w):
                    if binary[iy, ix] == 1:
                        # Check 4-connected neighbors
                        is_edge = False
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = iy + dy, ix + dx
                            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                                is_edge = True  # Out of bounds = edge
                                break
                            if binary[ny, nx] == 0:
                                is_edge = True  # Light neighbor = edge
                                break
                        if is_edge:
                            mask[iy, ix] = 1
        
        # Horizontal scan: for each row, find contiguous runs of mask pixels
        step = max(1, int(density))
        for iy in range(0, h, step):
            segments = []
            in_segment = False
            seg_start = 0
            
            for ix in range(w):
                if mask[iy, ix] == 1:
                    if not in_segment:
                        in_segment = True
                        seg_start = ix
                else:
                    if in_segment:
                        in_segment = False
                        segments.append((seg_start, ix - 1))
            
            if in_segment:
                segments.append((seg_start, w - 1))
            
            # Draw each segment
            for start_x, end_x in segments:
                if end_x > start_x:  # At least 2 pixels
                    # Convert image coords to world coords
                    x0 = ox + start_x
                    x1 = ox + end_x
                    y = oy + (h - 1 - iy)  # Flip Y
                    
                    turtle.jump_to(x0, y)
                    turtle.move_to(x1, y)
        
        # Vertical scan: for each column, find contiguous runs
        for ix in range(0, w, step):
            segments = []
            in_segment = False
            seg_start = 0
            
            for iy in range(h):
                if mask[iy, ix] == 1:
                    if not in_segment:
                        in_segment = True
                        seg_start = iy
                else:
                    if in_segment:
                        in_segment = False
                        segments.append((seg_start, iy - 1))
            
            if in_segment:
                segments.append((seg_start, h - 1))
            
            # Draw each segment
            for start_y, end_y in segments:
                if end_y > start_y:  # At least 2 pixels
                    x = ox + ix
                    y0 = oy + (h - 1 - start_y)  # Flip Y
                    y1 = oy + (h - 1 - end_y)
                    
                    turtle.jump_to(x, y0)
                    turtle.move_to(x, y1)
        
        return turtle

