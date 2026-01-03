"""GCode generation from turtle paths."""

from .config import Config
from .turtle import Turtle


class GcodeExporter:
    """Exports turtle paths to gcode."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.pen = self.config.pen

    def export(self, turtle: Turtle, comment: str = "") -> str:
        """Convert turtle paths to gcode string."""
        lines = []

        # Minimal header - no metadata that could leak into training
        if comment:
            lines.append(f"; {comment}")
        lines.append("; dcode")
        lines.append("")
        lines.append("G21 ; mm")
        lines.append("G90 ; absolute")
        lines.append(f"M280 P0 S{self.pen.up_angle} ; pen up")
        lines.append("G28 ; home")
        lines.append("")

        for path in turtle.paths:
            if len(path) < 2:
                continue

            # Move to start with pen up
            x0, y0 = path[0]
            lines.append(f"G0 X{x0:.2f} Y{y0:.2f} F{self.pen.travel_speed}")
            lines.append(f"M280 P0 S{self.pen.down_angle} ; pen down")

            # Draw path
            for x, y in path[1:]:
                lines.append(f"G1 X{x:.2f} Y{y:.2f} F{self.pen.draw_speed}")

            # Pen up
            lines.append(f"M280 P0 S{self.pen.up_angle} ; pen up")
            lines.append("")

        # Footer
        lines.append("G0 X0 Y0 F{} ; return home".format(self.pen.travel_speed))
        lines.append(f"M280 P0 S{self.pen.up_angle} ; pen up")
        lines.append("M84 ; motors off")

        return "\n".join(lines)

