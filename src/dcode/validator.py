"""Gcode validation and compilation for machine safety."""

import re
from dataclasses import dataclass, field

from .config import Config


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    corrected_gcode: str | None = None
    stats: dict = field(default_factory=dict)


class GcodeValidator:
    """Validates and corrects gcode for machine limits."""

    # Valid gcode commands for polargraph
    VALID_COMMANDS = {"G0", "G1", "G4", "G21", "G28", "G90", "G91", "M84", "M280"}

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.wa = self.config.work_area
        self.pen = self.config.pen

    def validate(self, gcode: str, auto_correct: bool = True) -> ValidationResult:
        """Validate gcode and optionally correct issues."""
        lines = gcode.strip().split("\n")
        errors = []
        warnings = []
        corrected_lines = []
        stats = {"lines": len(lines), "moves": 0, "pen_ups": 0, "pen_downs": 0}

        for i, line in enumerate(lines, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith(";"):
                corrected_lines.append(line)
                continue

            corrected, line_errors, line_warnings, line_stats = self._validate_line(
                line, i
            )
            errors.extend(line_errors)
            warnings.extend(line_warnings)
            corrected_lines.append(corrected if auto_correct else line)

            # Update stats
            for k, v in line_stats.items():
                stats[k] = stats.get(k, 0) + v

        corrected_gcode = "\n".join(corrected_lines) if auto_correct else None

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            corrected_gcode=corrected_gcode,
            stats=stats,
        )

    def _validate_line(
        self, line: str, line_num: int
    ) -> tuple[str, list[str], list[str], dict]:
        """Validate a single gcode line."""
        errors = []
        warnings = []
        corrected = line
        stats = {}

        # Extract command
        parts = line.split()
        if not parts:
            return corrected, errors, warnings, stats

        cmd = parts[0].upper()

        # Check for valid command
        cmd_base = re.match(r"([GM]\d+)", cmd)
        if cmd_base:
            if cmd_base.group(1) not in self.VALID_COMMANDS:
                warnings.append(f"L{line_num}: Unknown command {cmd_base.group(1)}")

        # Validate coordinates
        x_match = re.search(r"X([-\d.]+)", line, re.IGNORECASE)
        y_match = re.search(r"Y([-\d.]+)", line, re.IGNORECASE)
        f_match = re.search(r"F([\d.]+)", line, re.IGNORECASE)

        if x_match:
            try:
                x = float(x_match.group(1))
                if x < self.wa.left or x > self.wa.right:
                    errors.append(
                        f"L{line_num}: X={x:.2f} out of bounds [{self.wa.left}, {self.wa.right}]"
                    )
                    x_clamped = max(self.wa.left, min(self.wa.right, x))
                    corrected = re.sub(
                        r"X[-\d.]+", f"X{x_clamped:.2f}", corrected, flags=re.IGNORECASE
                    )
            except ValueError:
                errors.append(f"L{line_num}: Invalid X value")

        if y_match:
            try:
                y = float(y_match.group(1))
                if y < self.wa.bottom or y > self.wa.top:
                    errors.append(
                        f"L{line_num}: Y={y:.2f} out of bounds [{self.wa.bottom}, {self.wa.top}]"
                    )
                    y_clamped = max(self.wa.bottom, min(self.wa.top, y))
                    corrected = re.sub(
                        r"Y[-\d.]+", f"Y{y_clamped:.2f}", corrected, flags=re.IGNORECASE
                    )
            except ValueError:
                errors.append(f"L{line_num}: Invalid Y value")

        if f_match:
            try:
                f = float(f_match.group(1))
                max_speed = max(self.pen.travel_speed, self.pen.draw_speed)
                if f > max_speed * 2:
                    warnings.append(
                        f"L{line_num}: F={f:.0f} very high (max: {max_speed})"
                    )
                    corrected = re.sub(
                        r"F[\d.]+", f"F{max_speed}", corrected, flags=re.IGNORECASE
                    )
                elif f <= 0:
                    errors.append(f"L{line_num}: F={f} invalid (must be > 0)")
                    corrected = re.sub(
                        r"F[\d.]+",
                        f"F{self.pen.draw_speed}",
                        corrected,
                        flags=re.IGNORECASE,
                    )
            except ValueError:
                errors.append(f"L{line_num}: Invalid F value")

        # Track moves
        if cmd in ("G0", "G1"):
            stats["moves"] = 1

        # Validate servo angles
        m280_match = re.search(r"M280\s*P0\s*S([-\d.]+)", line, re.IGNORECASE)
        if m280_match:
            try:
                angle = float(m280_match.group(1))
                if angle < 0 or angle > 180:
                    errors.append(f"L{line_num}: Servo angle {angle} invalid [0, 180]")
                    angle_clamped = max(0, min(180, angle))
                    corrected = re.sub(
                        r"(M280\s*P0\s*S)[-\d.]+",
                        f"\\g<1>{int(angle_clamped)}",
                        corrected,
                        flags=re.IGNORECASE,
                    )

                # Track pen state
                if abs(angle - self.pen.up_angle) < 5:
                    stats["pen_ups"] = 1
                elif abs(angle - self.pen.down_angle) < 5:
                    stats["pen_downs"] = 1
            except ValueError:
                errors.append(f"L{line_num}: Invalid servo angle")

        return corrected, errors, warnings, stats

    def compile(self, gcode: str) -> str:
        """Validate and return corrected gcode."""
        result = self.validate(gcode, auto_correct=True)
        return result.corrected_gcode or gcode

    def add_safety_wrapper(self, gcode: str) -> str:
        """Add safety header and footer to gcode."""
        header = f"""; Safety wrapper added by dcode
G21 ; mm units
G90 ; absolute positioning
M280 P0 S{self.pen.up_angle} ; pen up
G28 ; home

"""
        footer = f"""
; End safety wrapper
M280 P0 S{self.pen.up_angle} ; pen up
G0 X0 Y0 F{self.pen.travel_speed} ; return home
M84 ; motors off
"""
        # Check if gcode already has header
        if not gcode.strip().startswith(";") and not gcode.strip().startswith("G21"):
            gcode = header + gcode

        # Check if gcode has footer
        if "M84" not in gcode:
            gcode = gcode + footer

        return gcode
