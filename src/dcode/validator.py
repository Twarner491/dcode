"""Gcode validation and compilation for machine safety."""

import re
from dataclasses import dataclass

from .config import Config


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    corrected_gcode: str | None = None


class GcodeValidator:
    """Validates and corrects gcode for machine limits."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.wa = self.config.work_area
        self.pen = self.config.pen

    def validate(self, gcode: str, auto_correct: bool = True) -> ValidationResult:
        lines = gcode.strip().split("\n")
        errors = []
        warnings = []
        corrected_lines = []

        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith(";"):
                corrected_lines.append(line)
                continue

            corrected, line_errors, line_warnings = self._validate_line(line, i)
            errors.extend(line_errors)
            warnings.extend(line_warnings)
            corrected_lines.append(corrected if auto_correct else line)

        corrected_gcode = "\n".join(corrected_lines) if auto_correct else None
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            corrected_gcode=corrected_gcode,
        )

    def _validate_line(self, line: str, line_num: int) -> tuple[str, list[str], list[str]]:
        errors = []
        warnings = []
        corrected = line

        # Extract X, Y coordinates
        x_match = re.search(r"X([-\d.]+)", line)
        y_match = re.search(r"Y([-\d.]+)", line)
        f_match = re.search(r"F([\d.]+)", line)

        if x_match:
            x = float(x_match.group(1))
            if x < self.wa.left or x > self.wa.right:
                errors.append(f"L{line_num}: X={x} out of bounds [{self.wa.left}, {self.wa.right}]")
                x_clamped = max(self.wa.left, min(self.wa.right, x))
                corrected = re.sub(r"X[-\d.]+", f"X{x_clamped:.2f}", corrected)

        if y_match:
            y = float(y_match.group(1))
            if y < self.wa.bottom or y > self.wa.top:
                errors.append(f"L{line_num}: Y={y} out of bounds [{self.wa.bottom}, {self.wa.top}]")
                y_clamped = max(self.wa.bottom, min(self.wa.top, y))
                corrected = re.sub(r"Y[-\d.]+", f"Y{y_clamped:.2f}", corrected)

        if f_match:
            f = float(f_match.group(1))
            max_speed = max(self.pen.travel_speed, self.pen.draw_speed)
            if f > max_speed * 1.5:
                warnings.append(f"L{line_num}: F={f} unusually high (max configured: {max_speed})")
                corrected = re.sub(r"F[\d.]+", f"F{max_speed}", corrected)

        # Validate servo angles for pen commands
        m_match = re.search(r"M280\s*P0\s*S([\d.]+)", line)
        if m_match:
            angle = float(m_match.group(1))
            if angle < 0 or angle > 180:
                errors.append(f"L{line_num}: Servo angle {angle} invalid [0, 180]")
                angle_clamped = max(0, min(180, angle))
                corrected = re.sub(r"(M280\s*P0\s*S)[\d.]+", f"\\g<1>{int(angle_clamped)}", corrected)

        return corrected, errors, warnings

    def compile(self, gcode: str) -> str:
        """Validate and return corrected gcode, raising on critical errors."""
        result = self.validate(gcode, auto_correct=True)
        if result.corrected_gcode:
            return result.corrected_gcode
        return gcode

