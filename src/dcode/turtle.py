"""Turtle graphics for path generation."""

from dataclasses import dataclass, field


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0


@dataclass
class Turtle:
    """Turtle graphics state machine."""

    position: Point = field(default_factory=Point)
    pen_up: bool = True
    paths: list = field(default_factory=list)
    _current_path: list = field(default_factory=list)

    def pen_down(self):
        if self.pen_up:
            self.pen_up = False
            self._current_path = [(self.position.x, self.position.y)]

    def pen_up_cmd(self):
        if not self.pen_up:
            self.pen_up = True
            if len(self._current_path) > 1:
                self.paths.append(self._current_path)
            self._current_path = []

    def move_to(self, x: float, y: float):
        self.position.x = x
        self.position.y = y
        if not self.pen_up:
            self._current_path.append((x, y))

    def jump_to(self, x: float, y: float):
        self.pen_up_cmd()
        self.position.x = x
        self.position.y = y
        self.pen_down()
        self._current_path = [(x, y)]

    def finalize(self):
        """Finalize any remaining path."""
        if len(self._current_path) > 1:
            self.paths.append(self._current_path)
            self._current_path = []

