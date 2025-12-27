"""Configuration management."""

import json
from pathlib import Path

from pydantic import BaseModel


class MachineConfig(BaseModel):
    width_mm: float = 1219.2
    height_mm: float = 1524.0


class WorkAreaConfig(BaseModel):
    left: float = -420.5
    right: float = 420.5
    top: float = 594.5
    bottom: float = -594.5


class PenConfig(BaseModel):
    up_angle: int = 90
    down_angle: int = 40
    travel_speed: int = 1000
    draw_speed: int = 500


class Config(BaseModel):
    machine: MachineConfig = MachineConfig()
    work_area: WorkAreaConfig = WorkAreaConfig()
    pen: PenConfig = PenConfig()

    @classmethod
    def load(cls, path: str | Path = "configs/machine.json") -> "Config":
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path: str | Path = "configs/machine.json"):
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)

