"""Route configuration: difficulty levels and control parameters."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class RouteLevel(BaseModel):
    """Control parameters for a difficulty level."""
    action_hz: int = 10
    max_steps: int = 100
    position_threshold: float = 0.05
    replan_after_subtask: bool = False


class RouteConfig:
    """Loads and serves route configurations from YAML."""

    def __init__(self, routes: dict[str, RouteLevel]):
        self._routes = routes

    @classmethod
    def load(cls, path: Path) -> "RouteConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        # Support nested structure: default.easy, default.hard
        levels = raw.get("default", raw)
        routes = {name: RouteLevel(**params) for name, params in levels.items()}
        return cls(routes)

    @classmethod
    def default(cls) -> "RouteConfig":
        return cls({
            "easy": RouteLevel(action_hz=10, max_steps=100, position_threshold=0.05, replan_after_subtask=False),
            "hard": RouteLevel(action_hz=20, max_steps=500, position_threshold=0.03, replan_after_subtask=True),
        })

    def get_route(self, difficulty: str) -> RouteLevel:
        if difficulty not in self._routes:
            return self._routes.get("hard", RouteLevel())
        return self._routes[difficulty]

    @property
    def difficulties(self) -> list[str]:
        return list(self._routes.keys())
