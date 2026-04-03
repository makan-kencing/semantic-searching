from dataclasses import dataclass
from app.engine.base import SearchEngine


@dataclass(slots=True, frozen=True)
class ApplicationContext:
    available_engines: tuple[type[SearchEngine]]
