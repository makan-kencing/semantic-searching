from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ApplicationContext:
    available_pipelines: tuple[str, ...]
