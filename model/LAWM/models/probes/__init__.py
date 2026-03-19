import importlib
import inspect
import pkgutil
import re

from .base import Prober

_PROBE_REGISTRY: dict[str, type[Prober]] = {}
_DISCOVERED = False


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _aliases_for_class_name(class_name: str) -> set[str]:
    base = class_name[:-5] if class_name.endswith("Probe") else class_name
    snake_base = _camel_to_snake(base)
    snake_class = _camel_to_snake(class_name)

    return {
        class_name,
        class_name.lower(),
        base,
        base.lower(),
        f"{base}Probe",
        f"{base.lower()}probe",
        f"{base}_probe",
        f"{base.lower()}_probe",
        snake_base,
        f"{snake_base}_probe",
        snake_class,
    }


def _register_probe_class(cls: type[Prober]) -> None:
    for alias in _aliases_for_class_name(cls.__name__):
        _PROBE_REGISTRY[_normalize_name(alias)] = cls


def discover_probe_classes(force_refresh: bool = False) -> dict[str, type[Prober]]:
    global _DISCOVERED

    if _DISCOVERED and not force_refresh:
        return dict(_PROBE_REGISTRY)

    _PROBE_REGISTRY.clear()

    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_") or mod.name == "base":
            continue

        module = importlib.import_module(f"{__name__}.{mod.name}")
        for value in vars(module).values():
            if not inspect.isclass(value):
                continue
            if not issubclass(value, Prober) or value is Prober:
                continue
            if value.__module__ != module.__name__:
                continue
            _register_probe_class(value)

    _DISCOVERED = True
    return dict(_PROBE_REGISTRY)


def get_probe_class(name: str) -> type[Prober]:
    if not name:
        raise ValueError("Probe class name must be a non-empty string")

    if not _DISCOVERED:
        discover_probe_classes()

    normalized = _normalize_name(name)
    cls = _PROBE_REGISTRY.get(normalized)
    if cls is not None:
        return cls

    discover_probe_classes(force_refresh=True)
    cls = _PROBE_REGISTRY.get(normalized)
    if cls is not None:
        return cls

    available = ", ".join(available_probe_classes())
    raise KeyError(f"Unknown probe class '{name}'. Available probes: {available}")


def build_probe(name: str, **kwargs) -> Prober:
    return get_probe_class(name)(**kwargs)


def available_probe_classes() -> list[str]:
    if not _DISCOVERED:
        discover_probe_classes()
    return sorted({cls.__name__ for cls in _PROBE_REGISTRY.values()})


__all__ = [
    "Prober",
    "discover_probe_classes",
    "get_probe_class",
    "build_probe",
    "available_probe_classes",
]
