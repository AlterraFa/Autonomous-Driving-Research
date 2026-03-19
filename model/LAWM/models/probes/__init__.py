import importlib
import inspect
import pkgutil
import re

from .base import Prober
from utils.logger import Logger

_PROBE_REGISTRY: dict[str, type[Prober]] = {}
_DISCOVERED = False
_logger = Logger(__name__)


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
    probe_cls = get_probe_class(name)
    sig = inspect.signature(probe_cls.__init__)

    accepted_args = set()
    has_var_kwargs = False
    for param in sig.parameters.values():
        if param.name in {"self", "args", "kwargs"}:
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kwargs = True
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        accepted_args.add(param.name)

    used_kwargs = {key: value for key, value in kwargs.items() if key in accepted_args}
    unused_kwargs = {key: value for key, value in kwargs.items() if key not in accepted_args}

    _logger.INFO(f"Building probe '{probe_cls.__name__}'")
    _logger.INFO("Used args:", used_kwargs if used_kwargs else {})

    if unused_kwargs:
        _logger.WARNING("Unused args:", unused_kwargs)

    if has_var_kwargs and unused_kwargs:
        _logger.DEBUG(
            f"{probe_cls.__name__} accepts **kwargs; unused args are intentionally not forwarded to avoid silent misconfiguration."
        )

    return probe_cls(**used_kwargs)


def available_probe_classes() -> list[str]:
    if not _DISCOVERED:
        discover_probe_classes()
    return sorted({cls.__name__ for cls in _PROBE_REGISTRY.values()})


def probe_constructor_args() -> dict[str, list[str]]:
    args_map: dict[str, list[str]] = {}
    for class_name in available_probe_classes():
        cls = get_probe_class(class_name)
        sig = inspect.signature(cls.__init__)
        args = []
        for param in sig.parameters.values():
            if param.name in {"self", "args", "kwargs"}:
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            args.append(param.name)
        args_map[class_name] = args
    return args_map


def probe_argument_report() -> dict[str, object]:
    args_map = probe_constructor_args()
    if not args_map:
        return {
            "richest_probe": None,
            "common_args": [],
            "args_by_probe": {},
        }

    richest_name = max(args_map, key=lambda key: len(args_map[key]))
    common_args = set(next(iter(args_map.values())))
    for args in args_map.values():
        common_args &= set(args)

    return {
        "richest_probe": richest_name,
        "common_args": sorted(common_args),
        "args_by_probe": args_map,
    }


__all__ = [
    "Prober",
    "discover_probe_classes",
    "get_probe_class",
    "build_probe",
    "available_probe_classes",
    "probe_constructor_args",
    "probe_argument_report",
]
