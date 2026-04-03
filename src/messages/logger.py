import sys
import time
import traceback
from typing import Literal, Union, Optional
from rich.console import Console

class Logger:
    # Static shared state across all Logger instances
    _seen_once_calls = set()
    _call_freq = {}
    _enabled_levels = {"INFO", "ERROR", "WARNING", "DEBUG", "CUSTOM"}
    _shared_console = None

    # Timestamp caching to avoid strftime overhead every single call
    _last_ts_time = 0.0
    _last_ts_str = ""

    @classmethod
    def set_levels(cls, *levels: Literal["INFO", "ERROR", "WARNING", "DEBUG", "CUSTOM"]):
        """Enable only the specified log levels globally."""
        cls._enabled_levels = set(levels)

    @classmethod
    def get_console(cls):
        """Returns the shared console instance."""
        if cls._shared_console is None:
            cls._shared_console = Console()
        return cls._shared_console

    @classmethod
    def set_progress_console(cls, progress_console):
        """Set a specific console (e.g., from rich.progress) to be shared."""
        cls._shared_console = progress_console

    def __init__(self, name: Optional[str] = None):
        self.class_name = name or self._infer_class_name()
        # Pre-calculate the rich-formatted class tag to save time during logging
        self._class_tag = f"[[purple]{self.class_name}[/][color(249)]]:[/]"

    def _infer_class_name(self) -> str:
        """Fast inference of class name using stack frames."""
        try:
            frame = sys._getframe(2)
            if "self" in frame.f_locals:
                obj = frame.f_locals["self"]
                return f"{obj.__class__.__module__}.{obj.__class__.__name__}"
            elif "cls" in frame.f_locals:
                cls_obj = frame.f_locals["cls"]
                return f"{cls_obj.__module__}.{cls_obj.__name__}"
        except (AttributeError, ValueError):
            pass
        return "Global"

    def _get_timestamp(self) -> str:
        """Returns a cached timestamp string (valid for 1ms)."""
        now = time.time()
        if now - self._last_ts_time < 0.001:
            return self._last_ts_str
        
        ts = time.strftime('%d/%m/%Y-%H:%M:%S', time.localtime(now))
        self._last_ts_str = f"[{ts}.{int((now % 1) * 1000):03d}]"
        self._last_ts_time = now
        return self._last_ts_str

    def _should_log(self, level: str, once: bool, frequency: Optional[float]) -> bool:
        """Fast gatekeeper. Returns False immediately if level is disabled or frequency gate is closed."""
        if level not in self._enabled_levels:
            return False
        
        # Get unique call site (filename, lineno) - much faster than inspect.currentframe
        f = sys._getframe(2)
        call_site = (f.f_code.co_filename, f.f_lineno)
        
        if once:
            if call_site in self._seen_once_calls:
                return False
            self._seen_once_calls.add(call_site)
            
        if frequency:
            now = time.time()
            last_time = self._call_freq.get(call_site, 0.0)
            if now - last_time < (1.0 / frequency):
                return False
            self._call_freq[call_site] = now
            
        return True

    def _print_log(self, level_tag: str, message: tuple, color_bracket: str = "color(249)"):
        """Assembles the final string and prints to console."""
        console = self._shared_console or Logger.get_console()
        
        # Handle formatting: if msg[0] is "Val: {}" and args follow, use .format()
        if len(message) > 1 and isinstance(message[0], str) and "{}" in message[0]:
            try:
                content = message[0].format(*message[1:])
            except (IndexError, KeyError):
                content = " ".join(map(str, message))
        else:
            content = " ".join(map(str, message))

        full_msg = f"{self._get_timestamp()} [{color_bracket}][[/]{level_tag}[{color_bracket}]][/]   {self._class_tag} {content}"
        console.print(full_msg)

    def INFO(self, *message, frequency: float = None, once=False):
        if self._should_log("INFO", once, frequency):
            self._print_log(" [color(118)]INFO[/]  ", message)

    def DEBUG(self, *message, frequency: float = None, once=False):
        if self._should_log("DEBUG", once, frequency):
            self._print_log(" [color(21)]DEBUG[/] ", message)

    def WARNING(self, *message, frequency: float = None, once=False):
        if self._should_log("WARNING", once, frequency):
            self._print_log("[color(220)]WARNING[/]", message)

    def ERROR(self, *message, frequency: float = None, exit_code: int = None, full_traceback: Exception = None, once=False):
        if self._should_log("ERROR", once, frequency):
            console = Logger.get_console()
            content = " ".join(map(str, message))
            console.print(f"{self._get_timestamp()} [color(249)][[/][b][color(196)]ERROR[/][/]]   {self._class_tag} {content}")
            
            if full_traceback:
                console.print(f"[red]Exception:[/] {full_traceback}")
                console.print("".join(traceback.format_exception(type(full_traceback), full_traceback, full_traceback.__traceback__)))
            
            if exit_code is not None:
                sys.exit(exit_code)

    def CUSTOM(self, mode: str, *message, color: Union[str, int] = 209, frequency: float = None, once: bool = False):
        if self._should_log("CUSTOM", once, frequency):
            color_code = f"color({color})" if isinstance(color, int) else color
            self._print_log(f"[{color_code}]{mode}[/]", message, color_bracket=color_code)