from rich.console import Console
import inspect
import traceback
from datetime import datetime

class Logger(Console):
    _seen_once_calls = set()
    
    def __init__(self):
        super().__init__()
        frame = inspect.currentframe()
        outer_frame = frame.f_back
        caller_self = outer_frame.f_locals.get("self", None)
        self.class_name = caller_self.__class__.__name__ if caller_self else "Global"

    def __get_call_site__(self):
        frame = inspect.currentframe()
        outer_frame = frame.f_back.f_back
        return (
            outer_frame.f_code.co_filename,
            outer_frame.f_code.co_name,
            outer_frame.f_lineno
        )
        
    @property
    def current_timestamp(self):
        return datetime.now().strftime("[%Y/%m/%d-%H:%M:%S]")
    
    def __print_once(self, message_func, *args, **kwargs):
        """Print the log only once per unique call site."""
        call_site = self.__get_call_site__()
        if call_site not in Logger._seen_once_calls:
            Logger._seen_once_calls.add(call_site)
            message_func(*args, **kwargs)
        
    def INFO(self, message: str, once = False):
        information = f"{self.current_timestamp} [color(249)][[/][color(118)]INFO[/]]    [[purple]{self.class_name}[/][color(249)]]:[/] {message}"
        if once:
            self.__print_once(lambda msg: self.print(information))    
        else: 
            self.print(information)
    
    def ERROR(self, message: str, exit_code: int = None, full_traceback: Exception = None, once: bool = False):
        def log_func(message, full_traceback):
            self.print(f"{self.current_timestamp} [color(249)][[/][color(196)]ERROR[/]]   [[purple]{self.class_name}[/][color(249)]]:[/] {message}")
            if full_traceback:
                self.print(f"[red]Exception:[/] {full_traceback}")
                self.print("".join(traceback.format_exception(type(full_traceback), full_traceback, full_traceback.__traceback__)))
        if once:
            self.__print_once(log_func, message, full_traceback)
        else:
            log_func(message, full_traceback)
        if exit_code is not None:
            exit(exit_code)

    def WARNING(self, message: str, once: bool = False):
        warning_msg = f"{self.current_timestamp} [color(249)][[/][color(220)]WARNING[/]] [[purple]{self.class_name}[/][color(249)]]:[/] {message}"
        if once:
            self.__print_once(lambda: self.print(warning_msg))
        else:
            self.print(warning_msg)

    def DEBUG(self, message: str, once: bool = False):
        debug_msg = f"{self.current_timestamp} [color(249)][[/][color(21)]DEBUG[/]]   [[purple]{self.class_name}[/][color(249)]]:[/] {message}"
        if once:
            self.__print_once(lambda: self.print(debug_msg))
        else:
            self.print(debug_msg)

    def CUSTOM(self, mode: str, message: str, once: bool = False):
        debug_msg = f"{self.current_timestamp} [color(249)][[/][color(202)]{mode}[/]] [[purple]{self.class_name}[/][color(249)]]:[/] {message}"
        if once:
            self.__print_once(lambda: self.print(debug_msg))
        else:
            self.print(debug_msg)