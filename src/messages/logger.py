import inspect
import traceback
import time

from datetime import datetime
from typing import Literal, Union
from rich.console import Console

class Logger(Console):
    _seen_once_calls = set()
    
    _call_freq = dict()
    
    _enabled_levels = {"INFO", "ERROR", "WARNING", "DEBUG", "CUSTOM"}  # default all on
    _shared_console = None  # Shared console for tqdm and other tools

    @classmethod
    def set_levels(cls, 
                   *levels: Literal["INFO", "ERROR", "WARNING", "DEBUG", "CUSTOM"]
        ):
        """Enable only the given levels (others disabled)."""
        cls._enabled_levels = set(levels)
    
    @classmethod
    def get_console(cls):
        """Get the shared console instance for use with tqdm and other tools."""
        if cls._shared_console is None:
            cls._shared_console = Console()
        return cls._shared_console
    
    @classmethod
    def set_progress_console(cls, progress_console):
        """Set the Progress object's console for coordinated output."""
        cls._shared_console = progress_console
    
    def __get_console(self):
        """Get the console to use - shared if set, otherwise self."""
        if Logger._shared_console is not None:
            return Logger._shared_console
        return self
    
    def __init__(self, name = None):
        super().__init__()
        if name is None:
            frame = inspect.currentframe()
            outer_frame = frame.f_back
            while outer_frame:
                if "self" in outer_frame.f_locals:
                    obj = outer_frame.f_locals["self"]
                    module = obj.__class__.__module__
                    class_name = obj.__class__.__name__
                    self.class_name = f"{module}.{class_name}"
                    break
                elif "cls" in outer_frame.f_locals:
                    cls = outer_frame.f_locals["cls"]
                    module = cls.__module__
                    class_name = cls.__name__
                    self.class_name = f"{module}.{class_name}"
                    break
                outer_frame = outer_frame.f_back
            else:
                self.class_name = "Global"
        else:
            self.class_name = name
            
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
    
    def __print_freq(self, message_func, frequency, *args, **kwargs):
        call_site = self.__get_call_site__()
        try:
            time_stamp = Logger._call_freq[call_site]
            if time.time() - time_stamp >= (1 / frequency):
                message_func(*args, **kwargs)
                Logger._call_freq[call_site] = time.time()
        except:
            Logger._call_freq[call_site] = time.time()
            message_func(*args, **kwargs)
        
    def INFO(self, *message, frequency: float = None, once = False):
        if "INFO" not in Logger._enabled_levels:
            return
        console = self.__get_console()
        concat_message = " ".join([str(mess) for mess in message])
        information = f"{self.current_timestamp} [color(249)][[/][color(118)]INFO[/]]    [[purple]{self.class_name}[/][color(249)]]:[/] {concat_message}"
        if once:
            self.__print_once(lambda: console.print(information))    
        elif frequency:
            self.__print_freq(lambda: console.print(information), frequency)
        else: 
            console.print(information)
    
    def ERROR(self, *message, frequency: float = None, exit_code: int = None, full_traceback: Exception = None, once: bool = False):
        if "ERROR" not in Logger._enabled_levels:
            return
        console = self.__get_console()
        concat_message = " ".join([str(mess) for mess in message])
        def log_func(concat_message, full_traceback):
            console.print(f"{self.current_timestamp} [color(249)][[/][b][color(196)]ERROR[/][/]]   [[purple]{self.class_name}[/][color(249)]]:[/] {concat_message}")
            if full_traceback:
                console.print(f"[red]Exception:[/] {full_traceback}")
                console.print("".join(traceback.format_exception(type(full_traceback), full_traceback, full_traceback.__traceback__)))
        if once:
            self.__print_once(log_func, concat_message, full_traceback)
        elif frequency:
            self.__print_freq(log_func, frequency)
        else:
            log_func(concat_message, full_traceback)
        if exit_code is not None:
            exit(exit_code)

    def WARNING(self, *message, frequency: float = None, once: bool = False):
        if "WARNING" not in Logger._enabled_levels:
            return
        console = self.__get_console()
        concat_message = " ".join([str(mess) for mess in message])
        warning_msg = f"{self.current_timestamp} [color(249)][[/][color(220)]WARNING[/]] [[purple]{self.class_name}[/][color(249)]]:[/] {concat_message}"
        if once:
            self.__print_once(lambda: console.print(warning_msg))
        elif frequency:
            self.__print_freq(lambda: console.print(warning_msg), frequency)
        else:
            console.print(warning_msg)

    def DEBUG(self, *message, frequency: float = None, once: bool = False):
        if "DEBUG" not in Logger._enabled_levels:
            return
        console = self.__get_console()
        concat_message = " ".join([str(mess) for mess in message])
        debug_msg = f"{self.current_timestamp} [color(249)][[/][color(21)]DEBUG[/]]   [[purple]{self.class_name}[/][color(249)]]:[/] {concat_message}"
        if once:
            self.__print_once(lambda: console.print(debug_msg))
        elif frequency:
            self.__print_freq(lambda: console.print(debug_msg), frequency)
        else:
            console.print(debug_msg)

    def CUSTOM(self, mode: str, *message, color: Union[str, int] = 209, frequency: float = None, once: bool = False):
        if "CUSTOM" not in Logger._enabled_levels:
            return
        console = self.__get_console()
        concat_message = " ".join([str(mess) for mess in message])
        color_code = f"color({color})" if isinstance(color, int) else color
        debug_msg = f"{self.current_timestamp} [{color_code}][[/][{color_code}]{mode}[/]] [[purple]{self.class_name}[/][color(249)]]:[/] {concat_message}"
        if once:
            self.__print_once(lambda: console.print(debug_msg))
        elif frequency:
            self.__print_freq(lambda: console.print(debug_msg), frequency)
        else:
            console.print(debug_msg)