import threading

class Extractor:
    def __init__(self, attr_list: dict = None):
        self._latest = None
        self._lock = threading.Lock()
        self._have_sample = threading.Event()

        self._attr_list = attr_list
        
    def put(self, data):
        with self._lock:
            self._latest = data
            self._have_sample.set()
        
    def get(self, wait: bool = True, timeout: float = 0.5):
        if wait and not self._have_sample.wait(timeout):
            return None
        with self._lock:
            data = self._latest
        if data is None:
            return None
        
        if self._attr_list is None or (len(self._attr_list) != 0 and isinstance(self._attr_list, dict) == False): 
            return data
        else:
            data_load = {}
            for shorthand, fullname in self._attr_list.items():
                data_load[shorthand] = getattr(data, fullname)
            return data_load