import datetime
from typing import Optional


class TimeTracker:

    def __init__(self, name: Optional[str] = None, print_start_end: bool = True, print_duration: bool = True):
        """
        A time tracking context, printing time information on enter and exit.
        :param name: Name to prefix the logging with (optional).
        :param print_start_end: Whether to print the start and end date time.
        :param print_duration: Whether to print the duration between start and end time.
        """
        self._start_time = None
        self._end_time = None
        if name is None:
            name_str = ''
        else:
            name_str = f'[{name}] '
        self._name_str = name_str
        self.print_start_end = print_start_end
        self.print_duration = print_duration

    def __enter__(self):
        self._start_time = datetime.datetime.now()
        if self.print_start_end:
            print(f'{self._name_str} Start: {self._start_time}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = datetime.datetime.now()
        duration = self._end_time - self._start_time
        if self.print_start_end:
            print(f'{self._name_str}End: {self._end_time}')
        if self.print_duration:
            print(f'{self._name_str}Duration: {duration}')
        self.start_time = None
        self.end_time = None
        return False
