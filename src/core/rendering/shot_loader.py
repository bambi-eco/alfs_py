import concurrent.futures
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from src.core.rendering.shot import CtxShot


class ShotLoader(ABC, Iterable[CtxShot]):

    @abstractmethod
    def next(self) -> Optional[CtxShot]:
        pass

    def __iter__(self):
        while True:
            result = self.next()
            if result is not None:
                yield result
            else:
                break
        return


class SyncShotLoader(ShotLoader):

    def __init__(self, shots: Iterable[CtxShot]):
        """
        Allows synchronous loading of the images associated with a shot object
        :param shots: The iterable object holding all shots to be loaded
        """
        self._shots_iter = iter(shots)
        self._has_next = True

    def next(self) -> Optional[CtxShot]:
        """
        Loads the image associated with the next shot in the iterable object
        :return: ``None`` if there are no shots remaining; otherwise the loaded shot
        """
        if self._has_next:
            try:
                shot = next(self._shots_iter)
                shot.load_image()
                return shot
            except StopIteration:
                self._has_next = False
        return None


class AsyncShotLoader(ShotLoader):

    def __init__(self, shots: Iterable[CtxShot], load_count: int, max_threads: Optional[int] = None):
        """
        Allows asynchronous loading of the images associated with a shot object
        :param shots: The iterable object holding all shots to be loaded
        :param load_count: The amount of images to be loaded at the same time
        :param max_threads: The maximum amount of threads used by the executor (optional)
        """
        if load_count < 1:
            raise ValueError('The load count must be at least one')
        self._shot_iter = iter(shots)
        self._has_next = True

        # self._loop = asyncio.get_event_loop()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_threads)

        self._cur_future_index = 0
        self._future_counter = 0
        # self._shot_tasks_gen = (self._loop.run_in_executor(self._executor, self._load_shot, shot) for shot in shots)
        self._shot_future_gen = (self._executor.submit(self._load_shot, shot) for shot in shots)
        self._has_next_task = True
        self._task_dict = {}

        for _ in range(load_count):
            self._add_task()

    def __del__(self):
        task: concurrent.futures.Future
        for item in self._task_dict:
            self._task_dict[item].cancel()
        self._task_dict.clear()


    def _add_task(self):
        if self._has_next_task:
            try:
                cur_task = next(self._shot_future_gen)
                self._task_dict[self._future_counter] = cur_task
                self._future_counter += 1
            except StopIteration:
                self._has_next_task = False

    def _remove_task(self, index: int):
        if index in self._task_dict:
            del self._task_dict[index]

    @staticmethod
    def _load_shot(shot: CtxShot):
        shot.load_image()
        shot.load_tex_input()

    def next(self) -> Optional[CtxShot]:
        """
        Waits for the image associated with the next shot in the iterable object to be loaded
        :return: ``None`` if there are no shots remaining; otherwise the loaded shot
        """
        if self._has_next:
            if self._cur_future_index in self._task_dict:
                task = self._task_dict[self._cur_future_index]
                task.result()
                self._remove_task(self._cur_future_index)
                self._add_task()
                self._cur_future_index += 1
                return next(self._shot_iter)
            else:
                self._has_next = False
        return None
