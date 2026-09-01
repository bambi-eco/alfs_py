import time

# Upstream imported ``src.test``, a path that never resolved (the module lived at
# ``src/alfspy/test.py``). It is now ``alfspy.scratch``; see that module's docstring.
from alfspy.scratch import main as scratch_main


def main() -> None:
    start_t = time.time()

    scratch_main()

    end_t = time.time()
    dur = end_t - start_t
    print(f'[Start: {time.ctime(start_t)} | End: {time.ctime(end_t)} | Runtime: {dur * 1000:.3f}ms]')


if __name__ == '__main__':
    main()
