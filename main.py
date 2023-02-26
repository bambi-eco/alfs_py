import time

import src.test as src


def main() -> None:
    start_t = time.time()

    src.main()

    end_t = time.time()
    dur = end_t - start_t
    print(f'[Start: {time.ctime(start_t)} | End: {time.ctime(end_t)} | Runtime: {dur*1000:.3f}ms]')


if __name__ == '__main__':
    main()


# Korrektur:
# - Varianz im Bild als metrik für schärfe implementieren
# - Mehrere Rotationen um X-Y-Z-Achse der Drohen durchtesten um schärfe zu maximieren (+ - 5°)
# - Edge Detector auf shots anwenden
# - Scipy minimum function optimierung auf winkel anwenden

