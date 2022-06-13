import argparse
import cv2
import time

from typing import Any


def main(args: Any) -> None:
    """
    Main method.

    Parameters
    ----------
    args : Any
        Command line arguments.
    """
    camera: int = args.camera
    threshold: int = args.threshold
    refresh: int = args.refresh

    cap = cv2.VideoCapture(camera)
    _, bg = cap.read()
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    i = 0
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = cv2.absdiff(gray, bg)

        mask[mask < threshold] = 0
        mask[mask >= threshold] = 255

        cv2.imshow('Mask', mask)
        cv2.imshow('Frame', gray)
        cv2.imshow('Background', bg)

        time.sleep(0.03)
        i += 1

        if (i > refresh):
            _, bg = cap.read()
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            i = 0

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sample script of inter-frame defference detection.')
    _ = parser.add_argument('--camera', '-c', type=int, default=0, help='camera device number, by default 0.')
    _ = parser.add_argument('--threshold', '-t', type=int, default=30, help='threshold of masking image, by default 30.')
    _ = parser.add_argument('--refresh', '-r', type=int, default=30, help='refresh count of base image, by default 30.')
    args = parser.parse_args()
    main(args)
