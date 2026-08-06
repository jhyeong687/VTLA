#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import numpy as np

from lerobot_robot_paxini.paxini_reader import SerialPaxiniReader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print live Paxini tactile statistics from the serial sensor."
    )
    parser.add_argument("--port", default="/dev/cu.usbmodem750A687485311")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--num-taxels", type=int, default=154)
    parser.add_argument("--mode", choices=["x", "y", "z", "magnitude"], default="z")
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--timeout-s", type=float, default=0.5)
    parser.add_argument(
        "--samples", type=int, default=0, help="Stop after this many samples; 0 means run forever."
    )
    parser.add_argument(
        "--sync", action="store_true", help="Read synchronously instead of using the background reader."
    )
    args = parser.parse_args()

    reader = SerialPaxiniReader(
        args.port,
        args.baudrate,
        num_taxels=args.num_taxels,
        timeout_s=args.timeout_s,
        taxel_value_mode=args.mode,
        taxel_scale=args.scale,
        async_read=not args.sync,
    )

    delay_s = 1.0 / max(args.hz, 0.1)
    print(f"Opening Paxini tactile sensor on {args.port} @ {args.baudrate} baud. Press Ctrl-C to stop.")
    reader.connect()
    try:
        count = 0
        while True:
            sample = reader.read_latest()
            taxels = sample.taxels.astype(np.float32, copy=False)
            nonzero = int(np.count_nonzero(np.abs(taxels) > 1e-6))
            print(
                f"t={sample.timestamp_s:.3f} "
                f"min={taxels.min(initial=0.0):8.3f} "
                f"mean={taxels.mean():8.3f} "
                f"max={taxels.max(initial=0.0):8.3f} "
                f"nonzero={nonzero:3d}/{taxels.size}",
                flush=True,
            )
            count += 1
            if args.samples and count >= args.samples:
                break
            time.sleep(delay_s)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        reader.disconnect()


if __name__ == "__main__":
    main()
