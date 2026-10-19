from pathlib import Path
from typing import Sequence

import cv2
import imageio
import numpy as np
from tqdm.auto import tqdm


def frames_to_video(video_path: str, frames: Sequence[np.ndarray], fps=25):
    writer = None
    for frame in frames:
        if writer is None:
            res_y, res_x = frame.shape[:2]
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (res_x, res_y),
            )

        writer.write(frame)
    if writer is not None:
        writer.release()


def frames_to_oio(frames: np.ndarray, teeth_shift: int):
    n, h, w, _ = frames.shape
    pano_height = h + teeth_shift * (n - 1)
    oio = np.zeros((pano_height, w, 3), dtype=np.float32)

    slits_per_height = h / teeth_shift

    for row_id in range(pano_height):
        sh = int(np.ceil(slits_per_height))
        collected_rows = []
        slit_row = row_id % teeth_shift
        middle_frame = row_id // teeth_shift

        # adjust range so that it's not being wasted
        for frame_offset in range(sh):
            frame_id = middle_frame - frame_offset

            pf_row_id = frame_offset * teeth_shift + slit_row

            if not (0 <= frame_id < n) or (not (0 <= pf_row_id < h)):
                continue

            current_frame = frames[frame_id, :, :, :3]
            row = current_frame[pf_row_id]
            collected_rows.append((pf_row_id, row))

        if len(collected_rows) == 1:
            _, calc_row = collected_rows[0]
        else:
            # Instead of calculating mean per row I weight them based on cosine
            row_ratio = np.array([r[0] / h for r in collected_rows])
            cos_xs = np.pi * (row_ratio * 2 - 1)
            cos_ys = (np.cos(cos_xs) + 1) / 2
            cos_ys_norm = cos_ys / np.sum(cos_ys)

            just_rows = np.array([r[1] for r in collected_rows])
            calc_row = np.sum(just_rows * cos_ys_norm[:, None, None], axis=0)

        oio[row_id] = calc_row
        return oio


def load_frames_as_numpy(path: Path):
    paths = list(sorted(Path(path).glob("*.png")))
    first = imageio.imread(paths[0])

    frames = np.empty(
        (len(paths), *first.shape),
        dtype=np.uint8,
    )

    frames[0] = first

    for i, path in enumerate(tqdm(paths[1:], initial=1, total=len(paths)), start=1):
        frames[i] = imageio.imread(path)
    return frames
