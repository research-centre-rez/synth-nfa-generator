import subprocess
import pathlib
import sys
# assuming the right environment has been activatedu

assert False, "this doesn't work"


def _count_vids(videos_dir):
    return len(list(videos_dir.rglob("*/*.mp4")))


videos_name = "synth_videos"
if len(sys.argv) == 2:
    videos_name = sys.argv[1]

retry_times = 100

for i in range(12):
    videos_dir = pathlib.Path(f"/disk/knotek/{videos_name}/video_{i+1:02}")
    print("starting", videos_dir)

    for _ in range(retry_times):
        process = subprocess.Popen(
            f"python app.py {str(videos_dir)} -n 3500 -f".split(),
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        output, error = process.communicate()
        print("output:", output)
        print("error:", error)

        existing_vids = _count_vids(videos_dir)
        if existing_vids == 6:
            print("Finished with", videos_dir)
            break
        else:
            print("Retrying", videos_dir)

print("DONE")
