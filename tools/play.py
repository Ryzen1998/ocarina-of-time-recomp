#!/usr/bin/env python3
# Written for ocarina-of-time-recomp (MIT, Copyright (c) 2026 Jeff Aigner).
"""Play the game from an input script, record it and write sheets an agent can look at.

    .venv-win\\Scripts\\python.exe tools\\play.py run --route "at 62s; press A; walk up 3s & camera left 1s; wait 8s"
    .venv-win\\Scripts\\python.exe tools\\play.py run --script build/logs/user-play-3.txt --from none --seconds 300
    .venv-win\\Scripts\\python.exe tools\\play.py compose --route "walk up 3s" --out build/logs/walk.txt
    .venv-win\\Scripts\\python.exe tools\\play.py sheets VIDEO --burst 45,auto --crop 0.5,0,0.5,0.5

A run goes through tools/run.py (run_game), so it refuses to overlap another run. It waits until the card is free,
never kills a game it did not start, and refuses to start when the driver has reset since the last boot. Output goes to
build/reference/play/<run>/: video.mp4, contact.png, burst-<t>.png, flicker.csv, flicker-heatmap.png,
flicker-timeline.png, input.txt, capture.log and run.json. After every run it prints those paths, the frame rate lines
of the log and the number of nvlddmkm events.

Route language: one step per line or separated by ";", and "&" starts steps at the same moment (the next line starts
after the longest). A duration or time is in 60 Hz ticks ("90") or seconds ("1.5s").
    wait D                 let time pass
    at T                   continue at this absolute time since the first present ("at 62s")
    press BUTTON [D]       A B X Y L R ZL ZR PLUS MINUS LCLICK RCLICK UP DOWN LEFT RIGHT; hold 10 ticks by default
    walk DIR D [PCT]       left stick; DIR is up down left right upleft upright downleft downright or x,y (-32767..32767)
    camera DIR D [PCT]     right stick, same directions; PCT is the stick travel, 100 by default
    raw LINE               a line of the replay format as it is (runtime/input_win.c, script_load)

The replay format has two clocks, plain ticks and "@ms", and both count display time from the first present
(runtime/frame_rate.c, frame_step_world_us), not from the moment a save is loaded. A route therefore has to allow for the
load: "--from autosave" (default) presses A through the title until tick 3000 with the movies skipped. In run-744 the
player stood idle in Kokiri Forest from about 45 s, and the time depends on the autosave; a route that needs the player in
control should open with "at 62s" or later, and check the contact sheet.

Capture is the game window only, found by its exact title and owner (oot.exe) and recorded by its handle through
Windows.Graphics.Capture (ffmpeg's gfxcapture filter). --capture game instead saves the presents from the runtime
(OOT_SHOTS); it costs frame rate but does not depend on the screen being on.
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as runner  # noqa: E402  tools/run.py: every run of the game goes through its run_game

ROOT = runner.ROOT
OUT = ROOT / "build" / "reference" / "play"
TITLE = "Ocarina of Time Recomp"
TICKS = 60
BUTTONS = {"A", "B", "X", "Y", "L", "R", "ZL", "ZR", "PLUS", "MINUS", "LCLICK", "RCLICK", "UP", "DOWN", "LEFT", "RIGHT"}
STICK = 32767
# Screen directions; stick y is positive up (input_win.c negates the SDL axis).
DIRS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0), "upleft": (-0.7071, 0.7071),
        "upright": (0.7071, 0.7071), "downleft": (-0.7071, -0.7071), "downright": (0.7071, -0.7071)}
AUTOSAVE_END_TICK = 3000   # the last A of the walk through the title, build/logs/a-continue.txt


def ticks(text: str) -> int:
    """A duration in ticks from "90" or "1.5s"."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(s?)", text)
    if not m:
        raise ValueError(f"not a duration: {text!r} (ticks like 90 or seconds like 1.5s)")
    return round(float(m[1]) * (TICKS if m[2] else 1))


def stick_xy(direction: str, pct: float) -> tuple[int, int]:
    if "," in direction:
        x, y = (int(v) for v in direction.split(","))
        return x, y
    if direction not in DIRS:
        raise ValueError(f"unknown direction {direction!r}: {', '.join(DIRS)} or x,y")
    dx, dy = DIRS[direction]
    return round(dx * STICK * pct / 100), round(dy * STICK * pct / 100)


def compose(route: str, start: int = 0) -> tuple[list[str], int]:
    """Replay-format lines from a route, and the tick at which it ends."""
    out, now = [], start
    for raw in re.split(r"[;\n]", route):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        longest = 0
        for step in line.split("&"):
            w = step.split()
            if not w:
                continue
            verb, args = w[0].lower(), w[1:]
            try:
                if verb == "wait" and len(args) == 1:
                    longest = max(longest, ticks(args[0]))
                elif verb == "at" and len(args) == 1:
                    now = ticks(args[0])
                elif verb == "press" and 1 <= len(args) <= 2 and args[0].upper() in BUTTONS:
                    hold = ticks(args[1]) if len(args) == 2 else 10
                    out.append(f"{now} {args[0].upper()} {hold}")
                    longest = max(longest, hold + 10)   # a gap, so that two presses in a row are two presses
                elif verb in ("walk", "camera") and 2 <= len(args) <= 3:
                    hold = ticks(args[1])
                    x, y = stick_xy(args[0].lower(), float(args[2]) if len(args) == 3 else 100.0)
                    out.append(f"{now} {'LSTICK' if verb == 'walk' else 'RSTICK'} {x} {y} {hold}")
                    longest = max(longest, hold)
                elif verb == "raw":
                    out.append(" ".join(args))
                else:
                    raise ValueError("unknown step or wrong number of arguments")
            except ValueError as e:
                raise ValueError(f"route step {step.strip()!r}: {e}") from None
        now += longest
    return out, now


def autosave_prefix() -> list[str]:
    """A every 120 ticks from tick 600 through the title's Continue (the contents of build/logs/a-continue.txt)."""
    return [f"{t} A 10" for t in range(600, AUTOSAVE_END_TICK + 1, 120)]


def write_script(path: Path, route: str | None, script: str | None, start_from: str) -> int:
    """Writes the input script for a run; returns its last tick (0 for a raw script, whose end is not known)."""
    lines = autosave_prefix() if start_from == "autosave" else []
    end = 0
    if script:
        src = Path(script) if Path(script).is_absolute() else ROOT / script
        lines += src.read_text().splitlines()
    if route:
        routed, end = compose(route, AUTOSAVE_END_TICK + 60 if start_from == "autosave" else 0)
        lines += routed
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return end


# --- the host: driver resets, the run lock, the window ---------------------------------------------------------------

def powershell(command: str) -> str:
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], capture_output=True, text=True).stdout.strip()


def driver_resets(since: str) -> int:
    """nvlddmkm events in the System log since an ISO time, or since the last boot for since="boot"."""
    t = "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime" if since == "boot" else f"[datetime]'{since}'"
    out = powershell(f"@(Get-WinEvent -FilterHashtable @{{LogName='System'; ProviderName='nvlddmkm'; StartTime={t}}} "
                     "-ErrorAction SilentlyContinue).Count")
    return int(out) if out.isdigit() else -1


def wait_until_free(max_wait: float, settle: float = 10.0) -> bool:
    """Waits for no game to run for settle seconds, so a session running tests back to back keeps the card."""
    deadline, free_since, said = time.monotonic() + max_wait, None, False
    while time.monotonic() < deadline:
        if runner.running():
            free_since = None
            if not said:
                print("play: another run is in progress; waiting for it to end", flush=True)
                said = True
        else:
            free_since = free_since or time.monotonic()
            if time.monotonic() - free_since >= settle and exe_settled():
                return True
        time.sleep(2)
    return False


def exe_settled() -> bool:
    """oot.exe was not just rewritten (a session may be building it)."""
    return runner.EXE.exists() and time.time() - runner.EXE.stat().st_mtime > 20


def find_window() -> int:
    """The handle of the game's window: the exact title, and owned by oot.exe. 0 when there is none."""
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    user32.FindWindowW.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    hwnd = user32.FindWindowW(None, TITLE)
    if not hwnd:
        return 0
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    process = kernel32.OpenProcess(0x1000, False, pid.value)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not process:
        return 0
    name, size = ctypes.create_unicode_buffer(1024), wintypes.DWORD(1024)
    ok = kernel32.QueryFullProcessImageNameW(process, 0, name, ctypes.byref(size))
    kernel32.CloseHandle(process)
    return int(hwnd) if ok and Path(name.value).name.lower() == "oot.exe" else 0


def session_interactive() -> bool:
    """False when the desktop is locked: GDI capture of the window gets nothing then."""
    import ctypes
    h = ctypes.windll.user32.OpenInputDesktop(0, False, 0x0100)   # DESKTOP_SWITCHDESKTOP
    if not h:
        return False
    ok = bool(ctypes.windll.user32.SwitchDesktop(h))
    ctypes.windll.user32.CloseDesktop(h)
    return ok


class WindowCapture:
    """ffmpeg recording the game window alone, through Windows.Graphics.Capture (the gfxcapture filter), by window handle.

    Not gdigrab: it copies through GDI, which never sees what Vulkan presents, and on 2026-11-09 its "window" capture
    held what was on the desktop behind the game. gfxcapture takes the window's own surface from the compositor.
    """

    def __init__(self, video: Path, fps: int, crf: int, hwnd: int):
        self.video, self.log = video, video.with_name("capture.log")
        source = f"gfxcapture=hwnd={hwnd}:max_framerate={fps}:capture_cursor=0,hwdownload,format=bgra,fps={fps},"
        self.args = ["ffmpeg", "-hide_banner", "-y", "-filter_complex", source + "crop=trunc(iw/2)*2:trunc(ih/2)*2",
                     "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-threads", "4",
                     "-movflags", "+frag_keyframe+empty_moov", str(video)]
        self.proc = None

    def start(self) -> None:
        self.logf = open(self.log, "w")
        self.proc = subprocess.Popen(self.args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.logf)

    def stop(self) -> None:
        if not self.proc:
            return
        try:
            self.proc.stdin.write(b"q")
            self.proc.stdin.flush()
        except OSError:
            pass   # ffmpeg already ended with the window
        try:
            self.proc.wait(20)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(10)
        self.logf.close()


def game_shots_env(frames: Path, seconds: float, step: float, full: bool) -> dict:
    """OOT_SHOTS by display time: the first present at or after each step (runtime/boot.c, series_shot)."""
    env = {"OOT_SHOTS": f"w0-{seconds:g}:{step:g}", "OOT_SHOTS_DIR": frames.relative_to(ROOT).as_posix()}
    if full:
        env["OOT_SHOTS_FULL"] = "1"
    return env


def shots_to_video(frames: Path, video: Path, step: float) -> bool:
    names = sorted(p.name for p in frames.glob("f*.png"))
    if not names:
        return False
    lst = frames / "list.txt"
    lst.write_text("".join(f"file '{n}'\nduration {step:g}\n" for n in names) + f"file '{names[-1]}'\n")
    subprocess.run(["ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-vf",
                    f"fps={1 / step:g},crop=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "16", "-pix_fmt", "yuv420p", str(video.resolve())], cwd=frames, capture_output=True)
    return video.exists()


def fps_lines(log: str) -> list[str]:
    text = Path(log).read_text(errors="replace") if log and Path(log).exists() else ""
    return re.findall(r"^(?:\[host\] )?(frametime: (?:60 s|exit):.*)$", text, re.M)


# --- reading the video -----------------------------------------------------------------------------------------------

def probe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(video)],
                         capture_output=True, text=True).stdout
    data = json.loads(out)
    s, num_den = data["streams"][0], data["streams"][0]["avg_frame_rate"].split("/")
    return {"width": s["width"], "height": s["height"], "fps": int(num_den[0]) / max(1, int(num_den[1])),
            "duration": float(data["format"]["duration"])}


def frames_raw(video: Path, width: int, height: int, pix: str, vf_pre: str = "", seek: float | None = None,
               limit: int | None = None):
    """Yields numpy frames of the video, gray or rgb24, scaled by area average."""
    import numpy as np
    depth = 1 if pix == "gray" else 3
    args = ["ffmpeg", "-v", "error"] + (["-ss", f"{seek:.3f}"] if seek is not None else []) + ["-i", str(video)]
    args += ["-vf", (vf_pre + "," if vf_pre else "") + f"scale={width}:{height}:flags=area", "-an", "-fps_mode", "passthrough"]
    args += (["-frames:v", str(limit)] if limit else []) + ["-f", "rawvideo", "-pix_fmt", pix, "pipe:1"]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = width * height * depth
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            a = np.frombuffer(buf, np.uint8)
            yield a.reshape(height, width) if depth == 1 else a.reshape(height, width, 3)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def duplicate(a, b) -> bool:
    """The same game frame captured twice (the capture runs faster than the game; the codec leaves a little noise)."""
    import numpy as np
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) < 0.35


def font(size: int):
    from PIL import ImageFont
    return ImageFont.load_default(size=size)


def label(img, text: str, size: int = 14) -> None:
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    box = d.textbbox((0, 0), text, font=font(size))
    d.rectangle((0, 0, box[2] + 6, box[3] + 6), fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 0), font=font(size))


def grid_sheet(cells: list, cols: int):
    from PIL import Image
    h = cells[0].height
    w = cells[0].width
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols - 1) * 2, rows * h + (rows - 1) * 2), (40, 40, 40))
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % cols) * (w + 2), (i // cols) * (h + 2)))
    return sheet


def contact_sheet(video: Path, out: Path, cols: int = 5, rows: int = 6, width: int = 320) -> Path:
    """Frames at even intervals of the whole video, each labelled with its time."""
    from PIL import Image
    info = probe(video)
    height = round(width * info["height"] / info["width"]) & ~1
    cells = []
    for i in range(cols * rows):
        t = info["duration"] * (i + 0.5) / (cols * rows)
        frame = next(frames_raw(video, width, height, "rgb24", seek=t, limit=1), None)
        if frame is None:
            continue
        img = Image.fromarray(frame)
        label(img, f"{t:.1f} s", 13)
        cells.append(img)
    grid_sheet(cells, cols).save(out)
    return out


def crop_filter(crop: str | None) -> str:
    if not crop:
        return ""
    x, y, w, h = (float(v) for v in crop.split(","))
    return f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}"


def burst_sheet(video: Path, t: float, out_dir: Path, n: int = 9, cols: int = 3, crop: str | None = None,
                width: int = 640) -> Path:
    """n successive different frames from time t; a crop (fractions x,y,w,h of the frame) shows them larger."""
    from PIL import Image
    info = probe(video)
    aspect = info["height"] / info["width"]
    if crop:
        _, _, cw, ch = (float(v) for v in crop.split(","))
        aspect *= ch / cw
    height = round(width * aspect) & ~1
    cells, last, i = [], None, -1
    for i, frame in enumerate(frames_raw(video, width, height, "rgb24", crop_filter(crop), seek=t, limit=n * 12)):
        if last is not None and duplicate(frame, last):
            continue
        last = frame
        img = Image.fromarray(frame)
        label(img, f"{t + i / info['fps']:.3f} s  (+{len(cells)})")
        cells.append(img)
        if len(cells) == n:
            break
    out = out_dir / (f"burst-{t:.2f}" + ("-crop" if crop else "") + ".png")
    if cells:
        grid_sheet(cells, cols).save(out)
    return out


def flicker(video: Path, out_dir: Path, width: int = 480, height: int = 270, still_median: float = 2.5) -> dict:
    """Per-region frame-to-frame change, over the distinct frames of the video.

    A 16x9 grid. "change" is the mean absolute difference from the previous frame. "reversal" is the part of it that
    goes back: pixels that moved one way and then the other over two steps, the size of the smaller step, ignoring
    steps under 6 of 255 (the codec's noise). A flicker gives a large reversal; a camera move gives change but mostly
    no reversal. A frame counts as near-still when the median cell changes by less than still_median.
    """
    import numpy as np
    from PIL import Image, ImageDraw
    info = probe(video)
    cw, ch = width // 16, height // 9
    a = b = None
    rows, reversal_sum, still_n, peak, seen, unique, dark = [], np.zeros((9, 16)), 0, (-1.0, 0.0), 0, 0, 0
    for i, frame in enumerate(frames_raw(video, width, height, "gray")):
        seen += 1
        if float(frame.mean()) < 5:
            dark += 1
        if b is not None and duplicate(frame, b):
            continue
        unique += 1
        if a is not None:
            d1 = b.astype(np.int16) - a.astype(np.int16)
            d2 = frame.astype(np.int16) - b.astype(np.int16)
            big = (np.abs(d1) >= 6) & (np.abs(d2) >= 6) & (np.sign(d1) != np.sign(d2))
            rev = np.minimum(np.abs(d1), np.abs(d2)) * big
            block = lambda m: m[:9 * ch, :16 * cw].reshape(9, ch, 16, cw).mean(axis=(1, 3))
            change_c, rev_c = block(np.abs(d2).astype(np.float32)), block(rev.astype(np.float32))
            med = float(np.median(change_c))
            t = i / info["fps"]
            still = med < still_median
            top = np.unravel_index(int(rev_c.argmax()), rev_c.shape)
            rows.append((round(t, 3), int(still), round(med, 3), round(float(rev_c.mean()), 3), int(top[1]), int(top[0]),
                         round(float(rev_c.max()), 3)))
            if still:
                reversal_sum += rev_c
                still_n += 1
                if float(rev_c.mean()) > peak[0]:
                    peak = (float(rev_c.mean()), t)
        a, b = b, frame
    if not rows:   # the picture never changed: a capture that got nothing from the game
        return {"frames": seen, "unique": unique, "dark_fraction": round(dark / max(1, seen), 3), "frozen": True}
    with open(out_dir / "flicker.csv", "w", newline="") as f:
        csv.writer(f).writerows([("t_s", "still", "median_cell_change", "mean_reversal", "top_col", "top_row", "top_reversal")] + rows)
    mean = reversal_sum / max(1, still_n)
    summary = {"frames": seen, "unique": unique, "video_seconds": round(info["duration"], 1),
               "distinct_fps": round(unique / max(info["duration"], 0.001), 1), "dark_fraction": round(dark / seen, 3),
               "still_frames": still_n, "peak_reversal_time": round(peak[1], 3) if still_n else None,
               "still_mean_reversal": round(float(mean.mean()), 3)}
    order = np.argsort(mean, axis=None)[::-1][:8]
    summary["top_cells"] = [{"col": int(c), "row": int(r), "x_pct": round(c / 16 * 100), "y_pct": round(r / 9 * 100),
                             "reversal": round(float(mean[r, c]), 3)} for r, c in (np.unravel_index(k, mean.shape) for k in order)]

    # The heat map: the reversal per cell over the near-still frames, drawn over a frame from the middle of them.
    still_times = [r[0] for r in rows if r[1]] or [rows[len(rows) // 2][0]]
    backdrop = next(frames_raw(video, 1280, 720, "rgb24", seek=still_times[len(still_times) // 2], limit=1))
    img = Image.fromarray(backdrop).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    scale = max(float(mean.max()), 0.001)
    for r in range(9):
        for c in range(16):
            x, y = c * 80, r * 80
            d.rectangle((x, y, x + 79, y + 79), fill=(255, 0, 0, int(170 * mean[r, c] / scale)), outline=(255, 255, 255, 50))
            d.text((x + 4, y + 3), f"{mean[r, c]:.2f}", fill=(255, 255, 255, 255), font=font(13))
    Image.alpha_composite(img, overlay).convert("RGB").save(out_dir / "flicker-heatmap.png")

    # The timeline: median change (camera motion) over the mean reversal (flicker), with the near-still frames marked.
    w, h, im = 1600, 300, Image.new("RGB", (1600, 300), (20, 20, 20))
    d = ImageDraw.Draw(im)
    end = rows[-1][0] or 1.0
    top_change = max(max(r[2] for r in rows), 0.1)
    top_rev = max(max(r[3] for r in rows), 0.01)
    for r in rows:
        x = round(r[0] / end * (w - 1))
        if r[1]:
            d.line((x, 0, x, 8), fill=(70, 200, 70))
        d.line((x, 145, x, 145 - round(r[2] / top_change * 125)), fill=(90, 150, 255))
        d.line((x, 295, x, 295 - round(r[3] / top_rev * 125)), fill=(255, 110, 70))
    d.text((5, 12), f"median cell change (camera motion), max {top_change:.2f}; green = near-still", fill=(255, 255, 255), font=font(13))
    d.text((5, 152), f"mean reversal (flicker), max {top_rev:.2f}; {end:.1f} s across", fill=(255, 255, 255), font=font(13))
    im.save(out_dir / "flicker-timeline.png")
    return summary


def make_sheets(video: Path, out_dir: Path | None = None, bursts: str = "auto", burst_frames: int = 9, crop: str | None = None,
                with_flicker: bool = True) -> dict:
    """The contact sheet, the flicker measure and burst sheets; "auto" in bursts means the strongest near-still flicker."""
    video = Path(video).resolve()
    out_dir = Path(out_dir) if out_dir else video.parent
    info = probe(video)
    result = {"video": str(video), "video_info": info, "contact": str(contact_sheet(video, out_dir / "contact.png"))}
    fl = flicker(video, out_dir) if with_flicker else {}
    result["flicker"] = fl
    times = []
    for item in (s.strip() for s in bursts.split(",")):
        if item == "auto":
            if fl.get("peak_reversal_time") is not None:
                times.append(max(0.0, fl["peak_reversal_time"] - 0.1))
            times.append(info["duration"] * 0.5)
            times.append(info["duration"] * 0.85)
        elif item:
            times.append(float(item))
    result["bursts"] = [str(burst_sheet(video, min(t, max(0.0, info["duration"] - 0.5)), out_dir, burst_frames, crop=crop)) for t in times]
    if with_flicker:
        result["flicker_heatmap"], result["flicker_timeline"] = str(out_dir / "flicker-heatmap.png"), str(out_dir / "flicker-timeline.png")
    return result


# --- a run -----------------------------------------------------------------------------------------------------------

def play(route: str | None = None, script: str | None = None, start_from: str = "autosave", seconds: int | None = None,
         label_text: str = "route", fps: int = 60, crf: int = 18, capture: str = "auto", shot_step: float = 0.1,
         shots_full: bool = False, env: dict | None = None, bursts: str = "auto", burst_frames: int = 9, crop: str | None = None,
         retry: int = 0, max_wait: float = 3600.0) -> dict:
    """One run from an input script to its video and sheets."""
    if sys.platform != "win32":
        sys.exit("play: this tool runs the game on native Windows (tools/play.py needs user32 and gdigrab)")
    if not shutil.which("ffmpeg"):
        sys.exit("play: ffmpeg is not on PATH")
    resets = driver_resets("boot")
    if resets > 0:
        return {"ok": False, "refused": f"{resets} driver reset(s) since the last boot: the PC has to be rebooted before the next GPU run (CLAUDE.md, GPU safety)"}
    run_dir = OUT / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"\W+", "-", label_text))
    run_dir.mkdir(parents=True)
    script_path = run_dir / "input.txt"
    end_tick = write_script(script_path, route, script, start_from)
    if seconds is None:
        seconds = max(30, end_tick // TICKS + 5)
    video, frames = run_dir / "video.mp4", run_dir / "frames"

    if capture == "auto":
        capture = "window" if session_interactive() else "game"
        if capture == "game":
            print("play: the desktop is locked, so the window cannot be captured; saving frames from the runtime", flush=True)
    child_env = dict(env or {})
    if capture == "game":
        frames.mkdir()
        child_env.update(game_shots_env(frames, seconds, shot_step, shots_full))

    # Wait for the card, then run; a session that wins the race between our check and run_game makes run.py refuse, and
    # then we go back to waiting.
    deadline = time.monotonic() + max_wait
    while True:
        if not wait_until_free(max(1.0, deadline - time.monotonic())):
            return {"ok": False, "refused": "the card did not become free in time", "run_dir": str(run_dir)}
        t0 = datetime.now().replace(microsecond=0).isoformat()
        box: dict = {}

        def work() -> None:
            try:
                box["result"] = runner.run_game(seconds, env=child_env, retry=retry,
                                                replay=script_path.relative_to(ROOT).as_posix(), skip_movies=start_from == "autosave")
            except SystemExit as e:
                box["refused"] = str(e)

        thread = threading.Thread(target=work)
        thread.start()
        time.sleep(1.0)
        recorder = None
        while thread.is_alive():
            hwnd = find_window() if capture == "window" and not recorder else 0
            if hwnd:
                recorder = WindowCapture(video, fps, crf, hwnd)
                recorder.start()
                print(f"play: recording the window \"{TITLE}\" of oot.exe", flush=True)
            time.sleep(0.25)
        thread.join()
        if recorder:
            recorder.stop()
        if "refused" in box and "refusing to overlap" in box["refused"]:
            print("play: another run started first; waiting again", flush=True)
            continue
        break

    if "refused" in box:
        return {"ok": False, "refused": box["refused"], "run_dir": str(run_dir)}
    r = box["result"]
    reset_count = driver_resets(t0)
    if capture == "game":
        shots_to_video(frames, video, shot_step)
    result = {"ok": bool(r["ok"]) and reset_count == 0, "run_dir": str(run_dir), "capture": capture, "input": str(script_path), "seconds": seconds,
              "log": r["log"], "exit": r["exit"], "seconds_run": r.get("seconds_run"), "fps": fps_lines(r["log"]), "driver_resets": reset_count}
    if reset_count:
        result["warning"] = "DRIVER RESET: stop all GPU runs, do not retry, and have the PC rebooted (CLAUDE.md, GPU safety)"
    if video.exists() and video.stat().st_size > 1000:
        result.update(make_sheets(video, run_dir, bursts, burst_frames, crop))
        fl = result.get("flicker", {})
        if fl.get("frozen") or fl.get("dark_fraction", 0) > 0.9:
            result["ok"] = False
            result["warning"] = (result.get("warning", "") + " The video is black or does not change: window capture got nothing "
                                 "from the game (screen off or locked?); try --capture game.").strip()
    else:
        result["warning"] = (result.get("warning", "") + " No video was recorded (see capture.log).").strip()
    (run_dir / "run.json").write_text(json.dumps(result, indent=2))
    return result


def report(result: dict) -> int:
    if result.get("refused"):
        print(f"play: REFUSED: {result['refused']}")
        return 2
    print(f"play: {'ok' if result['ok'] else 'FAILED'}: {result['run_dir']}")
    for key in ("video", "contact", "flicker_heatmap", "flicker_timeline", "input", "log"):
        if result.get(key):
            print(f"  {key}: {result[key]}")
    for path in result.get("bursts", []):
        print(f"  burst: {path}")
    fl = result.get("flicker", {})
    if fl:
        print(f"  frames: {fl.get('unique')} distinct of {fl.get('frames')} captured ({fl.get('distinct_fps')} distinct per second), "
              f"near-still {fl.get('still_frames')}, dark {fl.get('dark_fraction')}, mean reversal when still {fl.get('still_mean_reversal')}")
        for c in fl.get("top_cells", [])[:4]:
            print(f"  flicker cell col {c['col']} row {c['row']} ({c['x_pct']}%, {c['y_pct']}%): {c['reversal']}")
    for line in result.get("fps", []):
        print("  " + line[:200])
    print(f"  nvlddmkm events during the run: {result.get('driver_resets')}")
    if result.get("warning"):
        print(f"  WARNING: {result['warning']}")
    return 0 if result["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[2:]))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="play a route or script, record it, write the sheets")
    p.add_argument("--route", help="route text (see the route language above)")
    p.add_argument("--route-file", help="a file of route text")
    p.add_argument("--script", help="a raw input script (--replay format), played after the --from prefix")
    p.add_argument("--from", dest="start_from", choices=("autosave", "none"), default="autosave")
    p.add_argument("--seconds", type=int, help="length of the run, display time from the first present (default: the route's end + 5 s)")
    p.add_argument("--label", default="route", help="name for the run's folder")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--crf", type=int, default=18, help="video quality, lower is better and larger (16 wrote 5 MB/s)")
    p.add_argument("--capture", choices=("auto", "window", "game"), default="auto")
    p.add_argument("--shot-step", type=float, default=0.1, help="--capture game: seconds between saved frames")
    p.add_argument("--shots-full", action="store_true", help="--capture game: full-size frames instead of 480x270")
    p.add_argument("--env", action="append", default=[], metavar="OOT_X=V")
    p.add_argument("--burst", default="auto", help="seconds of the video, comma-separated; auto = strongest near-still flicker, middle, end")
    p.add_argument("--burst-frames", type=int, default=9)
    p.add_argument("--crop", help="burst crop: fractions x,y,w,h of the frame")
    p.add_argument("--retry", type=int, default=0)
    p.add_argument("--max-wait", type=float, default=3600, help="seconds to wait for another run to end")
    c = sub.add_parser("compose", help="write a route as an input script")
    c.add_argument("--route", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--from", dest="start_from", choices=("autosave", "none"), default="none")
    s = sub.add_parser("sheets", help="sheets and flicker measure of an existing video, no game run")
    s.add_argument("video")
    s.add_argument("--out", help="folder (default: next to the video)")
    s.add_argument("--burst", default="auto")
    s.add_argument("--burst-frames", type=int, default=9)
    s.add_argument("--crop")
    s.add_argument("--no-flicker", action="store_true")
    a = ap.parse_args()
    if a.cmd == "compose":
        out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
        end = write_script(out, a.route, None, a.start_from)
        print(f"play: {out}, the route ends at tick {end} ({end / TICKS:.1f} s)")
        return 0
    if a.cmd == "sheets":
        result = make_sheets(Path(a.video), Path(a.out) if a.out else None, a.burst, a.burst_frames, a.crop, not a.no_flicker)
        print(json.dumps(result, indent=2))
        return 0
    if bool(a.route) + bool(a.route_file) > 1:
        sys.exit("play: --route or --route-file, not both")
    route = Path(a.route_file).read_text() if a.route_file else a.route
    env = {}
    for kv in a.env:
        k, _, v = kv.partition("=")
        if not k.startswith("OOT_"):
            sys.exit(f"play: --env {kv}: only OOT_* variables reach the game")
        env[k] = v
    return report(play(route, a.script, a.start_from, a.seconds, a.label, a.fps, a.crf, a.capture, a.shot_step, a.shots_full,
                       env, a.burst, a.burst_frames, a.crop, a.retry, a.max_wait))


if __name__ == "__main__":
    sys.exit(main())
