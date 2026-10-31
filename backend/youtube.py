"""YouTube downloads for the studio (see pipeline.md). A link is first probed (title, length,
thumbnail; nothing downloaded), so the user can pick the part of the video they want. Only that part is then
downloaded, into storage/_youtube/<fetch_id>/. The UI then creates a run from the fetch straight away, which
moves the file into the run's folder.

Each fetch's state is saved as fetch.json next to its video, so a finished download survives a server restart.

Only youtube.com / youtu.be video links are accepted, and only the 11-character video id is kept from them;
the URL given to yt-dlp is rebuilt from it, so the server never fetches an address the user typed."""

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MIN_HEIGHT = 144
MAX_HEIGHT = 720  # the highest resolution we download
MAX_SECONDS = 60 * 60  # longest part downloaded at once
MIN_SECONDS = 0.5  # shortest part
STALE_SECONDS = 24 * 3600  # a fetch nobody used is deleted after this
FETCH_ID = re.compile(r"^[0-9a-f]{32}$")
# A part is fetched by ffmpeg straight from YouTube. The app's bundled ffmpeg is statically linked and crashes
# (SIGSEGV) when it opens a URL, so parts use a dynamically linked one: YT_FFMPEG_BIN, else the one on PATH.
NET_FFMPEG_BIN = os.environ.get("YT_FFMPEG_BIN") or shutil.which("ffmpeg")
ACTIVE = ("checking", "downloading")

HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"}
VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
PATH_ID = re.compile(r"^/(?:shorts|embed|v)/([A-Za-z0-9_-]{11})(?:/|$)")

# In-memory copies of the fetch records (fetch.json is the saved one). A record is replaced, never changed in
# place, like the studio's runs.
fetches: dict[str, dict] = {}
_lock = threading.Lock()
_root: Path | None = None


class Cancelled(Exception):
    """The user discarded the fetch while it was downloading."""


def configure(storage_dir: Path):
    global _root
    _root = storage_dir / "_youtube"
    _root.mkdir(exist_ok=True)


def video_url(url: str) -> str:
    """The canonical watch URL for a YouTube link. Raises ValueError for anything else."""
    parts = urlparse((url or "").strip())
    if parts.scheme not in ("http", "https") or (parts.hostname or "").lower() not in HOSTS:
        raise ValueError("that isn't a YouTube link")
    host = parts.hostname.lower()
    if host.endswith("youtu.be"):
        video_id = parts.path.strip("/").split("/")[0]
    elif parts.path == "/watch":
        video_id = (parse_qs(parts.query).get("v") or [""])[0]
    else:
        m = PATH_ID.match(parts.path)
        video_id = m.group(1) if m else ""
    if not VIDEO_ID.match(video_id):
        raise ValueError("couldn't find a video in that link")
    return f"https://www.youtube.com/watch?v={video_id}"


def _dir(fetch_id: str) -> Path:
    return _root / fetch_id


def _set(fetch_id: str, /, persist: bool = True, **fields):
    """Update the record and, unless it's only a progress tick, save it to fetch.json."""
    with _lock:
        rec = {**fetches.get(fetch_id, {}), **fields}
        fetches[fetch_id] = rec
        if persist and _dir(fetch_id).is_dir():
            tmp = _dir(fetch_id) / "fetch.json.tmp"
            tmp.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
            tmp.replace(_dir(fetch_id) / "fetch.json")


def _forget(fetch_id: str):
    """Delete the fetch's folder and record."""
    shutil.rmtree(_dir(fetch_id), ignore_errors=True)
    with _lock:
        fetches.pop(fetch_id, None)


def _wipe_files(fetch_id: str):
    """Delete what a failed download left, keeping fetch.json so the error can still be read."""
    for p in _dir(fetch_id).iterdir():
        if p.name != "fetch.json":
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)


def _yt_dlp():
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp isn't installed on the server (pip install yt-dlp)")
    return yt_dlp


def _extract(yt_dlp, link: str, **opts) -> dict:
    """The video's info, without downloading it. RuntimeError for a live stream or one without a length."""
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True, "socket_timeout": 30,
                           **opts}) as ydl:
        info = ydl.extract_info(link, download=False)
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise RuntimeError("live streams can't be downloaded")
    if not info.get("duration"):
        raise RuntimeError("couldn't read the video's length")
    return info


def probe(url: str) -> dict:
    """Title, length and thumbnail of a link, so the user can pick the part to download. ValueError for a bad
    link or a video that can't be downloaded; RuntimeError if yt-dlp is missing."""
    link = video_url(url)
    yt_dlp = _yt_dlp()
    try:
        info = _extract(yt_dlp, link)
    except (RuntimeError, yt_dlp.utils.DownloadError) as exc:
        raise ValueError(_clean(exc))
    return {"url": link, "title": info.get("title") or "YouTube video", "duration": info["duration"],
            "thumbnail": info.get("thumbnail"), "uploader": info.get("uploader"), "max_seconds": MAX_SECONDS}


def start(url: str, height: int, ffmpeg_bin: str, start_s: float | None = None,
          end_s: float | None = None) -> dict:
    """Check the link, then fetch the video, or only start_s..end_s of it (seconds; a missing end means to the
    end), in the background. Returns the record to poll. ValueError for a bad link, resolution or part."""
    link = video_url(url)
    if not MIN_HEIGHT <= height <= MAX_HEIGHT:
        raise ValueError(f"resolution must be between {MIN_HEIGHT}p and {MAX_HEIGHT}p")
    start_s = max(start_s or 0.0, 0.0)
    if end_s is not None:
        if end_s - start_s < MIN_SECONDS:
            raise ValueError("the end must be at least 0.5 s after the start")
        if end_s - start_s > MAX_SECONDS:
            raise ValueError(f"at most {MAX_SECONDS // 60} min can be downloaded at once")
    _delete_stale()

    fetch_id = uuid.uuid4().hex
    _dir(fetch_id).mkdir()
    _set(fetch_id, fetch_id=fetch_id, url=link, height=height, status="checking", title=None, duration=None,
         section=None, percent=0, error=None, created=time.time())
    threading.Thread(target=_fetch, args=(fetch_id, link, height, ffmpeg_bin, start_s, end_s), daemon=True).start()
    return fetches[fetch_id]


def _fetch(fetch_id: str, link: str, height: int, ffmpeg_bin: str, start_s: float, end_s: float | None):
    try:
        yt_dlp = _yt_dlp()
        fmt = f"bv*[height<={height}]+ba/b[height<={height}]"
        info = _extract(yt_dlp, link, format=fmt)
        duration = info["duration"]
        stop = duration if end_s is None else min(end_s, duration)
        if stop - start_s < MIN_SECONDS:
            raise RuntimeError(f"the video is only {duration} s long; pick a part inside it")
        section = [round(start_s, 3), round(stop, 3)] if start_s > 0 or stop < duration else None
        length = stop - start_s
        if length > MAX_SECONDS:
            raise RuntimeError(f"that's {length / 60:.0f} min; at most {MAX_SECONDS // 60} min can be downloaded"
                               " at once, so pick a shorter part")

        _set(fetch_id, status="downloading", title=info.get("title") or "YouTube video", duration=duration,
             section=section, height_got=info.get("height"))
        if section:
            _download_part(fetch_id, info, section)
            _set(fetch_id, status="ready", percent=100, filename="video.mp4")
            return

        # video and audio come as separate downloads
        streams = {"n": len(info.get("requested_formats") or [1]), "done": 0}

        def hook(d):
            if fetches.get(fetch_id, {}).get("status") == "discarded":
                raise Cancelled()
            # each stream is counted from 0 to 100
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            frac = 0.0
            if d["status"] == "finished":
                streams["done"] += 1
            elif total:
                frac = (d.get("downloaded_bytes") or 0) / total
            _set(fetch_id, persist=False, percent=round(100 * min((streams["done"] + frac) / streams["n"], 1.0), 1))

        opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": fmt,
            "merge_output_format": "mp4",
            "outtmpl": str(_dir(fetch_id) / "video.%(ext)s"),
            "ffmpeg_location": ffmpeg_bin,
            "progress_hooks": [hook],
            "socket_timeout": 30,
            "retries": 3,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([link])

        files = [p for p in _dir(fetch_id).glob("video.*") if p.suffix == ".mp4"]
        if not files:
            raise RuntimeError("the download produced no video file")
        _set(fetch_id, status="ready", percent=100, filename=files[0].name)
    except Cancelled:
        _forget(fetch_id)
    except Exception as exc:
        if _dir(fetch_id).is_dir():
            _wipe_files(fetch_id)
        _set(fetch_id, status="error", error=_clean(exc))


def _download_part(fetch_id: str, info: dict, section: list[float]):
    """Fetch only `section` of the video into video.mp4: ffmpeg seeks into each stream (range requests), so only
    that part is downloaded, and re-encodes it for an exact cut. We run ffmpeg ourselves rather than through
    yt-dlp's download_ranges, which waits for it without any progress, so that its -progress output (how far
    into the part it is) can drive `percent`."""
    if not NET_FFMPEG_BIN:
        raise RuntimeError("downloading a part needs a system ffmpeg (apt install ffmpeg, or set"
                           " YT_FFMPEG_BIN); the bundled one can't read from the web")
    start_s, stop = section
    length = stop - start_s
    formats = info.get("requested_formats") or [info]  # video + audio, or one stream with both
    args = [NET_FFMPEG_BIN, "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-progress", "pipe:1"]
    for fmt in formats:
        if not re.match(r"https?://", fmt.get("url") or ""):
            raise RuntimeError("this video's streams can't be cut while downloading; pick the whole video")
        headers = fmt.get("http_headers") or info.get("http_headers")
        if headers:
            args += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
        args += ["-rw_timeout", "30000000", "-ss", str(start_s), "-t", str(length), "-i", fmt["url"]]
    if len(formats) > 1:
        for i in range(len(formats)):
            args += ["-map", f"{i}:0"]
    part = _dir(fetch_id) / "video.mp4.part"
    args += ["-f", "mp4", str(part)]

    with subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True) as proc:
        try:
            # key=value lines, a block about every 0.5 s; out_time_us is how much of the part is written
            for line in proc.stdout:
                if fetches.get(fetch_id, {}).get("status") == "discarded":
                    raise Cancelled()
                key, _, value = line.strip().partition("=")
                if key == "out_time_us" and value.isdigit():
                    _set(fetch_id, persist=False, percent=round(100 * min(int(value) / 1e6 / length, 1.0), 1))
        except BaseException:
            proc.kill()
            raise
        err = proc.stderr.read().strip()  # small: only errors are logged
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg couldn't download the part: {err.splitlines()[-1] if err else f'exit {code}'}")
    part.replace(_dir(fetch_id) / "video.mp4")


def _clean(exc: Exception) -> str:
    """yt-dlp errors start with 'ERROR: ' and may carry ANSI colour codes."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(exc)).strip()
    return text[len("ERROR: "):] if text.startswith("ERROR: ") else text


def get(fetch_id: str) -> dict | None:
    """A fetch's record: from memory, else loaded from its fetch.json (after a server restart). One that was
    still downloading when the server stopped comes back as an error."""
    if not FETCH_ID.match(fetch_id):
        return None
    if fetch_id in fetches:
        return fetches[fetch_id]
    path = _dir(fetch_id) / "fetch.json"
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text())
    except ValueError:
        return None
    with _lock:
        fetches[fetch_id] = rec
    if rec.get("status") == "discarded":
        _forget(fetch_id)
        return None
    if rec.get("status") in ACTIVE:
        _wipe_files(fetch_id)
        _set(fetch_id, status="error", error="the download was interrupted by a server restart; start it again")
    return fetches[fetch_id]


def video_path(fetch_id: str) -> Path | None:
    rec = get(fetch_id)
    if not rec or rec.get("status") != "ready":
        return None
    path = _dir(fetch_id) / rec["filename"]
    return path if path.exists() else None


def take(fetch_id: str, dest: Path) -> dict:
    """Move a finished fetch's video to `dest` (a path with the .mp4 suffix) and forget the fetch. Returns its
    record. KeyError if there's no such fetch, ValueError if it isn't finished."""
    rec = get(fetch_id)
    if rec is None:
        raise KeyError(fetch_id)
    path = video_path(fetch_id)
    if path is None:
        raise ValueError("the video isn't downloaded (yet)")
    shutil.move(str(path), str(dest))
    _forget(fetch_id)
    return rec


def discard(fetch_id: str) -> bool:
    """Drop a fetch the user doesn't want. One still downloading is stopped by its progress hook."""
    rec = get(fetch_id)
    if rec is None:
        return False
    if rec["status"] in ACTIVE:
        _set(fetch_id, status="discarded")
    else:
        _forget(fetch_id)
    return True


def _delete_stale():
    """Delete fetches nobody used for STALE_SECONDS (by their folder's age), except ones still downloading."""
    cutoff = time.time() - STALE_SECONDS
    for d in _root.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff and fetches.get(d.name, {}).get("status") not in ACTIVE:
            _forget(d.name)
