#!/usr/bin/env python3
# Written for ocarina-of-time-recomp (MIT, Copyright (c) 2026 Jeff Aigner).
"""A small MCP server (stdio, JSON-RPC 2.0) around tools/play.py: play a route, compose input, make sheets.

Standard library plus what play.py needs; it speaks the protocol by hand because the `mcp` package is not installed in
the project's venv. See README.md for how to register it. All the work is in tools/play.py, which is also usable alone.
"""
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROTOCOL = "2025-06-18"
# Protocol messages go to the real stdout; everything else printed in this process goes to stderr.
protocol_out = sys.stdout
sys.stdout = sys.stderr

import play  # noqa: E402  tools/play.py

ROUTE_HELP = ("Route language: steps separated by ';' or newlines, '&' runs steps together. Durations are 60 Hz ticks or "
              "seconds ('1.5s'). wait D; at T (absolute since the first present); press BUTTON [D] (A B X Y L R ZL ZR "
              "PLUS MINUS LCLICK RCLICK UP DOWN LEFT RIGHT); walk DIR D [PCT] (left stick; up down left right upleft "
              "upright downleft downright or x,y); camera DIR D [PCT] (right stick); raw LINE. With from=autosave the "
              "route starts after tick 3060 unless it opens with 'at'; open with 'at 62s' or later for the player to "
              "be in control, and check the contact sheet.")

TOOLS = [
    {"name": "play_route",
     "description": "Play the game from an input route and record the game window. Blocks until the run ends (a run takes "
                    "1-4 minutes and waits for any other run to end first). Returns the contact sheet and a burst sheet as "
                    "images, the output paths, fps lines and the driver-reset count. " + ROUTE_HELP,
     "inputSchema": {"type": "object", "properties": {
         "route": {"type": "string", "description": "the route text"},
         "script": {"type": "string", "description": "path of a raw --replay script, instead of or before the route"},
         "from": {"type": "string", "enum": ["autosave", "none"], "default": "autosave"},
         "seconds": {"type": "integer", "description": "length of the run (default: the route's end + 5 s)"},
         "label": {"type": "string", "default": "route"},
         "bursts": {"type": "string", "default": "auto", "description": "comma-separated seconds of the video, or auto"},
         "crop": {"type": "string", "description": "burst crop: fractions x,y,w,h of the frame"},
         "capture": {"type": "string", "enum": ["auto", "window", "game"], "default": "auto"},
         "env": {"type": "object", "additionalProperties": {"type": "string"}, "description": "OOT_* variables for the game"},
     }}},
    {"name": "compose_input",
     "description": "Write a route as an input script in the --replay format, without running the game. " + ROUTE_HELP,
     "inputSchema": {"type": "object", "required": ["route", "out"], "properties": {
         "route": {"type": "string"},
         "out": {"type": "string", "description": "path, relative to the repo or absolute"},
         "from": {"type": "string", "enum": ["autosave", "none"], "default": "none"},
     }}},
    {"name": "make_sheets",
     "description": "Make the contact sheet, burst sheets and flicker measure of an existing video, without running the game. "
                    "Returns the contact sheet and the first burst as images.",
     "inputSchema": {"type": "object", "required": ["video"], "properties": {
         "video": {"type": "string"},
         "bursts": {"type": "string", "default": "auto"},
         "burst_frames": {"type": "integer", "default": 9},
         "crop": {"type": "string", "description": "burst crop: fractions x,y,w,h of the frame"},
     }}},
    {"name": "get_image",
     "description": "Return an image written by this tool, by path, such as a burst sheet or flicker-heatmap.png.",
     "inputSchema": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}},
]


def image_content(path: str, max_width: int = 1920) -> dict:
    """A PNG as an image block, shrunk and in JPEG so that the message stays small."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return {"type": "image", "data": base64.b64encode(buf.getvalue()).decode(), "mimeType": "image/jpeg"}


def text_content(value) -> dict:
    return {"type": "text", "text": value if isinstance(value, str) else json.dumps(value, indent=2)}


def within_output(path: str) -> Path:
    """get_image reads only PNG files under build/ of the repo."""
    p = Path(path)
    p = p if p.is_absolute() else play.ROOT / p
    p = p.resolve()
    if p.suffix.lower() != ".png" or not (p.is_relative_to(play.OUT) or p.is_relative_to(play.ROOT / "build")):
        raise ValueError("get_image: only PNG files under build/ of the repo")
    return p


def sheets_content(result: dict) -> list:
    blocks = [text_content(result)]
    if result.get("contact"):
        blocks.append(image_content(result["contact"]))
    if result.get("bursts"):
        blocks.append(image_content(result["bursts"][0]))
    return blocks


def call_tool(name: str, a: dict) -> list:
    if name == "play_route":
        if not a.get("route") and not a.get("script"):
            raise ValueError("play_route needs a route or a script")
        result = play.play(a.get("route"), a.get("script"), a.get("from", "autosave"), a.get("seconds"), a.get("label", "route"),
                           capture=a.get("capture", "auto"), env=a.get("env") or {}, bursts=a.get("bursts", "auto"), crop=a.get("crop"))
        return sheets_content(result)
    if name == "compose_input":
        out = Path(a["out"]) if Path(a["out"]).is_absolute() else play.ROOT / a["out"]
        end = play.write_script(out, a["route"], None, a.get("from", "none"))
        return [text_content({"script": str(out), "end_tick": end, "end_seconds": round(end / play.TICKS, 2)})]
    if name == "make_sheets":
        return sheets_content(play.make_sheets(Path(a["video"]), None, a.get("bursts", "auto"), int(a.get("burst_frames", 9)), a.get("crop")))
    if name == "get_image":
        return [image_content(str(within_output(a["path"])))]
    raise ValueError(f"unknown tool {name}")


def send(message: dict) -> None:
    protocol_out.write(json.dumps(message) + "\n")
    protocol_out.flush()


def handle(req: dict) -> None:
    method, rid = req.get("method"), req.get("id")
    if rid is None:
        return   # a notification (notifications/initialized, cancelled): nothing to answer
    if method == "initialize":
        version = req.get("params", {}).get("protocolVersion", PROTOCOL)
        send({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": version, "capabilities": {"tools": {}},
                                                      "serverInfo": {"name": "oot-play", "version": "1.0"}}})
    elif method == "ping":
        send({"jsonrpc": "2.0", "id": rid, "result": {}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        params = req.get("params", {})
        try:
            content = call_tool(params.get("name"), params.get("arguments") or {})
            send({"jsonrpc": "2.0", "id": rid, "result": {"content": content, "isError": False}})
        except (Exception, SystemExit) as e:   # play.py exits with a message when the machine is not set up
            send({"jsonrpc": "2.0", "id": rid, "result": {"content": [text_content(f"error: {e}")], "isError": True}})
    else:
        send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}})


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            handle(json.loads(line))
        except json.JSONDecodeError:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})


if __name__ == "__main__":
    main()
