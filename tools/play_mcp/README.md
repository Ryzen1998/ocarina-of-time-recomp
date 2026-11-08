# play_mcp

An MCP server that lets a Claude session play the game from an input route and look at the result. It is a thin wrapper
around [`tools/play.py`](../play.py), which does all the work and can be used on its own without any setup.

## Tools

| Tool | What it does |
| --- | --- |
| `play_route` | Plays a route or script through `tools/run.py`, records the game window, returns the contact sheet, a burst sheet, paths, fps lines and the driver-reset count. Blocks for the length of the run (1-4 minutes), and longer if another run is in progress. |
| `compose_input` | Writes a route as an input script in the `--replay` format. |
| `make_sheets` | Makes the contact sheet, burst sheets and flicker measure of a video that already exists. |
| `get_image` | Returns a PNG written by these tools, such as a burst sheet or the flicker heat map. |

The route language is described in the header of `tools/play.py` and in each tool's description. Output goes to
`build/reference/play/<run>/` (git-ignored).

## Registering it

A new MCP server loads when a Claude session starts, so this takes effect in sessions started afterwards. Nothing here
changes your settings until you run the command.

From the repository root, in a terminal:

```bash
claude mcp add --scope local oot-play -- "C:\Games\OoT Recomp\ocarina-of-time-recomp\ocarina-of-time-recomp-main\.venv-win\Scripts\python.exe" "C:\Games\OoT Recomp\ocarina-of-time-recomp\ocarina-of-time-recomp-main\tools\play_mcp\server.py"
```

`--scope local` keeps the registration for this project on this machine and out of the repository. To remove it:

```bash
claude mcp remove oot-play
```

Start a new session after registering it. A session that has not loaded the server can use `tools/play.py` directly and
get the same result.

`play_route` blocks while the game runs, so a client that limits tool calls to a short time needs a longer limit. In
Claude Code that is the `MCP_TOOL_TIMEOUT` environment variable, in milliseconds.

## The rules it keeps

It starts the game only through `tools/run.py`, waits for any other run to end, and never kills a game it did not start.
It refuses to run after a driver reset since the last boot, as the PC has to be rebooted first, and reports the number of
nvlddmkm events after every run. It captures only the window titled `Ocarina of Time Recomp`, never the desktop.
