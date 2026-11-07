#!/usr/bin/env python3
# Written for ocarina-of-time-recomp (MIT, Copyright (c) 2026 Jeff Aigner).
"""Make a translator patch (tools/xlate/patches) from edits to the built tree, without copying and diffing by hand.

    python3 tools/xlate_patch.py start            snapshot build/xlate/ryujinx/src (vendor + every patch, as built)
    (edit build/xlate/ryujinx/src/..., build with the edits: see below)
    python3 tools/xlate_patch.py status           the files changed or added since the snapshot
    python3 tools/xlate_patch.py diff             the patch as it would be written
    python3 tools/xlate_patch.py save SLUG        write tools/xlate/patches/<next number>-SLUG.patch

tools/build_xlate.py recreates build/xlate/ryujinx from tools/vendor plus the patches, so edits there are lost on the
next build: `start` right after a build, edit, `save`, then build_xlate.py (which now applies the new patch too).
To try edits before saving, `dotnet publish` in build/xlate/xlate is what build_xlate.py runs after patching.
The snapshot lives in build/xlate/patch-base; `start` refuses to overwrite a snapshot that has unsaved edits against
it unless --force. Paths in the patch are a/src/... and b/src/..., as git apply --directory expects.
"""
import argparse
import difflib
import filecmp
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TREE = ROOT / "build" / "xlate" / "ryujinx"
BASE = ROOT / "build" / "xlate" / "patch-base"
PATCHES = ROOT / "tools" / "xlate" / "patches"
SKIP = re.compile(r"(^|/)(bin|obj)(/|$)")


def files(top: Path) -> dict[str, Path]:
    src = top / "src"
    return {p.relative_to(top).as_posix(): p for p in src.rglob("*")
            if p.is_file() and not SKIP.search(p.relative_to(top).as_posix())}


def changes() -> list[tuple[str, Path | None, Path | None]]:
    if not BASE.exists():
        sys.exit("xlate_patch: no snapshot; run `start` after tools/build_xlate.py")
    old, new = files(BASE), files(TREE)
    out = []
    for rel in sorted(set(old) | set(new)):
        a, b = old.get(rel), new.get(rel)
        if a and b and filecmp.cmp(a, b, shallow=False):
            continue
        out.append((rel, a, b))
    return out


def render() -> str:
    text = []
    for rel, a, b in changes():
        la = a.read_text(errors="surrogateescape").splitlines(keepends=True) if a else []
        lb = b.read_text(errors="surrogateescape").splitlines(keepends=True) if b else []
        text += difflib.unified_diff(la, lb, f"a/{rel}" if a else "/dev/null", f"b/{rel}" if b else "/dev/null")
    return "".join(t if t.endswith("\n") else t + "\n\\ No newline at end of file\n" for t in text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[2:]))
    ap.add_argument("cmd", choices=["start", "status", "diff", "save"])
    ap.add_argument("slug", nargs="?")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "start":
        if BASE.exists() and changes() and not a.force:
            sys.exit("xlate_patch: the tree has edits against the current snapshot; `save` them or use --force")
        shutil.rmtree(BASE, ignore_errors=True)
        shutil.copytree(TREE / "src", BASE / "src", ignore=shutil.ignore_patterns("bin", "obj"))
        print(f"snapshot of {TREE.relative_to(ROOT)}/src in {BASE.relative_to(ROOT)}")
    elif a.cmd == "status":
        for rel, x, y in changes():
            print(("added    " if not x else "removed  " if not y else "modified ") + rel)
    elif a.cmd == "diff":
        sys.stdout.write(render())
    elif a.cmd == "save":
        if not a.slug or not re.fullmatch(r"[a-z0-9-]+", a.slug):
            sys.exit("xlate_patch: save needs a slug of lower-case words joined by dashes")
        text = render()
        if not text:
            sys.exit("xlate_patch: nothing changed since the snapshot")
        nums = [int(p.name[:4]) for p in PATCHES.glob("[0-9][0-9][0-9][0-9]-*.patch")]
        path = PATCHES / f"{max(nums, default=0) + 1:04d}-{a.slug}.patch"
        path.write_text(text, encoding="utf-8", newline="\n")   # LF on Windows too: git apply refuses CRLF hunks
        shutil.rmtree(BASE)
        shutil.copytree(TREE / "src", BASE / "src", ignore=shutil.ignore_patterns("bin", "obj"))
        print(f"wrote {path.relative_to(ROOT)} ({text.count(chr(10))} lines); the snapshot now includes it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
