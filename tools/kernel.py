#!/usr/bin/env python3
# Written for ocarina-of-time-recomp (MIT, Copyright (c) 2026 Jeff Aigner).
"""One shader program from the shader cache's stored inputs: find it, count it, extract, lower, translate, disassemble.

    python3 tools/kernel.py list [--cache DIR] [--tensor]       every stored input: size, HMMA/LDSM/MOVM, code hash
    python3 tools/kernel.py stats PREFIX [--cache DIR]           one program: workgroup, shared memory, top opcodes
    python3 tools/kernel.py extract PREFIX DIR                   ctl and code files (--probe-cs and sm86_lower input)
    python3 tools/kernel.py lower PREFIX [--tensor] [--out DIR]  sm86_lower: Maxwell .low and its .c1
    python3 tools/kernel.py xlate PREFIX [--tensor] [--out DIR]  lower, then oot_xlate: .spv, spirv-val, op histogram
    python3 tools/kernel.py dis PREFIX [sm86_dis.py options]     tools/sm86_dis.py on the code

PREFIX is the start of the input id (the <cache>/in/<id>.in name; OOT_CS_TIME and frame dumps print the same ids) or
a path to an .in file. The code hash is the FNV-1a 64 of the SM86 code that OOT_TENSOR_ONLY matches: the id known
when a program is translated is not always the final input id, so that switch matches the code (2026-11-06).
--cache defaults to build/v131072/shader-cache; build/logs/sc-* (run.py --fresh-cache) are searched too.
The sm86_lower tool is built into build/tools/ with gcc when its sources are newer.
"""
import argparse
import collections
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "build" / "v131072" / "shader-cache"
WINDOWS = os.name == "nt"   # native Windows Python, not WSL: no wslpath, the tools are .exe
TOOL = ROOT / "build" / "tools" / ("sm86_lower.exe" if WINDOWS else "sm86_lower")
XLATE = ROOT / "build" / "xlate" / "oot_xlate.exe"
VKSDK = Path(os.environ.get("VULKAN_SDK_ROOT", "C:/VulkanSDK" if WINDOWS else "/mnt/c/VulkanSDK"))


def cc() -> str:
    """The C compiler for the sm86_lower tool: gcc on PATH, else MSYS2 UCRT64's on Windows."""
    found = shutil.which("gcc")
    if found:
        return found
    msys = Path("C:/msys64/ucrt64/bin/gcc.exe")
    if WINDOWS and msys.exists():
        return str(msys)
    sys.exit("kernel: no gcc (on Windows install MSYS2 UCRT64 gcc, or put gcc on PATH)")
NAMES = {0x23c: "HMMA", 0x83b: "LDSM", 0x23a: "MOVM", 0x389: "SHFL", 0x223: "FFMA", 0x231: "HFMA2", 0x235: "HADD2",
         0x981: "LDG", 0x986: "STG", 0x984: "LDS", 0x388: "STS", 0x918: "NOP", 0x824: "IMAD", 0x810: "IADD3",
         0x812: "LOP3", 0x819: "SHF", 0xb1d: "BAR", 0x94d: "EXIT", 0x947: "BRA", 0xb60: "TEX", 0x290: "UIADD3",
         0x220: "FMUL", 0x221: "FADD", 0x202: "MOV", 0xc0c: "ISETP", 0x20b: "FSEL", 0x326: "?326"}


def wpath(p: Path) -> str:
    if WINDOWS:
        return str(p)
    return subprocess.check_output(["wslpath", "-w", str(p)], text=True).strip()


def find(prefix: str, cache: Path) -> Path:
    p = Path(prefix)
    if p.suffix == ".in" and p.exists():
        return p
    dirs = [cache / "in"] + sorted((ROOT / "build" / "logs").glob("sc-*/in"))
    for d in dirs:
        hits = sorted(d.glob(prefix + "*.in")) if d.exists() else []
        if hits:
            return hits[0]
    sys.exit(f"kernel: no stored input {prefix}* in {', '.join(str(d.relative_to(ROOT)) for d in dirs[:3])} ...")


def load(path: Path) -> dict:
    b = path.read_bytes()
    magic, ver, stage, length, ctl, ans = struct.unpack_from("<6I", b)
    control = b[24 + ans:24 + ans + ctl]
    code = b[24 + ans + ctl:24 + ans + ctl + length]
    h = 0xcbf29ce484222325
    for x in code:
        h = ((h ^ x) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    ops = collections.Counter()
    n = 0
    header = 0 if stage == 5 else 0x80
    end = struct.unpack_from("<I", control, 0x38)[0] if len(control) > 0x3c else len(code)
    for i in range(header, min(end, len(code)) - 15, 16):
        lo, hi = struct.unpack_from("<QQ", code, i)
        if lo == 0 and hi == 0:
            continue
        ops[lo & 0xFFF] += 1
        n += 1
    local = struct.unpack_from("<3I", control, 0x58) if stage == 5 and len(control) >= 0x64 else None
    shared = struct.unpack_from("<I", control, 0x64)[0] if stage == 5 and len(control) >= 0x68 else None
    return {"id": path.stem, "path": path, "stage": {0: "vs", 4: "fs", 5: "cs"}.get(stage, str(stage)),
            "control": control, "code": code, "hash": f"{h:016x}", "insns": n, "ops": ops, "local": local,
            "shared": shared}


def build_tool() -> Path:
    srcs = [ROOT / "tools" / "sm86_lower_main.c", ROOT / "runtime" / "sm86_lower.c", ROOT / "runtime" / "sm86_lower.h"]
    if not TOOL.exists() or any(s.stat().st_mtime > TOOL.stat().st_mtime for s in srcs):
        TOOL.parent.mkdir(parents=True, exist_ok=True)
        gcc = cc()
        env = dict(os.environ)
        if WINDOWS:   # MSYS2's gcc needs its own bin directory on PATH for its helper programs and DLLs
            env["PATH"] = str(Path(gcc).parent) + os.pathsep + env.get("PATH", "")
        subprocess.run([gcc, "-O1", "-w", "-I", str(ROOT / "runtime"), "-DSM86_TOOL", str(srcs[0]), str(srcs[1]),
                        "-o", str(TOOL)], check=True, env=env)
    return TOOL


def extract(k: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "ctl").write_bytes(k["control"])
    (out / "code").write_bytes(k["code"])


def lower(k: dict, out: Path, tensor: bool) -> tuple[Path, Path]:
    extract(k, out)
    low, c1 = out / f"{k['id'][:16]}.low", out / f"{k['id'][:16]}.c1"
    if WINDOWS:   # keep the system variables a Windows program needs; MSYS2's bin for its runtime DLLs
        env = dict(os.environ, SM86_TENSOR="1" if tensor else "0")
        env["PATH"] = str(Path(cc()).parent) + os.pathsep + env.get("PATH", "")
    else:
        env = {"SM86_TENSOR": "1" if tensor else "0", "PATH": "/usr/bin:/bin"}
    r = subprocess.run([str(build_tool()), str(out / "ctl"), str(out / "code"), str(low), str(c1)], env=env,
                       capture_output=True, text=True)
    print((r.stdout + r.stderr).strip())
    if r.returncode or not low.exists():
        sys.exit("kernel: lowering failed")
    print(f"lowered: {low.relative_to(ROOT) if low.is_relative_to(ROOT) else low}: {low.stat().st_size // 8} Maxwell "
          f"words ({'tensor cores' if tensor else 'HMMA emulated'})")
    return low, c1


def spirv_tools() -> tuple[Path | None, Path | None]:
    bins = sorted(VKSDK.glob("*/Bin"))
    if not bins:
        return None, None
    return bins[-1] / "spirv-val.exe", bins[-1] / "spirv-dis.exe"


def xlate(k: dict, out: Path, tensor: bool) -> None:
    if k["stage"] != "cs":
        sys.exit("kernel: xlate handles compute programs (stage cs) only")
    low, c1 = lower(k, out, tensor)
    key = k["id"][:16] + ("-tc" if tensor else "")
    jobs = out / "jobs.json"
    jobs.write_text(json.dumps([{"key": key, "stage": "cs", "code": wpath(low), "c1": c1.read_bytes().hex(),
                                 "debug": False, "local": list(k["local"] or (32, 1, 1)), "shared": k["shared"] or 0}]))
    spv_dir = out / "spv"
    spv_dir.mkdir(exist_ok=True)
    subprocess.run([str(XLATE), wpath(jobs), wpath(spv_dir)], capture_output=True)
    spv = spv_dir / f"{key}.spv"
    if not spv.exists():
        sys.exit(f"kernel: the translator wrote no {spv.name}")
    print(f"spir-v: {spv}: {spv.stat().st_size // 4} words")
    val, dis = spirv_tools()
    if val:
        r = subprocess.run([str(val), "--target-env", "vulkan1.3", wpath(spv)], capture_output=True, text=True)
        print("spirv-val: " + ("valid" if r.returncode == 0 else (r.stdout + r.stderr).strip().splitlines()[0]))
        text = subprocess.run([str(dis), wpath(spv)], capture_output=True, text=True).stdout
        (out / f"{key}.spvasm").write_text(text)
        ops = collections.Counter()
        for line in text.splitlines():
            parts = line.split()
            op = next((p for p in parts if p.startswith("Op")), None)
            if op:
                ops[op] += 1
        print("ops: " + ", ".join(f"{o[2:]} {c}" for o, c in ops.most_common(12)))
        print(f"disassembly: {out / (key + '.spvasm')}")


def describe(k: dict) -> str:
    o = k["ops"]
    return (f"{k['id'][:16]} {k['stage']} {k['insns']:5d} insns  HMMA {o[0x23c]:3d} LDSM {o[0x83b]:3d} MOVM {o[0x23a]:3d}"
            f"  hash {k['hash']}" + (f"  local {k['local'][0]}x{k['local'][1]}x{k['local'][2]}" if k["local"] else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[2:]))
    ap.add_argument("cmd", choices=["list", "stats", "extract", "lower", "xlate", "dis"])
    ap.add_argument("prefix", nargs="?")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--tensor", action="store_true", help="list: tensor-instruction programs only; lower/xlate: "
                                                          "HMMA on cooperative matrices")
    ap.add_argument("--out", type=Path)
    a, rest = ap.parse_known_args()   # rest: the extract directory, or options for sm86_dis.py
    a.rest = rest
    if a.cmd == "list":
        d = a.cache / "in"
        for p in sorted(d.glob("*.in")):
            k = load(p)
            if not a.tensor or k["ops"][0x23c] or k["ops"][0x83b] or k["ops"][0x23a]:
                print(describe(k))
        return 0
    if not a.prefix:
        sys.exit("kernel: PREFIX needed")
    k = load(find(a.prefix, a.cache))
    out = a.out or ROOT / "build" / "logs" / "kern" / k["id"][:16]
    if a.cmd == "stats":
        print(describe(k))
        print(f"input: {k['path']}")
        print(f"code {len(k['code'])} bytes, control {len(k['control'])} bytes" +
              (f", shared {k['shared']} bytes" if k["shared"] is not None else ""))
        print("top: " + ", ".join(f"{NAMES.get(op, hex(op))} {c}" for op, c in k["ops"].most_common(14)))
    elif a.cmd == "extract":
        target = Path(a.rest[0]) if a.rest else out
        extract(k, target)
        print(f"ctl and code in {target}")
    elif a.cmd == "lower":
        lower(k, out, a.tensor)
    elif a.cmd == "xlate":
        xlate(k, out, a.tensor)
    elif a.cmd == "dis":
        extract(k, out)
        return subprocess.run([sys.executable, str(ROOT / "tools" / "sm86_dis.py"), str(out / "code"), "--stage",
                               k["stage"], *a.rest]).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
