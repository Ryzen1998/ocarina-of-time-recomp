# From metroid-prime-4-recomp recomp/module.py @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied unchanged.
# From fe3h-pc-recomp recomp/module.py @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied unchanged.
# From totk-pc-recomp recomp/module.py @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), copied unchanged.
"""One loaded NSO module as the generator and its analyses see it: image at a guest base, .text as words,
function seeds, and bounded-cache instruction lifting.

The function seeds and the .eh_frame_hdr reader are adapted from LAS Recomp tools/analyze.py and tools/gen.py
@ da35b83 (MIT, Copyright (c) 2026 Peiikai).
"""
import struct
from collections import OrderedDict

import numpy as np

from lifter import Insn, Lifter, Unsupported, hx
from nso import Nso, R_AARCH64_RELATIVE

R_AARCH64_JUMP_SLOT = 1026

LIFT_CACHE = 1 << 16  # lifted instructions kept per module; analyses revisit nearby addresses only


def _read_encoded(img, off, enc, pc_base):
    fmt, app = enc & 0x0F, enc & 0x70
    if fmt == 0x0B:  # sdata4
        v, size = struct.unpack_from('<i', img, off)[0], 4
    elif fmt == 0x03:  # udata4
        v, size = struct.unpack_from('<I', img, off)[0], 4
    elif fmt in (0x0C, 0x04):  # sdata8, udata8
        v, size = struct.unpack_from('<q' if fmt == 0x0C else '<Q', img, off)[0], 8
    else:
        raise ValueError(f'unsupported pointer encoding {enc:#x}')
    if app == 0x10:  # pcrel
        v += off
    elif app == 0x30:  # datarel
        v += pc_base
    return v, size


class Module:
    def __init__(self, name, path, base, simd='c'):
        self.name = name
        self.nso = Nso(str(path))
        self.base = base
        n = self.nso
        self.image = bytes(n.image)
        self.lo = base + n.text[0]
        self.hi = self.lo + n.text[1]
        self.words = np.frombuffer(self.image, dtype='<u4', count=n.text[1] // 4, offset=n.text[0])
        self.lifter = Lifter(base, simd)
        self._cache = OrderedDict()

    # ---- memory

    def in_text(self, a):
        return self.lo <= a < self.hi and a % 4 == 0

    def in_image(self, a, n=1):
        return self.base <= a and a + n <= self.base + len(self.image)

    def word(self, a):
        return int(self.words[(a - self.lo) >> 2])

    def read(self, a, n, signed=False):
        """Little-endian integer of n bytes at guest address a (unrelocated image)."""
        o = a - self.base
        return int.from_bytes(self.image[o:o + n], 'little', signed=signed)

    # ---- lifting

    def lift(self, a):
        """Lifted instruction at a; Unsupported becomes an rt_udf trap."""
        ins = self._cache.get(a)
        if ins is not None:
            self._cache.move_to_end(a)
            return ins
        try:
            ins = self.lifter.lift(self.word(a), a)
        except Unsupported as e:
            ins = Insn('trap', f'rt_udf(c, {hx(a)}); /* {e} */')
        self._cache[a] = ins
        if len(self._cache) > LIFT_CACHE:
            self._cache.popitem(last=False)
        return ins

    # ---- function seeds

    def eh_frame_functions(self):
        """Function starts from .eh_frame_hdr's binary search table (absolute addresses)."""
        img, hdr = self.image, self.nso.eh_start
        if not hdr or hdr >= len(img):
            return []
        _, ptr_enc, cnt_enc, tbl_enc = img[hdr:hdr + 4]
        off = hdr + 4
        _, sz = _read_encoded(img, off, ptr_enc, hdr)
        off += sz
        count, sz = _read_encoded(img, off, cnt_enc, hdr)
        off += sz
        starts = []
        for _ in range(count):
            loc, sz = _read_encoded(img, off, tbl_enc, hdr)
            off += sz
            _, sz = _read_encoded(img, off, tbl_enc, hdr)
            off += sz
            starts.append(self.base + loc)
        return starts

    def seeds(self):
        """Known function starts: .eh_frame_hdr, exported FUNC symbols, R_AARCH64_RELATIVE targets in .text, and the
        return address of every call to a setjmp (setjmp_returns)."""
        starts = set(self.eh_frame_functions())
        for _, info, shndx, value, _ in self.nso.syms:
            if shndx and (info & 0xF) == 2:
                starts.add(self.base + value)
        for _, typ, _, add in self.nso.relocs:
            if typ == R_AARCH64_RELATIVE:
                starts.add(self.base + add)
        starts.update(self.setjmp_returns())
        return sorted(a for a in starts if self.in_text(a))

    # ---- setjmp / longjmp (2026-11-10, OOT; runtime/kernel.c rt_setjmp_site, rt_guest_longjmp)

    SETJMP_NAMES = frozenset({'setjmp', '_setjmp', '__setjmp'})
    LONGJMP_NAMES = frozenset({'longjmp', '_longjmp'})

    def _named(self, names):
        """Addresses in .text of this module's definitions of names, and of its PLT stubs bound to imports of them.
        A stub is `adrp x16, P; ldr x17, [x16, #o]; add x16, x16, #o; br x17` and its GOT slot P + o carries a
        JUMP_SLOT relocation of the symbol."""
        out = set()
        for name, info, shndx, value, _ in self.nso.syms:
            if shndx and name in names and (info & 0xF) == 2:
                out.add(self.base + value)
        slots = {self.base + off for off, typ, sym, _ in self.nso.relocs
                 if typ == R_AARCH64_JUMP_SLOT and sym < len(self.nso.syms) and self.nso.syms[sym][0] in names}
        if slots:
            w = self.words
            br = np.flatnonzero(w == 0xD61F0220)   # br x17
            for i in br[br >= 3]:
                adrp, ldr = int(w[i - 3]), int(w[i - 2])
                if (adrp & 0x9F00001F) != 0x90000010 or (ldr & 0xFFC003FF) != 0xF9400211:
                    continue
                pc = self.lo + 4 * (int(i) - 3)
                imm = ((adrp >> 5) & 0x7FFFF) << 2 | ((adrp >> 29) & 3)
                imm -= (1 << 21) if imm & (1 << 20) else 0
                got = (pc & ~0xFFF) + (imm << 12) + ((ldr >> 10) & 0xFFF) * 8
                if got in slots:
                    out.add(pc)
        return out

    def setjmp_targets(self):
        if not hasattr(self, '_sj'):
            self._sj = self._named(self.SETJMP_NAMES)
        return self._sj

    def longjmp_functions(self):
        if not hasattr(self, '_lj'):
            self._lj = self._named(self.LONGJMP_NAMES)
        return self._lj

    def setjmp_returns(self):
        """The address after every BL to a setjmp target: a function start, since a longjmp re-enters there."""
        targets = self.setjmp_targets()
        if not targets:
            return set()
        w = self.words
        bl = np.flatnonzero((w & 0xFC000000) == 0x94000000)
        imm = (w[bl] & 0x03FFFFFF).astype(np.int64)
        imm -= (imm & (1 << 25)) << 1
        tgt = self.lo + 4 * (bl.astype(np.int64) + imm)
        hit = np.isin(tgt, np.array(sorted(targets), dtype=np.int64))
        return {self.lo + 4 * int(i) + 4 for i in bl[hit]}
