# Phase 3: SM86 shaders

Hand-off written 2026-11-05 at commit `b46d663`, for a fresh session. Read `CLAUDE.md` and `AGENTS.md` first; the
history is in `docs/plan.md` (Phases 0 to 2 and the NVN2 survey). This file says where the port stands, exactly what
blocks it, what is already measured, and the order to do the work in.

## Current state (2026-11-08, end of day): read this first

The dated Status sections below are the lab notebook, oldest first; this one is the summary. Branch `visuals` of the
fork (Ryzen1998), last code commit `3f4d5c2`. KEY_VERSION 23; translator identity c0a08c42... (patches 0019 global
window, 0020 RED signed offset; after `tools/build_xlate.py` copy `oot_xlate.exe` + `.id` into `build/v131072/`).

### What works

- Boot, title, menus and the opening; the replay (`build/logs/a-spam.txt`, presses A) reaches Kokiri Forest / the
  Deku Tree area about 75 s in (frame ~4500).
- **The mesh pipeline is on by default** (since 2026-11-08 evening): the global window (`OOT_GW`), the task/mesh
  stages (`OOT_MESH`) and the indirect mesh draws (`OOT_MESH_IND`); `=0` turns each off. With them the frame's GPU
  culling counts, the culling kernels' indirect dispatches get real group counts, and the indirect mesh draws run:
  the forest ground, the stone path, rocks, ferns, fences, Link and Navi are drawn lit and textured where their
  programs lower (`build/logs/gwskip-ground.png`, `k23-final.png`, `defaults-on-1.png`, `defaults-on-2.png`).
- Level speed: ~50 fps with the mesh pipeline, ~59 fps without (`OOT_GW=0 OOT_MESH=0`).
- Stability: about 15 runs since the last reboot (run-146..162), from the title and from the level, no driver reset
  and no game crash. All six driver resets of 2026-11-08 were before `7ed675d`.

### What is still wrong

1. **Large white areas: fixed 2026-11-09 (workaround)**, see the Status section of that date: the game skipped the
   colour draws of objects whose occlusion test counted 0 samples; samples-passed reports now say visible by default
   (`OOT_OQ_VISIBLE=0` for the real counts). The cause of the zero counts is still open. Original notes: each frame first fills its HDR target
   with the sky (one full-screen draw, values 24..34), then draws the scene over it; wherever no geometry is drawn
   the lit sky shows and tone-maps to pure white. So the white is **missing geometry**, not a colour bug. It shows in
   default runs too (earlier "clean" default screenshots were taken in dark spots). Known sources:
   - 7 indirect-only mesh programs still do not lower (their draws are skipped), see next steps;
   - mesh program 072eb70d (code hash a11f465b) lowers but its draws turn the frame into black and white blocks, so
     it is skipped by default (`OOT_MESH_IND_SKIP` defaults to `a11f465b`);
   - possibly other draws not yet identified (no frame dump with everything on has been read yet).
2. **Ground colour**: with the indirect draws the ground looks over-bright, yellow and speckled in places; not
   studied.
3. **Fog** kernel 1a62084a: ~60% NaN volume, skipped (`OOT_SKIP_CS=1a62084a` in every test run).
4. **Speed**: the mesh pipeline costs ~9 fps (window selection on the CPU, extra task/mesh dispatches).
5. **Untested**: controller play, other areas, scene loads, combat, saving, the movies (skipped in tests).

### What was found and fixed (2026-11-07 / 11-08, in order)

| Commit | Bug | Effect / fix |
|---|---|---|
| `4669605` | The lowering's record-less jump-table fix-up rewrote constant bank 1 in programs without a BRX (the culling's 64-bit offset 12 became 8 << 32 + 12) | culling counters were written 32 GiB away; fix-up only with a BRX, plain c1 words as immediates (KEY 19) |
| `0804b09` | Accesses through pointers loaded at run time had no storage buffer (49 in culling kernel e6d5432f) | global window: translator patch 0019 + `gw_select` (24 bank-0 entries at 0x410) |
| `c8441d1` | PLOP3 was lowered to PSETP with its first and third operands swapped (asymmetric tables wrong; 231 of 902 programs change) | every culling thread exited before its atomics (KEY 20) |
| `c8441d1` | Ryujinx's `Red` reads RED's 20-bit offset unsigned (`Atom` sign-extends) | `RED [p - 12]` went 1 MiB off; translator patch 0020 |
| `c8441d1` | Indirect mesh draws were never run | `op_draw_mesh` runs them: a bound of task workgroups, the count copied on the GPU to bank 0 + 0x628, `task_prologue` (KEY 21) |
| `00c4d56` | Task records were 0x100 bytes; the indirect draws' task programs write payload to ~0x898 | records of 0x1000 (KEY 22) |
| `7ed675d` | The global window was never filled for task/mesh stage dispatches | task program c7f0444a read 0 and looped 0x07ffffff times: the device losses of run-126..135; `gw_select_mesh` |
| `0c2f1a0` | Storage views ignored the game's R32 view format (the AA kernel stores packed R11G11B10 through R32UI) | R32 views, STORAGE by extension (`storage_r32`), the live image over the same bytes |
| `932a551` | Kernels 374126bf / 358beb27 (wrong traces: stores to float words of bank 3) got window entries over the culling results | game crash at main+0xc938d0 in 3 of 12 window runs, none in 10+ since; `OOT_GW_SKIP` |
| `b0d964b` | Uniform registers: forwarding stopped at every block start, a register's span was every touch of its number, dead definitions still wrote; UP2R (0x883) not lowered | 5 of the 12 indirect-only task/mesh programs now lower (task c6f3721f = hash bbcf8a53, meshes f5d63a4c, b55ba187, d12f99f4, and 072eb70d which is skipped); KEY 23 |
| `3f4d5c2` | — | `OOT_GW`, `OOT_MESH`, `OOT_MESH_IND` on by default |

Uniform register changes in detail (`sm86_lower.c`): `ur_flow` is a forward dataflow of the constant each uniform
register holds; a block leader starts the local forwarding from it, so a register loaded with the same constant on
every path forwards there too. `ur_liveness` is a backward dataflow over the reads that do not forward; a register
that needs one gets its span from where it is live, and a definition whose value is dead writes no register
(`ur_has_reg` per instruction). UP2R (`case 0x083`): URd = (URa & ~mask) | packed uniform predicates, one OR per mask
bit under the predicate standing for that UP. 421 of 942 cached programs lower differently (mostly dropped dead
writes); runs show no regression. `SM86_UR=1` (tool build) lists the reads that do not forward.

How the bugs were found, for next time: the culling chain was traced with `OOT_IND_LOG` / `OOT_MESH_WATCH` /
`OOT_GW_TRACE`; PLOP3 and the RED offset with `OOT_SM86_COUNT` (threads per instruction, registers summed); the
device losses by bisecting the indirect draws (`OOT_MESH_IND_ONLY`), then `OOT_SM86_COUNT`'s dump at the loss showed
the loop and its zero inputs; the white with frame dumps at a fixed frame (`run.py --dump 7000`); the game crash with
`OOT_WATCH_RANGE`.

### Diagnostics added (keep using them)

- `OOT_SM86_COUNT=<code hash>:<off>[=rN|=pN],...`: per instruction, how many threads arrive (or the sum of a register,
  or of a predicate bit), for any compute/task/mesh program; logged every 120 frames and **at a device loss** (the
  counters live in host memory). It found both the PLOP3 bug and the hang. The code hash is `kernel.py`'s "hash".
- `OOT_IND_LOG=<frame>` (indirect dispatch group counts + counter word), `OOT_MESH_LOG=1` + `OOT_MESH_LOG_FROM`
  (indirect mesh draw arguments), `OOT_MESH_WATCH[=<frame>]` (who writes the arguments; a frame's dispatch list),
  `OOT_GW_TRACE=<id prefix>` (a kernel's pointer chains), `OOT_WATCH_RANGE=<lo>:<hi>[,...]` (every GPU write that may
  land in a range), `meshind:` lines (each new task/mesh pair of an indirect draw, logged before it is recorded).
- Indirect draw knobs: `OOT_MESH_IND_MAX` (bound, default 64), `OOT_MESH_IND_FROM=<frame>`, `OOT_MESH_IND_ONLY` /
  `OOT_MESH_IND_SKIP=<hash prefixes>`. Window: `OOT_GW_SKIP=<id prefixes>` (default `374126bf,358beb27`).
- Frame dumps (`run.py --dump <frame>`, `OOT_DUMP_SKIP_RUNS=1`): `images.txt` has per-target min/max/NaN; float
  targets are saved clipped (`_tm.png` is tone-mapped). Dispatches are named by a hash of their **SPIR-V**, not the
  input id: hash the cache's `.spv` files to find one.

### What to do next, in order

1. **Find the remaining missing geometry.** Narrowed on 2026-11-09 (Status section below): the depth-pass mesh
   draws write spiky terrain the colour pass does not colour; start there. Originally: a frame dump at a fixed frame (7000) with the defaults: list the draws
   of the main colour pass, find what is missing where the sky shows (skipped draws in the log: `translation
   failed`, `meshind:` pairs, `OOT_MESH_IND_SKIP`), and which of them matter most.
2. **Mesh program 072eb70d (code hash a11f465b)**: lowers since UP2R but draws garbage. Run it alone
   (`OOT_MESH_IND_ONLY=a11f465b OOT_MESH_IND_SKIP=none`) with a frame dump, read its mesh records (`OOT_MESH_LOG`),
   compare its UP2R / uniform-predicate swap path with task program c6f3721f, which works.
3. **The 7 mesh programs that still do not lower**: four "no ... register dead over a uniform register's span"
   (6d98affa x3 at +0x320, 9a4accb7 at +0x360: a computed uniform value live across code that uses R0..R251; needs
   spilling uniform registers to local memory, or recomputing them at each use), three "no general register free for
   scratch" (52ed87ef +0x6c90, fa159bf2 +0x3960, 47ebe041 +0x22a0: the same, for the expansion scratch). Failing
   inputs are in the cache as `.fail.ctl` / `.fail.code`; lower them with `build/tools/sm86_lower.exe --batch`
   (paths without spaces), `SM86_POOL=1 SM86_LIVE=1 SM86_UR=1` for details.
4. **The over-bright, speckled ground** (indirect draws): compare its pass-through vertex outputs with what the
   fragment shader expects (`mesh_vs`), and the depth/blend state of the stand-in draw.
5. **Speed** (~50 vs ~59 fps): profile `gw_select` (`OOT_TRACE=perf`) and the task/mesh dispatches (`OOT_CS_TIME`).
6. **Correctness left**: the fog kernel's NaN; fix the wrong traces of 374126bf / 358beb27 rather than skip them;
   counts above `OOT_MESH_IND_MAX` (64; seen up to 29) would lose meshes silently.
7. **A controller play session** with the user, to see what blocks real play (movement, other areas, saving).

### Status (2026-11-09): what the white is not, and the mesh-draw bound that was 1

Item 1 above, worked through with frame dumps (run-163..183, all with zero driver resets). Not fixed; narrowed down.

- **Dumps at 7000 never fire**: the level runs at ~40 fps here (with another GPU program open), so a 150 s replay
  presents ~5,000-5,800 frames. Dump at 4900. The replay is not deterministic: frame 4900 lands on different shots of
  the opening cutscene (forest/treehouse, forest path, Saria close-up, Navi, Mido), so **never compare two runs'
  pictures**; compare within one frame (`OOT_DEPTH_EQ_LE=3`, `OOT_DUMP_DRAWS=eq`, below).
- How the level draws: a depth pre-pass (cull back, depth write), then the game copies the depth image
  (`CopyTextureToTexture`, a bit-exact `vkCmdCopyImage`) and the colour pass draws over the copy with depth test
  **EQUAL, no writes**. Every colour-pass draw has a depth-pass draw with the same triangles (same index buffer, count,
  base vertex). The white pixels are ones where the depth image has geometry and no colour-pass draw writes.
- **Ruled out**, each checked: the shaders' position math (the depth-only and colour vertex programs compute
  `gl_Position` with identical SPIR-V expressions, skinning and dynamic bone indices included, all `NoContraction`,
  position `Invariant`); fragment discards (no colour-pass fragment program has `OpKill`/demote); a written
  `FragDepth`; depth bias (none in either pass, now printed by the dump); culling (both passes cull back); the depth
  copy; GPL (0 library links: all pipelines monolithic); the clip-space depth mode; the uniform snapshot (one frame is
  one submission, so both passes read the same snapshot; per-draw transforms c3/c4 and all common uniform blocks of
  paired draws are bit-identical in dump-179); global-window writes into uniform ranges (0 overlaps measured). An
  apparent "less-or-equal fixes it" was two different cutscene shots; on one frame (`OOT_DEPTH_EQ_LE=3`) left and right
  halves are equally white.
- **What the dumps do show** (dump-179/180/182, forest path): after the ordinary depth pass the depth image is right
  (fence, the rock Link stands on); after the indirect, mesh-only **depth-pass mesh draws** (programs 475e71d2,
  13c12eb4, 12b6f03f, 31b627ac, 4eea96ed, b4c61e5e, 0704b9fe, 3df0445d, 6cb6f80d) a jagged, spiky terrain covers them.
  The colour pass uses other mesh programs for the same meshes (3dfea3de, 23eaf836, 2f4435de, c5bdb428, d022e166,
  69066810); where their colour is missing the EQUAL test fails and the sky shows. In the Saria shot the colour pass
  also draws her bracelets and necklace in another place than the depth image has them. Checked and fine: the vertex
  layout the pass-through assumes (`(v >> 5) * lines * 0x80 + (v & 31) * 4`) matches 475e71d2's own addressing (8
  lines, 96 vertices, 64 primitives, header output map 8 bits).
- **Bug fixed**: `OOT_MESH_IND_MAX` was read with `rt_knob`, which returns 0/1 for switches, so every indirect mesh draw
  dispatched **one** task group (16 mesh groups) or one mesh group instead of the documented 64: most meshlets of the
  culled scenery were never drawn. Now parsed as a number (default 64, logged). Runs at 8 and 64: no driver reset,
  pictures score (noise 0.34 / 0.37); more mesh geometry is drawn (the vines around the tree trunks appear), the white
  remains.
- Next, in order: (1) find which depth-pass mesh program draws the spikes: dump without `OOT_DUMP_SKIP_RUNS` gives a
  depth image after each mesh pass, the `mesh draw` notes name the programs; then compare its records (positions) with
  the colour variant of the same meshlet (`OOT_MESH_LOG`, a record read-back) and its lowering; (2) the 7 mesh programs
  that do not lower (the log fails them on every draw: 6194772d, 4478d9b1, 392f0c37, 2b82fd56, 19f6e020; 3406d516 is
  "no general register free for the uniform predicates"); (3) the Saria pose difference (skinned depth vs colour).

**Later the same day: every program lowers.** The four colour-pass mesh programs the log failed on every draw
(52ed87ef, 47ebe041, fa159bf2, 9a4accb7; they draw big scenery such as the Deku Tree's trunk, which the depth pass draws
with ordinary draws) and 6d98affa and the compute program 3406d516 failed for want of general registers: around their
instructions 1600-1750 ~245 of 255 registers are live, and a uniform register read over most of the program needs one
register dead over all of it. `sm86_lower.c` now retries such a failure (`sm86_lower_ex`, `relaxed`): a uniform
register's home is held only over its span (`span_hold`), a uniform register with none dead over its span lives in a
local-memory word (`spill`, `spill_temp`: loaded into scratch 10/11/8/9 before each instruction naming it, stored back
after, unpredicated), the uniform predicates' register likewise (`up_spill`), and an instruction without 12 free
registers borrows 12 it does not name, saving the live ones before it and reloading them after (`scratch_at`,
`scratch_restore`; never around a branch). Local memory: compute and mesh programs only (the translator gives them
4 KiB), above the program's own LDL/STL. Survey of all 1,283 cached inputs, old against new lowering: 1,260 byte-identical,
23 newly lower, 0 newly fail; the 23 translate to SPIR-V that `spirv-val` accepts. Lowering failures are not cached
(the `.fail.*` files are only dumps), so no `KEY_VERSION` bump: the new programs translate on first use and nothing
else re-translates. Runs: zero driver resets, 0 "translation failed" lines (was ~7,000 per program per run), the level
at ~48 fps (was ~40; the failed lowering was retried on every draw). The white is **not** gone: the new mesh draws run
but, on the frames dumped, add no colour pixels; in the Saria close-up (found with the new `OOT_DUMP_WHEN=white:<n>@<present>`,
which arms the dump on the first frame with that many pure-white pixels, needs `OOT_SHOT_RING=2`) her body draws
(vertex programs 6ffe3c6b, 60aeb179) add almost nothing in the colour pass although their depth-pass partners
(de5a862a, bba76300) draw the same triangles from the same vertex buffer with the same uniforms. Their position code has
branches (value selections on an index == 0 / == 8) that the scratch expression evaluator does not model, so "same
maths" is not proven for skinned meshes. Next: run both programs on the dumped vertex data (`OOT_DUMP_VB`, or
`OOT_DUMP_XFB_PROBE` lifted beyond 64-vertex draws) and compare positions bit for bit; the white vines around the tree
trunks behave the same way. `--probe-tensor` fails (all values zero, the loads too) with the old lowering as well: a
separate, earlier break of the probe.

**Found: the white is occlusion culling.** A transform-feedback capture of the same mesh's positions showed that in
some frames the colour pass has **no draw at all** for a mesh the depth pre-pass drew: the game left it out. The game
tests objects for visibility on the GPU (samples-passed counters, `vk_exec_xfer.c`): pass ~115 renders at 256x144,
its first draw writes a coarse occluder depth (vertex program ce30c06b, from storage buffers the compute kernel
aea924bf fills from the depth image at 0x1500000000, i.e. the previous frame's), then ~85 proxy boxes are drawn
(vs 6192e1e8, depth test LEQUAL, no writes, depth clamp on) between counter resets and reports, and the CPU colours
only what counted samples. `OOT_OQ_LOG=1`: of ~3.5 M reports in a run, 61% were 0. With every report forced visible
(`OOT_OQ_VISIBLE=1`, then made the default) the white is gone: the `OOT_DUMP_WHEN=white:60000@4500` trigger never fires
from frame 4500 to 6665 (it fired at once before), the forest path, Link's house and the close-ups render whole
(`build/logs/oqvis2.png`, `oq-default.png`, noise 0.72 / 0.39), 48 fps either way, no driver reset. Drawing an occluded
object costs only its draws; the depth test still hides it. Still to find: why the proxies count 0 (the occluder's
minimum depth is 0.188 against ~0.77 in the scene, but pass 115 uses a projection of its own with near 0.6; or the
results reach guest memory too late, on the completion thread, after the CPU read them; or the camera cuts of the
cutscene, where the previous frame's depth is another shot). Turning the workaround off (`OOT_OQ_VISIBLE=0`) and
logging the counts per proxy against what the colour pass then draws is the way in. **Root cause, the same day**: timing, not the occluder. Mode `OOT_OQ_VISIBLE=2` writes "visible" when a report is
recorded and the real count when the GPU is done; it shows no white either (trigger silent from 4500 to 6970,
`build/logs/oq-pending.png`, `oq-default2.png`, noise 0.38). The only difference from mode 0 (real counts, white) is
what the report word holds before the GPU finishes, so the game reads its reports before this replay's work is done
and saw the stale word (mostly 0). TotK's port states the same rule ("a result that is not ready counts as visible",
refs/totk-pc-recomp docs/architecture.md); TotK makes ~4 reports a frame, OoT ~580. Mode 2 is now the default (real
counts still apply when the game reads after the GPU is done); 1 = always visible, 0 = the old behaviour. New diagnostics:
`OOT_XFB_POSITION=1` (the transform feedback probe captures gl_Position, `vk_pipeline.c spirv_add_xfb_position`),
`OOT_XFB_COUNT=<n>` (probe every indexed draw of the dump frame with n indices, any pass), `OOT_OQ_LOG=1`.

Diagnostics added: `OOT_DEPTH_EQ_LE=1|2|3` (EQUAL tested as LESS_OR_EQUAL / GREATER_OR_EQUAL / left half LE and
right half EQUAL in one frame); `OOT_DUMP_DRAWS=eq` (the colour target after each of the first 160 draws of the
frame's first full-size depth-EQUAL pass); the dump prints polygon offset state, uniform floats with `%.9g`, keeps
draws with different uniform blocks apart under `OOT_DUMP_UBO` (MAX_ENTS 2048), and notes every mesh draw's task/mesh
program hashes and group count (`note after pass N: mesh draw (...)`). Offline helpers used (scratch, easy to
rewrite): an SPIR-V expression evaluator for `gl_Position` (pair programs by FNV-1a of their SPIR-V; the dump saves
each as `sh_<hash>.spv`), draw pairing by index buffer + count + base vertex, and per-pass depth deltas.

### Status (2026-11-09, late): the dark lighting and black silhouettes — not started beyond observation

State: the white areas are fixed (commits 1e10bf6, 00189c1; `OOT_OQ_VISIBLE=2` default). The next visible problem is
that scenes render very dark and characters are often black silhouettes. Work on it had just begun when the session
was handed off; no code was changed for it, nothing is uncommitted. One dump run (run-199, `--dump 4900`) was
interrupted by the user before the dump frame: `build/logs/dump-199` is empty, no driver reset (0 since boot).

Observations so far (screenshots only, all scored as pictures):
- `build/logs/oqvis2.png` (forest path): ground, rocks, lanterns, fairies drawn and textured but dark; the Kokiri
  child is a black silhouette while the fairies and lanterns (emissive) are bright.
- `build/logs/oq-default.png`, `oq-default2.png` (Link's house): Link's skin and clothes are lit but very dark; only
  Navi (emissive) is bright.
- Earlier dumps: the HDR scene target is r11g11b10f; the sky fill is 24..34; the colour passes test depth EQUAL over
  the copied pre-pass depth; dump-176/179 colour passes produced very dark values for characters (mean ~10-30 of 255
  after `_tm` tone mapping), the ground over-bright/yellow-speckled in some shots (item 2 of "What is still wrong").
- Emissive surfaces are fine, so the base colour textures and the tone map are not wholly broken; what is missing is
  most likely light: direct light (sun / shadow map), indirect light (the light probes / irradiance volume, the 3D
  rgb10a2 512x512x64 texture and the 128x128 r11g11b10f cube bound to the colour programs), or exposure.

Hypotheses, most likely first, and how to test each offline or with one dump:
1. **Shadows say "all in shadow".** The colour programs sample a d16 1600x1600x3 shadow array with a compare sampler
   (`samp ... compare`, `2DArrayShadow`). A wrong compare op, depth range (minus_w / [-1,1]) or a reversed result
   darkens everything the sun lights. Test: dump the shadow pass depth (passes ~0169-0188 at 1600x1600, d16) and
   check it has geometry; then force the compare to pass (a diagnostic switch in `vk_texture.c` sampler creation,
   `si.compareOp`) and compare a scored screenshot of the same shot (use `OOT_DEPTH_EQ_LE=3`'s left/right trick:
   one frame, half with the switch).
2. **Indirect light volume is black.** The 3D texture (512x512x64 rgb10a2, `lut852` in dumps) and the cube map come
   from compute kernels; a skipped or failing kernel leaves them 0. Test: the dump's `images.txt` min/max of those
   textures and of the kernels' outputs; `OOT_CS_TIME=1` lists the kernels that ran.
3. **Exposure / tone map.** The auto-exposure kernels read the HDR target's luminance (small 1x1 .. 8x8 passes,
   pass_0119-0131 chain); a NaN or a huge value (the skipped fog kernel 1a62084a writes ~60% NaN when run) can drive
   exposure down. Test: the 1x1 luminance values in `images.txt` across the chain.
4. **Characters only**: skinned programs' normals (attribute decode of fmt 0x27/0x12, packed normals) wrong, so N.L
   is 0 or negative. Test: dump the G-buffer-like rgba8 target (pass_0193 in dump-165 looked like encoded normals)
   for a character and check the normal colours vary.

Commands (one GPU run at a time, count nvlddmkm events after each):
- Dump the frame that is on screen at a fixed present: `.venv-win\Scripts\python.exe tools\run.py --seconds 160
  --retry 0 --replay build/logs/a-spam.txt --skip-movies --env OOT_SKIP_CS=1a62084a --env OOT_DUMP_SKIP_RUNS=1
  --dump 4900` (frame 4900 lands on varying cutscene shots; never compare pictures across runs).
- Per-draw colour after each draw of the colour passes: add `--env OOT_DUMP_DRAWS=eq` (first 160 draws of each
  full-size depth-EQUAL pass).
- Uniform values of each draw: `--env OOT_DUMP_UBO=48`; kernels' GPU time and which ran: `--env OOT_CS_TIME=1`.
- Scoring: `.venv-win\Scripts\python.exe tools\shot_diff.py <png> [<png>]` (noise above 1.2 is not a picture).

Then the fog kernel 1a62084a (`OOT_SKIP_CS=1a62084a` in every test run): ~60% NaN volume when it runs (Status
2026-11-07); start from `OOT_SM86_PROBE` / `OOT_SM86_COUNT` on it with a warm cache, as described there.

### Status (2026-11-09, night): the ambient light cube was black — a jump table the translator could not follow

Found with three dumps (run-200 to run-203, all frame 4900, 0 driver resets):

- **The cause of most of the darkness.** The level's ambient/reflection light is a 128x128 cube (tex 753 in run-200)
  that 175 draws of the frame sample. The game renders one face of a capture cube per frame (passes ~0090) and a
  filter pass blends it into the light cube level by level (`fs ea1f57f5`, blend 0.2*src + 0.8*dst, the blend
  constant now printed in the dump). The filter's vertex program (`cb9f78977313150b` input) picks the face direction
  with a `switch` on the face index: `ULDC` of the jump table in c1, then `BRXU` (0x958). We lowered it to a Maxwell
  BRX on a uniform register's home; the translator only finds BRX targets by matching an LDC from c1 guarded by a
  bounds-check branch, this program clamps with `USEL` instead, so it logged "Failed to find targets for BRX
  instruction" and dropped the rest of the program: every vertex at (0,0,0,1), nothing drawn, the light cube stayed
  zero for ever. 24 cached translations had the note: 12 vertex (this program under different answers) and 12 compute
  (`865cee2bf198d6f1` and variants).
- **Fix** (`sm86_lower.c` lower_brx): a BRXU with a known jump-table record becomes a compare chain on the loaded
  value (ISETP.EQ on a free predicate + predicated BRA per entry, a plain BRA to the last), and its table in c1 keeps
  the native offsets. BRX (0x949) is unchanged (the fragment programs' tables are matched fine). No KEY_VERSION bump:
  programs with a BRXU get their own cache key through `sm86_probe_tag`, so only those are translated again (a bump
  would translate the whole cache in a run). Checked: the compute program validates (spirv-val) and its SPIR-V grew
  from 2,425 to 3,274 words; in the game the cube chain is non-zero at every level (1x1 = 0.18 0.14 0.10, no NaN), no
  "Failed to find targets" left, screenshot `build/logs/brxu-on.png` (Saria close-up) noise 0.31, skin and hands lit,
  ~43 fps.
- **Also fixed:** a depth texture read without a compare kept the game's compare sampler (`vk_texture.c`
  vk_sampler_point_by_id_locked); Vulkan leaves that undefined (the forward pass reads a 4096x4096 d16 map so).
  `OOT_POINT_CMP=1` restores the old behaviour. Not measured separately.
- **Dump tools:** `OOT_DUMP_TEX` copies depth textures from their depth aspect, every layer of an array (`_L<n>`);
  the blend line names NVN's equations right (1 is add, the dump printed it as "-") and prints the blend constant.

Still open, in order:
1. **The three shadow cascades (1600x1600 d16 array) stay empty** (all 1.0, also as the next frame's compute pass
   finds them), though ~100 draws per frame target them (vertex draws and indirect mesh draws without a task stage).
   An empty map reads "lit", so this does not darken, but there are no sun shadows. Next: per-draw depth of one
   cascade pass (`OOT_DUMP_DRAWS=<pass>` works on colour target 0 only: extend it to depth) and the draws' positions.
2. **The compute variants of the same BRXU programs** also drop `RED.ADD.F32.FTZ.RN` ("Invalid reduction type:
   F32FtzRn" in the translator notes): float atomic adds are lost. Find which kernel `865cee2bf198d6f1` is and add the
   reduction to the translator (a patch) if it matters for the picture.
3. Scenes may still be darker than on the console; compare with a reference picture of the same shot before chasing
   exposure.
4. Then the fog kernel 1a62084a, as planned.

### Status (2026-11-09, later): why the shadow maps are empty, and where the fog's NaN starts

Runs run-204 to run-212 (10 runs, 0 driver resets). No fix yet; both are narrowed down.

**Shadow cascades.** A dump that copies every pass (no `OOT_DUMP_SKIP_RUNS`) shows the pattern per cascade (three
layers of the 1600x1600 d16 array): six or seven passes of one mesh draw each (indirect, mesh program only, no task:
the level's terrain and objects), then one pass of four vertex draws (characters). The vertex draws write depth
(cascade 1 down to 0.21, cascade 2 to 0.14); the mesh draws write nothing. `OOT_MESH_LOG=1 OOT_MESH_LOG_FROM=4900`
reads their indirect arguments back as the draw executes: word 0 (the mesh group count) is 0 for almost all of them
(a few 1), so they draw nothing. The counts are written by the GPU through pointers (the global window), not through
bound buffers, so the dump's binding list does not name the writer; the frame starts with a chain of eleven indirect
dispatches (`0x0x0` in passes.txt: eae34290, 407d3d66, ba9439f5, 8119aee6, 5fa77bc0, 038ada51, 22fc2690, 0c4e2311,
146bd84b, 945063b5, d0dcac29) that are the likely culling. Next: find which of them writes the shadow arguments
(`OOT_GW_TRACE=<hex>` per kernel, or `OOT_SM86_COUNT` on their stores) and why it culls everything for the cascades
(the cascade frustum, or a test against an empty shadow depth pyramid from the previous frame).

**Fog kernel 1a62084a** (input 1a62084a3f963a65, code hash d72101b788a2e492): translates with no notes and validates
now, but still writes NaN in ~69% of its volume, and the frame goes black with it (`build/logs/fog-on.png`, blank).
- `OOT_SM86_COUNT` on its stores: every one of the 110,592 threads per dispatch stores; ~58% take the computing path
  (stores at 0x3a00, 0x3b00, 0x7290), the rest write (0,0,0,1) (0x8ad0..0x8be0). The non-NaN voxels are all 0.
- `OOT_SM86_PROBE=d72101b788a2e492:3990:24`: the value stored at 0x3a00 is NaN in every thread that reaches it.
- Its uniform blocks are plausible (c5[0x1b8] = 1, c4[0x384] = 0.0066; `OOT_DUMP_UBO=64`, run-210).
- The block at 0x3200-0x3510 blends eight light-probe samples trilinearly (weights from validity bits, SGXT/FSEL,
  a 1e-5 floor), sums the weights and takes the reciprocal (MUFU at 0x33d0), then scales spherical-harmonics terms
  (0.2821, 0.0796). `OOT_SM86_PROBE=d72101b788a2e492:33b0:29`: **the weight sum is exactly 0 in every thread** ->
  1/0 = inf -> inf * 0 = NaN. Next: probe the weights' inputs (R85, R84, R43, R76 before 0x32b0) and the probe data
  the kernel loads through its storage-held pointers (`LDG [UR10+0x60]` at 0x3220 etc., bank 0 + 0x60 table); zeros
  there would explain it.
- Not the cause: the unbacked storage buffer in the log. `vk_exec_draw.c` now names the program and binding:
  0x3f12864e3e51b3a4 (two floats read as a pointer) is kernel 374126bf's c3+0x640, and 0x1501194600+0xfd200 (render
  target memory used as storage) is programs 745220f3 and 3f15911a, c0+0x110/0x1c0. Both are leads of their own.

### Status (2026-11-09, night 2): the shadow culling chain, and a virtual pool the GPU could not see

Runs run-213 to run-219, 0 driver resets.

- **Who writes the shadow draws' counts** (`OOT_MESH_WATCH=1` with `OOT_MESH_LOG`): each mesh-only object has a
  0xf33080-byte buffer whose head holds the indirect arguments, +0x00 the main view and +0x10/+0x20/+0x30 the
  cascades; kernels 027a50b7, 68e84b77 and 75dc8d60 write them (vote, popc, one ATOMG per view; per cascade a
  six-plane frustum test against c3[0x520..0x5dc]). All three translate cleanly.
- **The chain** (`OOT_IND_LOG=4800`, every frame the same): c8b82584 1, 7f05518a 1, e6c023a5 7, 44e1015c 10, 3076c0a9
  1, 032f18b8 1-6, 3360a3c0 26-59, 75dc8d60 1, **68e84b77 0**, **2bb46270 0**, d295f592 1, 3f15911a ~60 groups.
  `OOT_SM86_COUNT` on 68e84b77 confirms no thread of it ever runs. 2bb46270's size is written by e6d5432f (into
  0x29bb4c020, the third slot of a window buffer); 68e84b77's (0x29f9aa000) by the occlusion kernels 959ef9bb /
  fac1644d / d1bf02dd. e6d5432f picks a draw distance per object type (a 3-entry jump table at 0x1380, c3 limits)
  and exits objects beyond it; counted over 120 frames: 2.6 M threads start, 0.2 M reach 0x3150, none reach the
  store at 0x3420 or the atomic at 0x3530. Next: what decides that branch (and quote the `OOT_SM86_COUNT` value in
  PowerShell: unquoted, `17e0` and `6e70` are read as numbers, 17 and 6E+70).
- **The virtual pool's physical pages (found on the way, opt-in for now).** The game maps its 305 MiB physical
  (GPU-only) pool into a virtual pool and places render targets and transient buffers there (0x15xxxxxxxx). The NVN
  front end only shadow-copied mappings from CPU-visible pools and dropped the physical ones, so every storage buffer
  there was bound to the zero buffer (0x1501194600+0xfd200, read by 745220f3 and 3f15911a). `OOT_VPOOL_PHYS=1`
  (`nvn_api_core.c` nvnMemoryPoolMapVirtual) forwards them to `vk_pool_map_virtual`: the warnings go away, two runs
  without a reset, but the culling counts did not change and the shadows stay empty, so it stays off until it is
  shown to help. The unbacked-buffer log line now also says how the first byte resolves.
- Screenshot `build/logs/vpool-on2.png` (Mido, `OOT_VPOOL_PHYS=1`): noise 1.22 from the known over-bright, speckled
  ground in front; the rest is a picture. `vpool-on.png` was black (a fade; frame 4900 of the same run is a normal
  picture).

### Status (2026-11-09, night 3): the object counters that never start, and stores lost to a wrong pointer trace

Runs run-220 to run-224, 0 driver resets. The shadows are still empty; the cause is one step further back.

- **Why e6d5432f appends nothing** (`OOT_SM86_COUNT`, values quoted): it reads a 64-byte record per object from a
  table it reaches through c4 (`LDC.64 c[4][type * 0x40]`, then record `(index * 4 + 0x82..0x85) * 16`), and the low
  byte of the record's last word is a counter: the object goes on the work list only when it is exactly 0xff (0x3440
  `P5 = R13 == 0xff && !P6`, 0x3450 PLOP3 table 0x80 `P4 = P4 && P5`, decoded by hand: the lowering is right), and
  the kernel advances a counter that is not 0 (store at 0x3420). Every object's counter is 0, so the shadow culling
  (2bb46270, and 68e84b77 behind the occlusion kernels) gets 0 groups. Its sibling 374126bf (same layout, the same
  counter word) writes 1 when it appends an object (0x4570) and R11 + 1 (0x4330). Who sets the first 0xff is open: no
  GPU writer of the record range does it (`OOT_WATCH_RANGE`, run-222); next suspects are the game's CPU code (a
  record it fills when an object appears) and the sync of CPU writes into a range the GPU also writes.
- **Fixed: stores lost to a partly traced pointer** (translator patch 0021,
  `0021-global-window-after-partly-traced-pointers.patch`). 374126bf's two counter stores take their address through a
  phi whose real source is the indexed c4 table, which the search cannot trace, and whose other path held a float of
  c3+0x640 in the same register, which it took for the base: the store went only to "c3+0x640" (two floats, so the
  zero buffer) and was dropped. Now a compute program's global access whose sources are partly traced tries the traced
  buffers first and then the global window. e6d5432f has 2 such accesses and 68e84b77 3. Translator identity
  9123437e...; checked: 374126bf validates; level screenshot `build/logs/x21-on.png` noise 0.23; 0 resets (the first
  run translates again, slower). The counters stay 0, so the shadows do not change yet.
- Lichii (nx2 branch, GPL, read for behaviour only): its NX2 notes cover mesh-shader emulation and sparse textures for
  another title, nothing on shadows, culling or fog in this game.

### Status (2026-11-09, night 4): the culling kernels that had no window; the first shadow geometry

Runs run-225 to run-229, 0 driver resets.

- **`OOT_GW_SKIP` was the blocker.** Since 2026-11-08 the two jump-table culling kernels 374126bf and 358beb27 got no
  global window by default (their stores, traced to floats in c3, seemed to corrupt the culling buffers; 3 crashes in
  12 runs then). Without the window every pointer access of theirs reads 0 and every such store is dropped, and they
  are the ones that write the per-object records (374126bf: counter 1 when it appends an object, 0x4570). With
  translator patch 0021 those stores go to the wrong candidate first and then to the window, so the skip list is now
  empty by default (`OOT_GW_SKIP=374126bf,358beb27` restores it). Result (run-228, frame 4900): 68e84b77 gets 1 group
  instead of 0, and the far cascade holds real shadow-casting geometry for the first time (pass 0164: 24,637 distinct
  depths, small props and a curved surface; `build/logs/gwskip-none.png` noise 0.64, a picture). The terrain is not in
  the cascades yet and e6d5432f's counters are still 0 (its own append path still never runs), so the shadows are
  far from complete. Two runs with the new default, no crash (run-229 ran 200 s; its last screenshot is black, a
  loading/fade moment at 1% low 3.9 fps; frame 4900 of run-228 is a normal picture).
- Correction to "night 3": c4[type] is not the record table but a pointer cell (`LD R2, [R2]` at 0x550 follows it);
  the cells are in a pool (the log's storage-held pointers read them: 0x29bb2b940 -> 0x2a19e9000), so the records are
  at those addresses. `OOT_DUMP_BUF` now logs why it snapshots nothing (a range in no pool) and also saves the CPU's
  view of the range once (`cpu_<pass>.bin`) next to the GPU's copies.
- Next for the shadows: why e6d5432f's own append path (counter == 0xff) never runs; with the records now written,
  count R13 again and look at what 358beb27 writes.
- Later (run-230..232, 0 resets, no crash in 5 runs with the new default): e6d5432f's counters are still all 0 and its
  append never runs; the chain's sizes vary with the shot (68e84b77 and 2bb46270 0 again in that run). 374126bf is
  not dispatched at all in the level (`OOT_SM86_COUNT` saw 0 dispatches), so it is not what starts them here. The
  record table behind cell 0x29bb2b940 (0x2a19e9000, 991 records) has counter byte 0 in every record on both the CPU
  copy and the GPU copy (`OOT_DUMP_BUF` with `cpu_0000.bin`), the copies differing only in a word the GPU writes
  (+0x38 of each record): no CPU write of 0xff is lost there. Not settled: whether that is the table e6d5432f reads for
  the objects in question (it picks one per object type through c4 rows 0-3), and which program or CPU code should set
  0xff. Parked here for the fog.

### Safety (learned the hard way)

Six driver resets on 2026-11-08, all from the indirect draws before `7ed675d`; none since. After every GPU run count
nvlddmkm events; after any reset stop and reboot before the next run (check `LastBootUpTime`). Bisect a hang by
halves with `OOT_MESH_IND_ONLY` / `OOT_SKIP_CS` and put `OOT_SM86_COUNT` points in the suspect: its dump at the device
loss shows the loop. Run one game at a time (`run.py` refuses overlaps; run them one after the other, never in
parallel tool calls).

## Status (2026-11-05, later the same day)

Steps 1 to 3 of the order of work below are done; step 4 is at 86% of the corpus; step 5 is where the work is.

- The translator builds (`build/xlate/oot_xlate.exe`, `python tools/build_xlate.py` from the Windows venv, 20 s).
  `tools/xlate/Program.cs` now echoes the job's `c1` into the `.res.json` (the host binds the lowered jump tables).
- `tools/shader_census.py` (NVN2 version; `python3.14`, Linux) reads the archives and writes
  `build/v131072/shaders/{ctl,code}/<key>`, `programs.json` and `opcodes.txt`: 14,363 programs (10,994 pixel, 3,106
  vertex, 251 compute, 9 mesh, 3 task) from the four `Shader/*.bfsha.zs`, the `Layout2Shader` bfsha and the two agl
  `.sharcb` of the ApplicationPackage sarc; the ExeFS modules hold none. `--lower` builds `runtime/sm86_lower.c` with
  gcc (`tools/sm86_lower_main.c`) and lowers every program, grouping the failures with the failing instruction in
  `lowered.txt`; `--translate` runs the translator over the lowered ones through WSL interop (`failures.txt`).
  `tools/sm86_dis.py <key>` lists a program's fields. The corpus is regenerated from the RomFS in 2 min; later runs
  reuse `programs.json` (`--extract` reads the archives again).
- The control section's fields are settled (the table below, columns "Reading", and `sm86_control_parse`): size at
  `+0xd4` (the BNSH archives repeat it at `+0x30` and `+0x98`, the agl packages leave those 0), instruction end at
  `+0x38`, jump-table constants at `+0x40` with their length at `+0x3c`, code size at `+0x44`, the stage at byte
  `+0x56` (NVN's enum plus 6 task, 7 mesh), the compute workgroup size at `+0x58`; the tail past `+0xe0` holds a u32
  count and 16-byte records `{BRX instruction offset, entry count, 0x10000, byte offset of the table in c1}`.
  Graphics code starts with the 0x80-byte Turing+ header, whose first 0x50 bytes are the Maxwell header the
  translator reads. The early-fragment-tests flag is not identified (candidates: `+0xa4` bits 0 and 8).
- `runtime/sm86_lower.c` lowers 13,444 of 14,363 programs (93.6%). Registers: the lowering's uniform registers
  and scratch live above the program's own; a uniform register loaded by a single direct `ULDC` (or `UMOV`), or by
  one earlier in the same basic block, is forwarded as that constant and needs no register; when the program leaves
  no room, a liveness analysis over the basic blocks (`insn_regs`, exact destination widths, reads over-stated only
  for texture coordinates) finds scratch among the registers dead at each instruction and a uniform register's
  register among those dead over its span. Uniform predicates take free predicates, or are swapped through a
  register around each instruction naming them. What is left: 656 vertex shaders (skinning: up to 247 registers
  live at once, no adjacent pair free where a load needs one), the surface stores (`SUST`, 86 compute programs),
  `ISBERD`/`ISBEWR` (task and mesh stages), a few rare forms (`lowered.txt`).
- The host (`runtime/gfx_shader.c`, `vk_exec_draw.c`, `vk_dump.c`, the shader probe and the pipeline self-test)
  reads the NVN2 control through `sm86_control_parse`, hashes the control section (its own size) and the whole code
  block for the cache key (`KEY_VERSION` 5, `IN_VERSION` 2, `PRG_VERSION` 2), lowers before translating, and binds
  the lowered program's c1 (`gfx_shader_c1`, uploaded once per shader) as uniform slot 1 instead of the game's, since
  the jump tables' targets moved with the code.

## Status (2026-11-06)

The game plays its title cinematic through a host decoder, and the picture is still black for a reason outside the
shaders lowered so far: the game's video path runs on the tensor cores.

- `runtime/ov_movie22.c` decodes the cinematics with dav1d (linked statically from MSYS2, BSD-2, row in
  `THIRD_PARTY_NOTICES.md`): the title is "video/av01" 1920x1080 at 60 fps, a 64 s stream with a key frame every
  321 samples (`nn::movie::Buffer` flag 4 marks those, not the end of the stream, which is an empty flagged buffer).
  Frames come back to the game as NV12 in the buffer it registered (it reads planes 0 and 1 only), with the
  sample's time stamp. Audio is still dropped. The decode runs on the game's decode thread; the game then runs
  at about 14 fps there, to be profiled (dav1d on 4 threads, the NV12 interleave, or the game waiting on frames).
- The lowering grew: SUST (bindless store, the handle loaded from the constant word), LEA and IMAD.X carries
  through comparisons and predicate adds (the translator ignores Maxwell LEA's carry predicate and X), the
  uniform register budget counts registers read before any definition, uniform pairs are allocated adjacently
  in both the pool and the liveness mode, the swap register for uniform predicates comes from the liveness
  allocator when the pool is full, and only the predicate fields an op has are rewritten. Corpus: 13,514 of 14,363
  lower (94.1%), all translate.
- What blocks the picture: every shader of the video path (19 compute programs, dumped by the host to
  `shader-cache/<key>.fail.{ctl,code}` for `tools/sm86_dis.py`) uses the tensor-core instructions: HMMA (0x03c,
  4,478 instructions), LDSM (0x03b, 1,670) and MOVM (0x03a, 24). They are the game's neural upscaler (the SDK
  list names "SMAA_For_NDSR"; the swapchain is 2560x1440 while the movie is 1080p), and they are not in the RomFS
  archives the census reads (nor in the ExeFS modules; the raw `romfs.bin` has no match either, so they sit in a
  compressed file the census does not visit, or the game builds them). NVN2 device property `0x67`, kept by the
  game as a capability flag and required by its video surface setup, is most likely "supports tensor cores".
  Next: lower HMMA.16816 / 1688 f16 (LDSM's matrix loads into per-lane fragments, the warp-cooperative product
  through SHFL and FFMA, MOVM's transpose) in front of the translator, or find the non-tensor path if the game has
  one (with `0x67` = 0 the player aborted in its surface setup; the shader's decision may lie elsewhere).
- Reaching the game proper: `OOT_INPUT_SCRIPT` (lines "<ticks> A 10") passes the title; the file select and the
  opening are cinematics too, so they need the same path.
- The intermittent death right after the first frame (one run in three, exit 40 or 5): the crash handler's dump
  (`main.c`) caught two. One executed from the thread's own stack while logging a guest `Break` from
  `sdk+0x1a5560` (the SDK abort), the log line itself garbled; the other executed from the heap region of the
  `Cpu` structures with `rt_svc+0x59` (after `svc_body`: the host-depth bookkeeping, `kern_preempt_pending` /
  `kern_host_left`) as the return address and lifted `main+0x3779bc0` below it. Both point at the asynchronous
  preemption (`kernel.c`, the interrupt trampoline that pushes the interrupted thread's registers on its stack).
  Test: `OOT_ASYNC_PREEMPT=0` over several runs; then read the trampoline against a thread suspended inside host
  code.

## Status (2026-11-06, later): the title frame decodes, the upscaler does not

The movie frame reaches the GPU and converts correctly: the first compute pass of the game's video path writes a
1920x1080 RGBA8 image of the title cinematic (Link on Epona by the river; dump `build/logs/dump-93`, pass 0000,
147,050 distinct colours). What follows it, the neural upscaler's twelve kernels and the final 2560x1440 pass, still
produces a near-constant image and then zeros, so the screen stays black. Found and done on the way:

- The tensor-core instructions are emulated in the lowering (`runtime/sm86_lower.c`, "tensor-core instructions"):
  LDSM (1, 2 or 4 matrices, plain or transposed) through SHFL.IDX of the row addresses and LDS; HMMA.16816.F16.F16
  through the PTX fragment layouts (lane = 4g + t), SHFL.IDX and HFMA2 (f16 accumulation, the tensor core accumulates
  wider); MOVM through two SHFL and a PRMT. Also added: SUST (bindless store, the handle a dword index in the
  constant word as TEX's), HFMA2 (all forms, .RELU through HSET2.GT.BF and HMUL2), F2FP (two F2F.F16.F32 and a
  PRMT), HMNMX2 (HSET2, HADD2, HFMA2), BAR.SYNC. All 19 kernels lower and translate; the corpus stays at 13,514
  lowered, 0 translation failures.
- NVN2's constant bank 0 for compute is not Maxwell's driver bank. `nvnCommandBufferPushComputeConstants(cmd, 0,
  offset, bytes, data)` (an NVN2 entry point the host only stubbed) pushes the kernel's parameter block, and the bank
  reads as: 0x10 reserved bytes (the dispatch size is put there), the pushed constants, then the bound resources in
  slot order, storage buffers {u64 address, u32 size, u32 0} for control[0xb2] slots, texture handles (u64) for
  control[0xb3] slots, image handles (u64) for control[0xb6] slots (the counts are bytes of the control section;
  control[0xb8..0xba] holds the bank's size). Measured on the kernels: the final pass pushes 0x54 bytes and reads its
  storage slot 2 at 0x90, textures 0 and 3 at 0xa0 and 0xb8, image 1 at 0xc8; the NV12 pass pushes 0x40 bytes and
  reads textures 1 and 2 at 0x58 and 0x60, image 0 at 0x68; the kernels without a push read storage slots 0 and 1 at
  0x10 and 0x20. The host records the push as OP_PUSH_CC, builds the bank at each dispatch (`vk_exec_draw.c`
  nvn2_c0_fill_cs), consumes the push with the dispatch, and resolves the shader's bank-0 words back to the bound
  slots for its descriptors (nvn2_c0_slot, cbuf_handle, ssbo_info). The graphics stages still get Maxwell's driver
  bank; the game's vertex shaders read 64-bit addresses at c0+0x60, which is the same scheme and will need the push
  (nvnCommandBufferPushGraphicsConstants is still a stub) when a draw depends on it.
- Diagnostics added: `OOT_BIND_LOG=1` logs the compute stage's binds, pushes, dispatches and inline uniform updates
  from frame 290 on; `OOT_PROG_DUMP=1` hex-dumps the memory around compute programs' control sections at
  SetShaders; the frame dump prints each stage's bound texture, image and uniform slots per pass and resolves texture
  declarations as the replay does (`vk_exec_res_handle`), so storage images written by dispatches get their PNGs.
- The death after the first frame, one run in three, is the SDK's own abort in
  `nn::os::detail::MultiWaitImpl::InternalWaitAnyImpl` (sdk+0x167be0): after a holder's virtual IsSignaled (vtable
  +0x18) returns true, it aborts when its count of found holders reaches the caller's capacity (the compare before
  the jump to sdk+0x167ec4, which calls `nn::diag::detail::AbortImpl()` through the GOT). The host then dies with
  the abort's return address as the instruction pointer (`executing 0x7005367ec8`, the LR of that call, rcx holding
  the Cpu), before `AbortImpl(const char *, ...)` is reached (an observer on it, `ov_oot.c` obs_abort_impl, never
  fires). Two bugs: more holders report signalled than the game expects (an nn::os event or timer the host leaves
  signalled at start-up, to find), and the abort path crashing instead of reporting. `OOT_ASYNC_PREEMPT=0` does not
  prevent it.

Next (as of that section): find where the upscaler's values go wrong. Resolved in the section below.

## Status (2026-11-06, night): the cinematic plays

The title cinematic renders: the AV1 frame decoded by dav1d, upscaled to 2560x1440 by the game's neural network
through the emulated tensor-core instructions, composited with its subtitles (`build/logs/run-184.png`, 27,548
distinct colours). At 21 fps while the cinematic plays, 60 fps before it. Six distinct defects stood between the
black picture and this, found one after the other with the tools listed at the end:

- The translator's dead-node removal (`Optimizer.RemoveNode`) followed into any producer whose use count reached
  zero, even one still live through another destination. SHFL defines a value and a validity predicate; dropping the
  dead predicate copy stripped the use lists of the shuffle's lane index, and the next pass deleted the index
  computation: every emulated LDSM and HMMA shuffled from an undefined lane. Patch 0013 follows only into producers
  that are dead through every destination.
- PRMT, the byte permute, was an unimplemented logging stub in Ryujinx in every form (R, I, C, Rc). Half the lowered
  programs use it (packing halves), and the emulation builds every half pair with it. Patch 0014 implements it; a
  constant selector (the usual case) is resolved at translation time, the half-word packs 0x5410 and 0x7632 to two
  instructions.
- With those two fixed the kernels' SPIR-V grew to 16-22 MB (over a million instructions; the NVIDIA driver took 20
  to 30 s to build each pipeline): every emulated shuffle was a helper-function call with four pointer arguments.
  Patch 0015 emits SHFL.IDX with the plain mask (clamp 31, no segment) on a 32-lane subgroup as the subgroup shuffle
  itself, the lane index masked to the subgroup. The pipelines still take 0.5 to 30 s to build the first time (the
  driver's disk cache keeps them); the emulation's instruction count is the next performance item.
- The uniform-predicate swap (`sm86_lower.c` lower_one): when a program uses all seven predicates, a uniform
  predicate is swapped into the predicate of the same number around each instruction naming it, and the game's
  predicate restored after. The first neural-network layer ends its tile loop on `PLOP3 P0 = UP0`: the restore put
  the old P0 back over the result, the loop ran until its counters wrapped (2.2 s per dispatch, measured with the
  kernel probe) and the GPU hit Windows' 2 s timeout, reported as a device loss with the instruction pointers of
  every killed warp. The swap now leaves the instruction's own predicate outputs alone.
- The upscaler's final pass stores its 2560x1440 picture into the game's sRGB output texture, which Vulkan cannot
  make a storage image; the host bound a dummy. `vk_texture.c` now creates such images with EXTENDED_USAGE and the
  STORAGE usage, and `vk_exec_draw.c` get_view stores through a view of the UNORM sibling (`vk_unorm_of_srgb`).
- The lowering put the scheduling words of graphics programs at absolute multiples of 32 bytes from the block
  start, but Ryujinx counts them from the end of the 0x50-byte header: every third instruction of every vertex and
  fragment shader sat where the decoder skips a scheduling word, and the scheduling word was decoded as an
  instruction. The compositing quad's texture fetch was one of them. `addr_of` now counts from the header's end.
  (The pipeline check `module_binding_missing` in `vk_pipeline.c` also had to learn that a declared but unreferenced
  binding is not a layout error: the translator declares a constant buffer whose binding it reserved even when the
  load was removed.)

Also on the way: the game's movie worker feeds samples whenever it has them without consulting the input event;
decoding each as it came ran the whole stream through dav1d in seconds, the ready ring dropped pictures and the game
waited forever for the frame of a sample it had sent. `ov_movie22.c` now queues the samples and decodes them as the
ring has room. The SHFL emitted by the emulation sends its validity predicate to PT (it clobbered P0). The device
fault handler keeps the vendor's crash blob (`fault-vendor.bin` next to the log), compute pipeline builds over 100
ms are logged with their SPIR-V size.

Tools that paid off, all kept: `oot.exe --probe-tensor` (a hand-assembled SM86 kernel: LDSM x4, LDSM x2 transposed,
HMMA.16816.F16 against a CPU reference, through the lowering, the translator and a real dispatch);
`oot.exe --probe-cs <dir>` (one of the game's compute kernels dispatched in isolation with stand-in resources and
timing: ctl, code, bindings.txt; the kernel's weights from an `OOT_DUMP_BUF` snapshot); `OOT_DUMP=dispatch` (the
frame dump armed on the first compute dispatch); `OOT_SKIP_DISPATCH_FROM=<n>` (bisecting a device loss down to a
dispatch); `OOT_CS_DUMMY_TEX=1`; `OOT_MQ_LOG=1` (nn::os LightMessageQueue and EventType traffic with threads and
call sites, which found the movie worker's stall); `OOT_SM86_STUB=hmma|ldsm|all` (the emulation replaced by zeros,
for timing); `OOT_THREAD_DUMP=s<seconds>` with host backtraces (the render thread inside `nvgpucomp64.dll`).

Open: the frame rate (21 fps in the cinematic: the emulated kernels, and the decode on the game's thread); the
start-up crash, now about one run in two (the SDK abort in MultiWait, see the previous section); the 656 skinning
vertex shaders the lowering refuses; the mesh stages. The game has not yet been taken past the cinematic.

## Status (2026-11-06, late night): the title screen as the player sees it

Watched with sound for the first time (the user), three things looked wrong. All three are settled:

- **The game froze and its music stopped for a while.** It was out of focus: the host reports the window's focus to
  the game as the console reports the HOME menu, and the game pauses itself. By design. `OOT_ASSUME_FOCUS=1` pins the
  focus for watched test runs. The title movie has no sound track: no run has ever created a movie AudioDecoder.
- **The cinematic ran at a third of its speed.** The game takes one movie picture per game frame and does not look at
  the time stamps, so at 21 fps the 60 fps movie played at a third of its speed (a pacing line every 300 pictures:
  5 s of video in 15 s). `ov_movie22.c` now hands pictures back on a playback clock and skips the ones already
  late. dav1d runs on a host thread per decoder, which signals the output event; the events are interprocess
  system events for that. The frame rate stays at about 22 fps. Decoding was not what limited it.
- **The logo was pixelated.** Its ASTC blocks were decoded right (ARM's astcenc agrees texel for texel) but put in the
  wrong places. Inside a GOB the Switch 2 puts the 16-byte column at 64 bytes and the second row bit at 32, the
  reverse of the Tegra X1. This was found by matching each block's edges against every other block in the stored
  order, and holds for the ASTC logo (16-byte blocks) and the BC4 masks (8-byte blocks). `tex_deswizzle.c` applies it
  to elements of 8 and 16 bytes. Smaller elements keep the X1 order: no game data has settled them yet.
  `OOT_GOB_SWITCH2_MIN=1` (or 2, 4) tries the Switch 2 order for them, and `OOT_TEX_RAW=<w>x<h>` writes a texture's
  raw blocks for such checks. Frame dumps now also write the alpha of 8-bit RGBA images (`..._a.png`).

Getting to the title menu needs an A press (`OOT_INPUT_SCRIPT` with `600 A 10`).

## Status (2026-11-06, end of the night): the title runs at 60 fps on the tensor cores

The cinematic was GPU-bound: 48 ms of GPU time per frame, all of it the upscaler's kernels, each emulated HMMA about
116 Maxwell instructions (32 shuffles, 32 half FMAs that the translator expands further). `OOT_CS_TIME=1` showed the
time spread over seven kernels. HMMA now runs on the host's tensor cores through VK_KHR_cooperative_matrix:

- `--probe-coopmat` found f16 16x8x16 subgroup matrices on the RTX 4070 and measured their per-lane element order:
  HMMA's register fragments, except that A's four registers go in as 0, 2, 1, 3. No data moves between lanes.
- The lowering (`sm86_lower.c lower_hmma`) emits a marker for an unconditional HMMA of a compute program with
  whole-subgroup workgroups; translator patch 0016 decodes it into one IR operation with two results and emits
  `OpCooperativeMatrixMulAddKHR` with element inserts and extracts. `vk_caps.c tensor_cores` turns it on when the
  device has the shape, half floats and 32-lane compute subgroups (`OOT_TENSOR_CORES=0` keeps the emulation); the
  choice is mixed into the cache keys.
- The first full run drew garbage at 60 fps. Bisection (`tools/bisect_kernels.py`, after a first attempt whose
  allow-list never matched) found kernel 5d415995, and the isolated kernel on random data found the cause in the
  translator: its AST optimizer moved single-use memory loads to their use, past later stores, so the kernel read its
  weights after it had reused that shared memory. Patch 0017 keeps memory reads in place. The emulation hid the bug
  by luck of scheduling.
- Result: 4.3 ms of GPU time per frame, 60 fps (the game's cap), the movie on its clock with no skipped pictures,
  and screenshots scored as pictures (`build/logs/check-on.png` against the emulated `check-off.png`).

Open: spirv-val asks for the VulkanMemoryModel capability next to CooperativeMatrixKHR (the driver accepts the
modules; switching the translator's memory model touches every shader). The tools made tonight are in CLAUDE.md
("Running and checking") and the project skill `verify-shader-change`.

## Status (2026-11-06, first play): into a game, then black

The user played from the title into a new game with a Switch Pro Controller (`tools/run.py --seconds 3600`,
build/logs/run-243.log). The title, menu and opening cutscene run at a steady 60 fps. Two problems:

- **The opening's voice-over is silent.** The game is voiced: about 7,700 voice streams in nine languages, an English
  voice bank for 99 cutscenes including `Dm_Kokiri_0000`, the Deku Tree's opening. No audio service was refused
  and the renderer runs. Next: a run with `OOT_TRACE=fs` to see whether the game opens the voice bank and streams
  (if not, suspect the language the host reports, `ov_system.c ov_language`).
- **Gameplay is black.** When the opening hands over to the game, the screen goes black while the game keeps
  presenting frames at 60 fps (no crash, no stall). In the same moment the game starts calling
  `nvnCommandBufferDrawMeshTasks`, which is a stub that draws nothing, and only 51 distinct shaders were seen in the
  whole session. The world is most likely drawn with mesh shaders: the mesh stages (listed as open since the
  start of Phase 3) are now the blocker. Next: a frame dump of a gameplay frame (`tools/run.py --dump`) to confirm
  what is drawn and how much goes through mesh tasks. Reproduced unattended (run-244): A pressed every two seconds
  from the title and `OOT_MOVIE_SKIP=2` (`tools/run.py --skip-movies`) reach the mesh draws in about 25 s, and the
  screenshot is pure black.
- **Faint rectangle outlines in the cinematic** (the user, on a frame filled with blue sky): the upscaler's tiles
  show at their edges. Not yet measured whether the emulated HMMA shows them too (tensor cores round once, the
  emulation per step) or whether a tile border is handled wrongly.

## Status (2026-11-06, past the opening): the level renders, then the GPU times out

The black screen after the opening cutscene was not the mesh stages. The game marks a movie's end with an empty
buffer of flag 8 (flag 4 marks the first sample of a stream); `ov_movie22.c` took 4 for the end, so no movie ever
ended and the game waited for the end-of-stream frame forever (run-248 logged the flag-8 buffer). Past it, the game
aborted in `nn::prepo::AsyncContext::GetCompletionEvent` (a play report's context; `ov_system.c` now completes it at
once with a signalled kernel event attached through the SDK). With both, the Kokiri Forest level renders: about 80
passes, 744 draws and 126 dispatches a frame (run-257), for about ten frames, until a device loss with the GPU
timeout's pattern (one instruction pointer per killed warp: a shader that runs too long, as the upscaler's loop did).

Open, in order:
- (Done, the same night) The GPU timeout on the first level frames. `OOT_SKIP_CS=all` removed it (a compute kernel);
  `OOT_SKIP_CS=!<list>` (run only those) and leaving one out at a time found a pair, eb67c232 (one thread copying
  light data from its uniform block, sampling two textures) and c82d3cba (a volume kernel with counted loops), that
  hung only together. The frame dump armed on eb67c232 (`OOT_DUMP=cs:<id>`) showed its two textures unbound: the
  host placed NVN2's bank-0 resource tables after the pushed constants by the length the game pushed, but the shader's
  layout ends the tables at the u16 at control +0xba (eb67c232 ends at 0x40 with 1 storage and 2 texture slots and
  reads its storage slot at 0x20). With the base from +0xba every compute texture slot of the level frame resolves,
  the level runs (662 frames, A/B against the old rule: device lost after 14) and the Deku Tree's dialogue plays
  (subtitles over a still black scene). Also translator patch 0018: NMin/NMax/NClamp and a saturating, NaN-to-0 F2I,
  as Maxwell defines them (not the cause, but SPIR-V left them undefined).
- The scene behind the dialogue was black for two reasons (2026-11-06, dumps of frame 1500 with
  `OOT_DUMP_SKIP_RUNS=1`; the dump's pass cap is now 1200 and it notes draws and dispatches skipped for a missing
  shader, with the code hash `tools/kernel.py` prints):
  - The last steps of the frame, the 1080p-to-1440p kernel 0a9afab0 and the kernel after it 7034cdb8, did not
    lower; the frame showed only the subtitles drawn over an empty target. The lowering now has SULD (23 level
    kernels), REDUX (27), HSETP2, HMUL2, negated HMNMX2 operands and TLD4 with per-tap offsets and fp16 results.
    With them the level shows: the sky and the forest's shapes (`build/logs/level-up.png`), the geometry still
    mostly black.
  - The volumetric fog: kernel 1a62084a (8x4x4 groups, the 64x36x96 in-scattering volumes) writes only 0 and NaN,
    and the integration along depth and the fog's full-screen pass carry the NaN into every pixel of the scene.
    Its uniform blocks, textures and storage buffers are bound and plausible; the cause is open.
    `OOT_SKIP_CS=1a62084a` leaves the fog out for now.
  - The materials: most fragment shaders take their texture handles from memory (bindless: a handle loaded into a
    register, R2UR, then TEX through a uniform register), which Ryujinx 1.3.2 cannot resolve; they sample the
    placeholder. True bindless (a descriptor array over the texture pool, indexed in the shader) is the next big
    piece.
- 605 programs the lowering refuses in the level (`gfx_shader: translation failed`), so about 280 of 744 draws are
  skipped: unknown opcodes 261, `TEX unpacked operands` 128, scratch and predicate pressure 53, `RED address` 37,
  uniform datapath opcodes 35, and smaller groups (TLD, TLD4, HMNMX2, PLOP3, STG of wide zeros).
- The mesh stages: the level draws with `DrawMeshTasksIndirect` too (stubs).
- The movie start stutters for about 8 s (every other frame ~160 ms while the GPU needs 5-13 ms: CPU side), and the
  cutscene's audio stream runs 10-20 s off the picture (the user); the opening's voice lines are silent.
- (Done) `OOT_MOVIE_SKIP`: the cutscene follows the time stamps of the pictures it shows, not the movie's end alone
  nor its audio stream (the user: the cut to the game came before the lagging audio ended). The skip now hands back,
  once the game has sent its end buffer, one black picture stamped with the last sample's time, then the end; the
  quick route (`tools/run.py --replay build/logs/a-spam.txt --skip-movies`) reaches the level in about 75 s.
  The game has no button to skip the opening (the user).

## Status (2026-11-06, the start-up crash): two host threads on one fiber stack

The death right after the first frame (about one run in two: an access violation executing a heap or guest address,
or in `kern_fiber_finalize`, often on two host threads at once) was not the SDK abort in MultiWait, nor the
asynchronous preemption (`OOT_ASYNC_PREEMPT=0`: 4 of 10 runs died, against 2 of 10 with it). Every dump had the same
guest stack: `nn::os` waits (`obs_ev_trywait`, `InternalWaitAnyImpl`) down to a `WaitSynchronization` (SVC 0x18)
whose return landed on garbage. `OOT_FIBER_LOG=1` (new: a line per InitializeFiber, FinalizeFiber, SwitchToFiber
and fiber return, with guest and host thread) showed the pattern before each death: SDK 22's `nn::async` coordinator
yields a fiber with `SwitchToFiber(nullptr)` on one worker, another worker picks it up, and the fiber returns there.
`fiber_main` then switched to the base fiber of the thread it had *started* on: clang took the address of the
thread-local `tcur` once at entry and kept it in a callee-saved register across `rt_call` (LLVM treats a
thread-local's address as fixed within a function). That base fiber was already running on its own thread, so two
host threads ran one stack. `kernel.c` now reads `tcur` after a switch only through the out-of-line `tcur_now()`
(in `fiber_main` and after `SwitchToFiber` in `fiber_switch_to`, which also wrote the context's host depth into the
thread it had left). Result: 20 runs of 12 s, none died (before: 8 of 12, 2 of 10, 4 of 10 in three series).

Also: the SDK's argument-less `nn::diag::detail::AbortImpl()` (the one `InternalWaitAnyImpl` calls) gets an observer
in `ov_oot.c` that, for the MultiWait abort, lists the holders by vtable; the observer on the four-argument overload
never fired because the SDK does not call that one there. Left as is: `hostfn_tracked` and `rt_original` keep the
thread they started on across a call that can move a fiber (`c->thread` before the call), so with asynchronous
preemption on the host depth of a moved fiber is counted on the wrong thread.

## Status (2026-11-06, the mesh stages measured): the level's scene is drawn by mesh tasks

With the start-up crash gone, the quick route (`--replay build/logs/a-spam.txt --skip-movies`, 110 s) reaches the
Deku Tree's dialogue over a black scene every time. Measured on the way:
- The NVN2 stubs now log their call count at 10, 100, 1,000... (`nvn_procs.c` NVN_STUB). In about 40 s of the level
  the game made 10,000 to 99,999 `nvnCommandBufferDrawMeshTasksIndirect` and 1,000 to 9,999
  `nvnCommandBufferDrawMeshTasks` calls (run-85), so at least about ten mesh draws a frame, against about 100
  ordinary draws.
- The frame dump of frame 2000 (dump-84) has no scene pass at all: 59 dispatches and 86 render passes, the largest
  960x540 (10 passes, 36 draws), then the 2560x1440 composition with the subtitles. The mesh draws, being stubs, do
  not reach the dump; everything the scene is made of goes through them.
- 110 distinct programs fail to lower in the level (51,824 log lines, the same programs retried): unknown opcodes
  57, scratch pressure 17, `RED address` 10, uniform datapath opcodes 8, `PLOP3` 6, and smaller groups.
- The corpus holds 12 mesh-pipeline programs, 9 mesh and 3 task, all from the agl package
  (`ApplicationPackage.Nin_Ounce_NVN2.release.sarc.zs/AglShader.sharcb`), 16 to 616 instructions: a few generic
  mesh programs feed the materials' thousands of pixel shaders. None lowers: the stages write their outputs (and the
  task stage its payload) with `ISBEWR` and read their inputs with `ISBERD`, whose immediates are undocumented. The
  smallest mesh program, 28b308906146a98b (112 instructions, workgroup 32), reads `ISBERD [0x0]`, writes
  `ISBEWR 0x3ff` and `0x30a` (likely the vertex and primitive counts), then per vertex `0x4`/`0x6`, `0x8012`/`0x8018`,
  `0x100ff`, `0x18014`, `0x20000`/`0x20008`, `0x28001`/`0x28009` (an attribute word address with what may be a
  component in bits 15-17) and per primitive `0x40d`, `0x505`, `0x607`, `0x707`, `0x805`, `0x90f`, `0x8002`, `0x3`,
  `0x1000a`, `0x1800b`. The task program 9d61d62aa9337355 (16 instructions) writes `0x1001` and `0x400`.

What the mesh stages need, in order: decode the `ISBEWR`/`ISBERD` address encoding from these programs and what the
game draws with them; lower them (`sm86_lower.c`) to an output form the translator can take; a translator patch for
SPIR-V's `MeshEXT`/`TaskEXT` execution models (per-vertex and per-primitive output arrays, `SetMeshOutputsEXT`, the
task payload, `EmitMeshTasksEXT`), which Ryujinx 1.3.2 lacks; and on the host `VK_EXT_mesh_shader` pipelines with
task and mesh stages and `vkCmdDrawMeshTasks*EXT` for the eight NVN2 entry points, still stubs. The first step is the
uncertain one.

Also fixed: `tools/shader_census.py --lower` lowered nothing when the repository's path held a space (the helper's
batch request is split at spaces); it now passes paths relative to the output folder.

## Status (2026-11-07, the mesh stages decoded): ISBEWR / ISBERD and the stage buffer layout

With PR #7 merged on top of the work below, every program the level route meets lowers (0 refusals in run-55, 804
programs offline; 173 refused before) and the frame dump shows the tree, the roots, textured rocks and rippled water.
What is left white is the horizon showing through where the ground should be: the ground is mesh-drawn, and the
eight NVN2 mesh entry points are still stubs.

The ISBE encoding, from Lichii's `NX2_NOTES.md` (row 89) and its lowering's field comments (GPL-3.0, read for
behaviour only), checked against all 12 of this game's mesh-pipeline programs (`build/v131072/shaders`, a listing
script over the 0x923 / 0x927 words; every instruction matches):
- Fields: address register 24-31, byte offset 40-63; ISBERD destination 16-23, ISBEWR data 32-39; bit 75 `.32`
  (else a byte), bits 76-77 the region (3 attributes, 0 the index map), 78 `.SKEW`, 79 `.O` (the output buffer). No
  other bits of word 2 are set anywhere.
- **Task** (3 programs, 16-32 instructions, workgroup 1 or 32): `ISBERD.ATTR.SKEW [0x0]` is the workgroup index;
  it writes the mesh workgroup count with `ISBEWR.ATTR.O [0x4]` and the payload from `[0x10]` (one or two words, or
  `[R+0x10]` per thread).
- **Mesh** (9 programs, 104-616 instructions, 32 threads): `ISBERD.ATTR [0x0]` is the mesh workgroup's index and
  `[0x10]` / `[0x14]` (or `[R+0x10]`) the task payload. Vertices: `ISBEWR.32.ATTR.SKEW.O [Rv + 0x80 c]`, one 128-byte
  line per output component c (32 vertices of 4 bytes): position at lines 0-3 (`0x0..0x180`), then two or three
  varying words (`0x200`, `0x280`, `0x300`). Index map: `ISBEWR.32.MAP.O [0x3]` the primitive count (written as 0
  first by the leader, `@!P0`, then the real count), then `ISBEWR.U8.MAP.O [Ri + 4..9]`: six byte indices per thread,
  two triangles.
- Control +0x58..+0x63 of a mesh program: workgroup 32 in the low half of the first word, and in the third the maximum
  vertices (low half: 0x20 or 0x80) and primitives (high half: 0x20 or 0x40), which fit what the programs write (one
  or four vertices per thread, two triangles per thread).

A way that needs no mesh-shader support in the translator: run both stages as **compute** programs. ISBERD /
ISBEWR become loads and stores into storage buffers indexed by the workgroup (the payload and counts of each task
workgroup, the vertex lines and index map of each mesh workgroup), which the lowering can express like the
storage-held pointers above (bank-0 words the host fills). A small expansion pass then turns the maps into an index
buffer (unused slots degenerate), and one ordinary indexed draw reads the vertex lines through a generated
pass-through vertex shader whose outputs follow the mesh program header's output map, so the materials' pixel
shaders link unchanged. Lichii instead translates to `VK_EXT_mesh_shader` (its notes, row 89); either way the host
must also learn which programs NVN2 binds for a mesh draw.

Step 1 done (2026-11-07): `sm86_lower.c` `lower_isbe` lowers task and mesh programs as compute programs, the stage
buffer accesses becoming LDG / STG on per-workgroup records whose bases the host is to put at bank 0 + 0x6c0 (mesh)
and + 0x6c8 (task); a mesh workgroup that its task did not ask for stores a primitive count of 0 and exits first
(`mesh_prologue`); the layout is in `sm86_lower.h` (`SM86_MESH_*`). All 12 programs lower and translate to SPIR-V
offline as compute (notes: none, but one untraced storage load in task 20eee5a0). A request for a task or mesh
program in a graphics slot is refused (`gfx_shader.c` `req_init`). Nothing calls them yet. The draw stubs' arguments,
from the stub log: `DrawMeshTasks(cmd, first 0, count 12)`; `DrawMeshTasksIndirect(cmd, address, 1 or 0)`.

Step 2 written, off by default (2026-11-07, `OOT_MESH=1` turns it on): `DrawMeshTasks` and `DrawMeshTasksIndirect`
are real entry points recorded as `OP_DRAW_MESH(_IND)`; the replay keeps the programs bound for stages 6 and 7 and
their uniform, storage, texture, sampler and image binds (`VkExec.mesh_prog`, `mubo`...; all were dropped before), and
`vk_exec_draw.c` `op_draw_mesh` runs a direct draw's task program (n workgroups) and mesh program (n << 4, or n
without a task) as compute dispatches on records in one device-local buffer, with the stage's resources swapped into
the compute slots (`mesh_dispatch`). Indirect draws only read their arguments back (`OOT_MESH_LOG=1`), as their layout
is not known. What the game does, measured with `OOT_MESH_LOG=1` (run-64..67): the mesh programs are 2-stage programs
(fragment + mesh); they are bound with `BindProgram` stage mask 0xdf and task/mesh 0xc0 or 0x80; the direct draws are
`DrawMeshTasks(0, 12)` and `(0, 1)` from main+0x273690, recorded on a thread that bound no program in that command
buffer (the state comes from the caller); the stages' bank-0 entries are now 16 bytes {address, size}
(`SM86_MESH_C0_*`), as the translator checks the size. **The first run with it on lost the device** (run-68, 27 s, on
the first direct mesh draws: three new programs translated, no read-back came back). Not understood yet; candidates:
a program bound for stage 6/7 that is stale for these draws, an out-of-range access in a record (no robust access),
or the task count. Next: run one mesh program alone with `--probe-cs` on made-up records (one dispatch, no game), then
the game with `OOT_MESH=1` and `OOT_MESH_LOG=1`, one run.

**The mesh program faults the GPU on its own** (2026-11-07, `--probe-cs` now replays a mesh draw: `stage 7` programs
are drawn with `mesh <count>`, `ubo` lines bind uniform blocks, `task <ctl> <code>` adds a task program). The
smallest mesh program, 28b308906146a98b, drawn alone with a uniform block saying 64 items (2 workgroups), caused a
GPU exception and reset every time (nvlddmkm events 13 and 153, fifteen runs), while the harness printed "fence
reached": **the harness does not notice a device loss; read the driver's event log after every probe run.** So the
title's device loss (run-68) is the lowered program itself, not the replay's state; the likeliest cause is an access
outside the bound record buffers (the per-workgroup addressing in `lower_isbe`, or the bank-0 record entries the
translator traces through `GlobalToStorage`). Next: read the translated SPIR-V of 28b30890 for the buffer indexing
(offline), and check it against a CPU model of the lowered program before any GPU run. Also fixed on the way:
`vkCmdUpdateBuffer` has no recorder wrapper (`vk_rec.h`), so it went into the command buffer out of order and from a
second thread; and completion callbacks in the harness run before its submission, hence `OOT_MESH_SYNC`.

**Found and fixed (2026-11-07, after a reboot):** `vk_exec_draw.c` `answers_for` read the workgroup size raw from control
+0x58 as 16-bit values: for a mesh program that is 32 x 1 x 128 = 4096 threads (the high halves are its output limits),
four times Vulkan's limit. It now takes sm86_control_parse's values. With it the probe draws 28b30890 cleanly (2 triangles
per workgroup, plausible vertices, 0 validation errors, 0 driver resets), and a full level run with OOT_MESH=1 (run-70)
has no device loss and no driver reset over 69 direct mesh draws. But every task record comes back with a mesh count
of 0, and every indirect mesh draw's arguments read 0: both likely come from the frame's culling compute passes,
whose outputs are empty. Next: find which dispatches write the indirect arguments and the task programs' inputs.

**The task programs' inputs (2026-11-07, run-72):** the level's task program is not one of the archive's three but a
culling program built at run time (c72ef95d, 1059 instructions), which reads its object list from storage buffers
through bank 0 + 0x670. A census of the programs' bank-0 operands gives each stage's storage table: vertex 0x110,
fragment 0x510 (the Maxwell layout the replay already serves), task 0x670, mesh 0x770, which nothing filled, so the
culling read zeros and asked for no meshes. mesh_c0 now fills the running stage's table from its storage binds
(SM86_TASK_TABLE, SM86_MESH_TABLE), ssbo_slot maps them, the record entries moved to 0x600 and these stages'
storage-held pointers to 0x480 (KEY_VERSION 18). Result: 58 of 69 logged draws' first task asks for 7 to 32 mesh
workgroups, and the mesh workgroups emit 30 to 99 triangles each with plausible positions; no device loss, no driver
reset. Odd, to check: some draws whose first task record reads a count of 0 still show triangles in their first
mesh records (the prologue should have zeroed them). Next is step 3, drawing the records.

**Step 3 (2026-11-07, run-73/74):** the records are drawn. sm86_lower.c mesh_vs generates a Maxwell vertex program per
mesh program (the host passes the mesh code with a copy of its control section marked SM86_MESH_VS_MAGIC, stage 0):
vertex id / 3 is the primitive, the instance the mesh workgroup; it loads the primitive count and index byte from the
record's map and the vertex's lines, and stores line k to the output attribute of the k-th set bit of the mesh
header's output map (bitfield from header byte 0x34, bit b = attribute 0x40 + 4 b; checked on all 26 mesh programs).
`vk_exec_draw.c` `mesh_draw` pairs it with the game's fragment stage of the same program and draws 3 * max-primitives
vertices x workgroups instanced, the records at vertex storage slot 15 (bank 0 + 0x200). All 26 translate; no device
loss, no driver reset. The direct mesh draws are the depth pre-pass (colour mask 0) and 128x128 shadow layers: with
them the scene's depth buffer gets the ground where it had holes (`build/logs/mesh-depth-keep.png`). The colour of
the ground comes from the indirect mesh draws, still only read back. OOT_MESH_NODRAW=1 skips step 3.

**Indirect mesh draws and an intermittent timeout (run-75/76):** the indirect draws come from about ten call sites
(main+0x2be2dbc, +0x2be2710, +0x186xxx, +0x184xxx, +0x6fb70), argument buffers such as 0x29bc62000, second argument 1 or 2;
their first 16 bytes still read 0 at the draw. Run-76 (OOT_MESH=1, nothing else new) lost the device at 86 s with 37
shader instruction pointers, the pattern of a shader that runs too long; runs 72-75 with the same code were clean
over 110-150 s. Suspect: a loop in the culling task program (c72ef95d) or a mesh program driven by a count read
from storage. Keep OOT_MESH off until that is found; next, find the timed-out shader offline (the fault addresses
against the pipelines built in that run) before any further run.

**The timeout was not the mesh path (2026-11-07, run-77..80).** OOT_CS_TIME with OOT_MESH=1 showed one ordinary kernel,
ace1cec0 (64x64 groups), at 400-670 ms a dispatch (99.9% of the GPU), against 0.008 ms without the mesh path; the
task and mesh stages took 0.001 ms. Its control section declares no storage slots: the list it loops over is a
pointer the game pushes with its constants (bank 0 + 0x60), which the replay bound to the shared zero buffer. And every
*written* storage binding without backing also got that zero buffer, so a kernel's stores could make it non-zero for
every unresolved reader: a garbage list length, a near-endless loop. Fixed: pointers in the pushed constants (before the
NVN2 tables) bind the buffer they point to (ssbo_info), and written bindings without backing go to a separate
discard buffer. Result without the mesh path (run-79): the level runs at about 50 fps instead of 1-2, the slowest
kernel 0.25 ms, and the scene renders lit and textured (`build/logs/level-push-keep.png`, Navi over the moss). With
OOT_MESH=1 (run-80): the same speed, no driver reset, the Deku Tree area lit and textured where drawn
(`build/logs/level-push-mesh-keep.png`); the white left is ground drawn by the indirect mesh draws, still not run.

**Why the indirect mesh draws read 0: the frame's culling never counts anything (2026-11-07, run-81..110).** The
chain, found with `OOT_MESH_WATCH`, `OOT_IND_LOG=<frame>`, `OOT_GW_TRACE` and a frame dump's `OOT_DUMP_BUF`:

- The indirect mesh draws' arguments sit at the head of buffers whose rest is a list (16-byte records, e.g.
  0x2a0e04010/020/030); the culling kernels that fill them (c8b8, 3076, 75dc, 68e8, 2bb4, d295...) are themselves
  *indirect* dispatches, and their group counts read `0 1 1` every frame: the CPU writes y = z = 1 and x = 0, a GPU
  counter at record + 12 is to be counted up first. Nothing writes them: the dump's buffer snapshot of the argument
  records stays `{0, 1, 1, 0}` through two whole level frames.
- **Fixed: the lowering corrupted constant bank 1.** The record-less jump-table fix-up in `sm86_lower.c` moved every
  c1 word that is 0 or a multiple of 16 as if it were a jump target, also in programs without a BRX. The culling
  kernels add the 64-bit constant 12 in c1 to their argument pointer (`ULDC.64 c[1][0]` + `c[0][0xc0]`); its zero high
  word became 8, so the counters went 32 GiB past the buffer, and the translator traced the store to c1 + 0 (no
  address), into the discard buffer. Now only programs with a BRX get the fix-up (`has_brx`), and a bank-1 word of a
  program without one is forwarded as an immediate (`c1_imm`), so the translator traces such addresses to the bank-0
  pointer. All 827 of 828 cached programs still lower (the one refusal predates this). KEY_VERSION 19.
- **The counters are atomics through pointers no trace reaches.** The frame's first kernel, e6d5432f (169 x 128
  threads, one warp per object), reads an object's type record address from an array in its uniform block 1 at a
  run-time index (`LDC.64 c[4][R]`), a list pointer from the record, the count at that pointer + 4, and adds the
  visible objects with `ATOMG.ADD` at `c4[0xe0 + 16 type] + 12`, the culling dispatches' argument records. The
  translator found no storage buffer for 49 of its accesses (35 loads, 5 stores, 6 atomic adds, 3 atomic max) and
  emitted nothing for them. **Added, off by default (`OOT_GW=1`): a global window** (`tools/xlate/patches/0019`,
  `vk_exec_draw.c` `gw_select`): such an access in a compute program is checked on the GPU against 24
  {address, size} entries at bank 0 + 0x410 (`VK_GW_C0`), each bound as a storage buffer; before each dispatch whose
  program has them, the host fills them with the buffers the program's pointers can reach: 64-bit values in its read
  storage buffers, uniform blocks (and this submission's inline updates) and push, looked up among the buffers bound
  as compute storage so far (a registry of the bound ranges) or, failing that, a backing of at most 256 MiB; then one
  hop further (256 bytes at each pointer found), then its own buffers. Windows are bound without the mirror's
  write-back (whole ranges came back every frame). Its cost: 3 ms of CPU a frame on the level (scans cached per
  buffer for 32 frames; scanning per dispatch cost 28 ms, a poor hash made every lookup miss). With it, e6d5432f's
  window holds every record, list and argument buffer of the chain (`OOT_GW_TRACE=e6d5432f` checks each hop), the
  bindings are right (buffer, offset, range logged), the SPIR-V's range check is right (spirv-dis: bank 0 vec4 65 on),
  and the GPU's copies of the lists hold the CPU's counts (2364 at 0x2a1bb0000 + 4)...
- ...**and still no thread reaches the atomics**: the argument records never change. What is left is inside e6d5432f
  after the count check: a warp vote and shared-memory step (`VOTE`, `LDS [0x400]`, `@!P0 EXIT` at 0x6a10) guards the
  atomics. Next: a probe that can store into a buffer (the `OOT_SM86_PROBE` one needs a storage image, which this
  kernel lacks) to bisect e6d5432f between 0x430 (the count check) and 0x6bc0 (the first atomic), then the culling
  kernels' own counters, then `op_draw_mesh`'s indirect path (16-byte records; the argument word layout is still to
  be read from the culling's writes).
- With `OOT_GW=1` the intro cutscene looked worse (more of it white, the tree in silhouette, `build/logs/gw-11.png`
  against `default-after-gw.png`): some kernels whose pointer reads become real get worse, another reason it stays
  off. No device loss and no driver reset in the 30 runs of this work.

**The culling counts (2026-11-08, run-117..126): two lowering/translation bugs, found with a counting probe.**

- **New diagnostic `OOT_SM86_COUNT=<code hash>:<offset>[=rN|=pN],...`** (`sm86_lower.c` `count_emit`,
  `vk_exec_draw.c` `cnt_log`): before each listed instruction of one compute program, every arriving thread adds 1
  (or register N, or 1 << N when predicate N is set) into a host-visible counter buffer bound through bank 0 +
  `SM86_COUNT_C0` (0x5f0); the host logs the sums every 120 frames (`count: f..: @6a20:0 ...`). A probed program gets
  its own cache key (`sm86_probe_tag`, also for `OOT_SM86_PROBE`), so neither the cache nor a fresh cache is needed.
  One run gives a whole profile of where threads go; with registers and predicates it reads values too.
- **PLOP3 lowered with the PSETP operands swapped (fixed, KEY_VERSION 20).** e6d5432f's threads all reached 0x69a0
  and none 0x6a20: the `@!P0 EXIT` there takes P0 from `PLOP3 P0 = c & !(a & b)` (c = PT), lowered to Maxwell PSETP.
  The translator computes PSETP as (P@12 op P@29) op2 P@39; the lowering put its first operand at 39 and its third at
  12, so every table not symmetric in them came out wrong (here a ^ b: with a = b = 0 every thread left). 231 of the
  902 cached programs lower differently now (fragment shaders, the culling kernels, the fog kernel 1a62084a, 374126bf,
  358beb27). After it: e6d5432f's atomics run, its dispatches' group counts are no longer 0 (`OOT_IND_LOG`), and the
  intro looks cleaner without the window too (`build/logs/psetp-default.png`).
- **RED's offset read unsigned by the translator (fixed, translator patch 0020).** The second culling level still read
  0 groups with a counter of 139 (e6c023a5 for 032f18b8): a count probe showed the right group count reaching its
  `RED.E.MAX [UR4 - 12]` (the count word next to the counter the pointer names), but Ryujinx's `Red` passes the 20-bit
  offset without sign extension (its `Atom` extends it), so the max went 1 MiB past. After it every culling level runs
  with its counts (032f18b8 2 groups, 3360a3c0 38) and the indirect mesh draws' arguments hold counts.
- **The indirect mesh draws' arguments**: 16 bytes, word 0 the count (task workgroups with a task program, mesh
  workgroups without), words 1-2 zero, word 3 the culling's counter (`{0xc 0 0 0xc}`, `{0x1 0 0 0x4}`). `op_draw_mesh`
  now runs them behind `OOT_MESH_IND=1` (default off): a bound of `OOT_MESH_IND_MAX` (64) task workgroups, the count
  copied on the GPU into bank 0 + `SM86_MESH_C0_COUNT` where a new task prologue (`task_prologue`) exits the
  workgroups past it; a mesh-only draw gets the count as its stand-in task's mesh count. KEY_VERSION 21.
- **Device loss (run-126, one driver reset):** the first run with `OOT_GW=1 OOT_MESH=1` and the indirect path (then
  not behind a knob), 38 s in, on the title, during the full re-translation of KEY_VERSION 21; fault address
  0x2003bed40, no checkpoint reached; `build/logs/fault-run126.bin`. Not isolated: the indirect path (a bound of task
  workgroups whose count copy did not happen would read past their lists), the task prologue, or the re-translation
  load. The indirect path is now off by default and refuses to run without the count copy. Next, one change per run:
  `OOT_MESH=1` alone (prologue, translator), then `OOT_GW=1`, then `OOT_MESH_IND=1`.
- **Bisected after a reboot (run-127..130):** defaults (re-translating KEY 21), `OOT_MESH=1`, and
  `OOT_MESH=1 OOT_GW=1` all ran clean (59, 58, 56 fps in the level; no reset). Adding `OOT_MESH_IND=1` (bound 16)
  lost the device again, 30 s in on the title, within 2 s of the first indirect mesh draws (fault IP 0x2003b7e70,
  `build/logs/fault-run130.bin`). So it is the indirect path or a task/mesh program only the indirect draws use,
  running for the first time. The programs that translated in that run were not saved before the crash. Added
  `meshind:` log lines (each new task/mesh pair, before it is recorded), `OOT_MESH_IND_FROM=<frame>` and
  `OOT_MESH_IND_SKIP=<hash prefixes>` to name and then exclude the culprit.
- **run-131** (after a reboot; `OOT_MESH_IND_FROM=4500`, bound 16): the level's first frame of indirect draws lost
  the device too (third reset; fault IP 0x2009d6030, `build/logs/fault-run131.bin`). 29 task/mesh pairs were
  recorded in that frame (the `meshind:` lines: 2 task programs, 1ddd3643 and c7f0444a, and 27 mesh programs, 9 of
  them translated for the first time in that frame). Read offline: the mesh programs have no loops (only the
  trailing self-branch); the task program 1ddd3643 (a01eebd2) loops over a list but every loop is bounded by clamped
  counts (0x100 >> 5, 0x11, 0x1f) and its exit is a predicated CALL lowered as a branch. Nothing found offline yet.
  Added `OOT_MESH_IND_ONLY=<hash prefixes>` to run a subset (bisection costs one reset per bad half).
- **run-132** (bisection, after a reboot): only the 4 indirect draws with a task program (1ddd3643, c7f0444a,
  bbcf8a53) lost the device (fourth reset, `build/logs/fault-run132.bin`). Read offline: those task programs write
  their payload at 0x18, 0x418 and 0x818 plus 4 * a lane index (to about 0x898), the direct draws' two task programs
  only to 0x10; the task record stride was 0x100, so tasks overwrote each other's payloads (meshlet lists) and the mesh
  programs ran on garbage. `SM86_TASK_STRIDE` is now 0x1000 (KEY_VERSION 22). run-133: still lost the device (fifth reset), so not the (only) cause.
  The count probe now works in task and mesh programs too and logs its counters at a device loss (`count: at the
  device loss`, from host-coherent memory), to show where a hanging program's threads are.
- **Found (run-134, run-135):** task program 1ddd3643 alone ran clean (but every thread was culled before its loop);
  c7f0444a alone hung, and the count probe's dump at the device loss showed why: its meshlet list words read 0
  (R9 = R6 = 0), so its loop bound `(min(R9 - R6, 256) - 1) >> 5` became 0x07ffffff and one warp looped 361 million
  times. Those reads are 8 loads the translator sends through the global window, which the host never filled for task
  and mesh stages. Now `gw_select_mesh` fills it for them (`OOT_GW=1`; not when the stage has storage-held pointers,
  whose table at 0x480 overlaps the window).
- **Working (run-136, run-137, after a reboot):** c7f0444a alone: its list words are real, its loop runs a few times
  per thread, every thread reaches its exit, no reset. All indirect draws (`OOT_MESH=1 OOT_GW=1 OOT_MESH_IND=1`,
  bound 16, from frame 4500): no reset, and the ground under Link in Kokiri Forest is drawn coloured and textured for
  the first time (`build/logs/ground-all.png`; too bright and speckled, much of the scene still white). Level fps
  44 against 56 without them. Left: the colour itself (brightness, speckle), the 12 programs that do not lower, the
  title-screen draws (`OOT_MESH_IND_FROM` still set), the bound (16 vs real counts), then default-on.
- **What the white is (2026-11-08, frame dumps at 5500 and 7000, run-138..145).** Not the mesh path: the default run is
  white at frame 7000 too (`dump-141`; the clean default screenshots were taken later, in a dark spot). The frame
  fills its HDR target with the sky first (one full-screen draw, 24..34), then lights the geometry over it; the lit
  scene (`dump-143/pass_0172..._tm.png`) shows the sky wherever no geometry was drawn: the ground, tree trunks, at
  times Link. Lit sky is far above 1, so the tone-mapped picture is white there. So the white is undrawn geometry:
  the indirect mesh draws (off by default, bounded, 12 programs not lowered) and whatever else is missing.
- **Fixed on the way: R32 storage views.** The frame's anti-aliasing kernel (SPIR-V hash 1cb856c3) packs R11G11B10
  itself and stores the word through an R32UI view (`view-nvn-fmt 0xb`) of the R11G11B10F scene target; the host made
  every storage view in the image's own format, so each word was converted as a float (the 65024 maxima). Now an R32
  game view of another 32-bit colour format is an R32_SFLOAT view (the translator declares such images float with
  no format) or R32UI/R32I (`get_view`); 32-bit colour formats without storage support (RGB9E5 here) get STORAGE
  through such views (`storage_r32`, EXTENDED_USAGE); and when the slot's texture is another descriptor over the
  same bytes (RGB9E5 over the R11G11B10F target, seen at times), the store goes to the live GPU-written image there
  (`vk_texture_r32_sibling_locked`). No visible change in the white (it is upstream), no regression (59.7 fps).
- **Game crash with `OOT_GW=1` (3 of about 12 runs: run-119, run-134, run-145):** main+0xc938d0 reads +0x150 of a
  null table entry; the index comes from an object record at 0x29bc10a88, in the culling buffers the window lets the
  GPU write. Likely GPU-written bytes reaching what the CPU reads (write-back of a range a window entry also
  covers, or a kernel's wrong store). Open; the window stays off by default.
- **The `OOT_GW=1` game crash, probably (run-146..153):** a range watch (`OOT_WATCH_RANGE=<lo>:<hi>[,...]`, every
  written binding, window entry, copy and fill over the ranges) found no GPU write to the object records nor to the
  game's structures in 0x1a0000000..0x1b0000000, so the bad index comes from GPU results the game reads back. It did
  show the two jump-table kernels with known wrong traces (374126bf, 358beb27: stores traced to float words of bank 3)
  getting window entries over the culling arguments; run-119's crash came right after 358beb27's first window. They
  now get no window (`OOT_GW_SKIP`, default those two). Since: 5 runs with the window, 3 of them with every ground
  draw, no crash (before: 3 of 12; 5 clean by chance about 1 in 4). `build/logs/gwskip-ground.png`: the path,
  ground, rocks and Link drawn, dark; the sky still shows where parts are missing.
- 12 task/mesh programs used only by the indirect draws do not lower (11 mesh, 1 task: register pressure at the
  start, "no general register dead over a uniform register's span" and similar, and two "uniform datapath opcode");
  the same with the lowering before the task prologue. Their draws are skipped, so even with the crash fixed, part
  of the ground will stay white until the lowering handles them.
- Also seen once (run-119, `OOT_GW=1` with a count probe, during a re-translation): the game itself crashed on a null
  pointer (main+0xc938d0 reads +0x150 of an entry its table lacks, index from an object record at 0x29bc01638); not
  seen again in the three `OOT_GW=1` runs after.

Left, in order: (2) the host dispatch: which programs a mesh draw uses (the bound program's stages 6 and 7), the
workgroup size from control +0x58 (packed for these stages: the low half), the record buffers and the bank-0 words,
the task count read back or bounded (mesh grid = tasks << SHIFT); (3) a generated pass-through vertex program that
reads the records (vertex = mesh record of `index / (3 max-primitives)`, index map byte, attribute lines) and writes
the outputs the mesh program header's map names, so the materials' pixel shaders link unchanged; a primitive past
the count gives a degenerate triangle; (4) the draw itself, non-indexed, `workgroups * 3 * max-primitives` vertices.

## Status (2026-11-07): pointers held in storage buffers; the fog's volume still NaN

A contributor's run on native Windows (RTX 5070 Ti, driver 621.01, no WSL), from `d84bf5d`, on the fog of
2026-11-06 (kernel 1a62084a writes only 0 and NaN):

- **Storage-held pointers (fixed).** The translator logged "Failed to find storage buffer for global memory operation
  Load" 208 times for 1a62084a (`tools/kernel.py xlate`, the `.res.json` notes). The kernel loads five 64-bit
  pointers from the 0x40-byte table at bank 0 + 0x60 (NVN storage slot 2, `LDG.E.64 [UR10+0..0x20]` after
  `ULDC.64 UR10, c[0][0x60]`) and reads its light and froxel data through them; GlobalToStorage cannot trace a pointer
  that comes from memory, so every such load read zero. The lowering now turns a 64-bit load of that shape, whose
  pair is used as an address next, into a read of bank 0 + `SM86_PTR_C0` (0x700) + 8 i and appends the table of
  (descriptor, offset) pairs to the lowered c1 (`sm86_lower.c` `ptr_load`, `sm86_lower.h`); the host fills those
  words from guest memory at each dispatch and binds the buffer behind each pointer (`vk_exec_draw.c` "storage-held
  pointers"). The translator then traces all of them: 0 notes for 1a62084a. Nine kernels of the level route are
  affected (bank 0 + 0x60, 0x100, 0x30). `KEY_VERSION` 15; `OOT_SM86_PTR=0` turns it off. A first version that
  rewrote every such load, whatever its use, lost the device on the level. Measured with the fog skipped
  (`OOT_SKIP_CS=1a62084a`, the level route, screenshots at the end): blown-out white areas 20.3% of the frame before,
  4.1% after; both score as pictures (`shot_diff.py` noise 0.28, 0.36).
- **Depth maps read as colour (fixed, no visible change yet).** The game renders a 4096x4096 D32F depth map and
  samples the same storage as an R32F texture (1a62084a's texture at bank 0 + 0x78, and fragment shaders of the
  frame). The two descriptors were two images and the R32F one was never written. At the end of each pass the depth
  target's texels are now copied into live R32F / R16 textures over its storage, and a new one takes them when created
  (`vk_texture.c` `depth_colour_pair`, `vk_texture_depth_rendered`; `OOT_DEPTH_COLOUR=0` turns it off).
- **The fog volume is still 0 and NaN** with both (frame dump, 64x36x96 volume, about 60% of the voxels NaN), so
  there is another cause. `OOT_SM86_PROBE` (`sm86_lower.c`) stores, for one register at one instruction of one
  program, NaN / positive / negative flags per thread into the volume. Results so far: the first channel stored at
  0x7290 (R12) is 0 or positive, never NaN or negative; R90 after 0x5300 (a weight product) is positive in 73% of
  the voxels and 0 in the rest, never NaN. (An earlier probe of R23 read "0 everywhere", but that version wrote the
  same value for 0 and for "not reached": discard it. The probe now writes a state code, 0 meaning not reached.) So
  the NaN is not an all-zero weight; where it comes from is open. The
  kernel uses about ten instruction forms that no other cached program uses (FSETP, ISETP, PLOP3, I2FP and FFMA
  variants, `forms` from the census), the likeliest place for a lowering defect.
- **Hazards met.** A probe run must translate the probed program alone (a copy of the warm shader cache with its
  translation removed): an empty cache translated everything in the background and lost the device on the title
  screen. Flags of 1.0 in the volume made the next kernel (c82d3cba) loop until the GPU timed out; skipping c82d3cba
  also loses the device. Eight such losses in an hour took Windows down once (bugcheck 0x10E). The start-up crash came about one run in three on this machine before the fiber fix above; `tools/run.py` works unchanged under native Windows Python.

## Where the port stands

`oot.exe --seconds 15` runs the whole start-up of OoT 1.1.1 without a crash: 82 threads, 10,988 IPC messages, 1,485
file opens, saves mounted, NPLN offline, the audio renderer at 48 kHz, 3,661 shader programs set up through NVN2's
program builder, three frames presented at 2560x1440 (`build/logs/run-45.log`). Then it stalls in the third frame
(51.7 s) because no shader translates:

```
[host] gfx_shader: translation failed (-): bad control section
[host] vk: a stage 0 shader of program 4136 failed to translate: draws with it are skipped
```

Draws with failed programs are skipped, the game waits on GPU work that never completes, nothing is drawn. Everything
below the graphics API is believed to work; every remaining known problem is in this one area or is minor (see "After
the shaders").

## Why every shader fails

The host's shader path (`runtime/gfx_shader.c`, `runtime/vk_exec_draw.c`, `tools/xlate/`) is TotK's, for Switch 1:

- `gfx_shader.c` `req_init()` reads the c1 offset at `control+0x700` and the code length at `control+0x704`, and
  `CONTROL_SIZE` is `0x880`: the cache key, the `.in` files, `build_code_hash()` and the async job copies all take
  0x880 bytes from the control pointer. `vk_dump.c:585` reads `+0x700` too.
- The NVN2 control section is 224, 248 or 280 bytes (`0xe0`, `0xf8`, `0x118`). Reading `+0x700` reads whatever lies
  after it, so the length check fails and every program is rejected before translation.
- The code is Ampere SM86 (the Switch 2's T239 GPU): 128-bit instructions, opcode in the low 12 bits of word 0, no
  Maxwell scheduling words. The vendored translator (`tools/vendor/ryujinx`, Ryujinx 1.3.2, MIT) decodes Maxwell and
  Pascal 64-bit forms only (`src/Ryujinx.Graphics.Shader/Decoders/InstDecoders.cs`).
- The translator itself is not built here yet: the log says `translator identity 0000000000000000 (none: no
  translator)`. `.NET 10.0.403` is installed; `python tools/build_xlate.py` (Windows venv) builds
  `build/xlate/oot_xlate.exe`. MP4's `mp4_xlate.exe` next door is the same source and would do for a first smoke test.

So there are two separate jobs: the control-section layout (small, data-driven) and an SM86 front end (large).

## What is measured about the NVN2 control section

From the decompressed `Shader/Model.Product.110.product.Nin_Ounce_NVN2.bfsha` (320 MB; zstd with the `zs.zsdic`
dictionary from `Pack/ZsDic.pack.zs`): 8,517 blocks found by the magic `0x98761234`, 20,000 of them surveyed by
per-dword variance. The first 0xe0 bytes are the fixed part; the block's own size sits at `+0x30` and is repeated at
`+0x98` and `+0xd4`; the tail after `+0xe0` is 0, 0x18 or 0x38 bytes of extra records.

| Offset | Seen | Reading |
|---|---|---|
| `+0x00` | `0x98761234` | magic, as on Switch 1 |
| `+0x04`, `+0x08`, `+0x0c` | `0xc9`, `0x14`, `0xe0170` in every block | constants (version words) |
| `+0x18`, `+0x1c` | `0x220014`, `0x288040` in every block | constants (SDK 22.x / NVN2 version?) |
| `+0x20..+0x27` | unique per block | a 64-bit hash |
| `+0x28..+0x2f` | 7,832 distinct over 8,517 | a second 64-bit hash, shared by a few blocks |
| `+0x30` | `0xf8` (5,900), `0xe0` (2,177), `0x118` (315) | **control section size**; repeated at `+0x98`, `+0xd4` |
| `+0x38` | 402 values, `0x6e80..0x7000` typical | code size candidate (multiple of 0x80) |
| `+0x3c` | `0x14` (5,900), `0` (1,906), `0x20` (394) | tracks `+0x30`; `+0xdc` has the same classes |
| `+0x40`, `+0x44` | rise with `+0x38` (`0x9e80, 0x9f00, 0xa000`) | sizes with headers, or offsets of c1 and end |
| `+0x4c` | 150 values, `0x5a..0x60` common | register count or similar small per-shader number |
| `+0x50..+0x60` | small flags, `0x10000` common | stage/feature flags |
| `+0x64..+0x70` | `0x3f800000` in a few blocks | floats (1.0f) |
| `+0xa4` | `0x1000001` or `0x1000101` | flags |
| `+0xbc`, `+0xc4`, `+0xc8`, `+0xd8` | `0xe0` always | size of the fixed part |

The survey script is a few lines: find each magic, measure to the next, tabulate each dword. Not yet known and needed
first: which fields are the **code length**, the **c1 offset** (the constant bank the translator needs for `BRX`
jump tables) and the **stage**, and whether an early-fragment-tests flag exists (Switch 1: `control[0x718] & 1`).
Two ways to pin them, both cheap:

1. The archive's own tables. BNSH's shader code section records per program the code and control offsets and sizes
   (TotK's `refs/totk-pc-recomp/tools/shader_census.py`, from MacNativeGames' `prepare_shader_seed.py`, MIT, parses
   Switch 1 BNSH; the container is unchanged here). Match those sizes against `+0x38`/`+0x40`/`+0x44`.
2. The game at run time. `prog_set_shaders()` (`nvn_api_obj.c`) logs `data0`; add a once-only dump of the 0x120 bytes
   behind `control` and the size of the pool mapping behind `data`, and compare with the bytes that follow the code
   (the program's own end is where the code size says).

Note the game hands over the Switch 1 `NVNshaderData` form, `{data, control}` of 16 bytes
(`nvnProgramBuilderSetShaders`, built on the stack at `main+0x14e2468`), so the host side needs no ABI change.

## What is known about the code

- Every stage ends in a 128-bit `EXIT`, opcode `0x94d`; none has Maxwell's. Instruction words are 16 bytes, the opcode
  is `word0 & 0xfff` (Lichii's `instruction.h` uses exactly this `Bits(0, 12)`).
- Phase 0 measured a 0x60-byte program header before the instructions, against Maxwell's 0x50-byte SPH. Lichii's
  environment reads `sizeof(sph)` (0x50) from the program base for Switch 2 titles too, so settle the header size on
  the corpus before anything else: the first instruction is where a plausible opcode stream begins.
- The translator job format (`tools/xlate/Program.cs`) is "graphics: 0x50-byte SPH + instructions, compute:
  instructions"; `gfx_shader.c` skips `CODE_SKIP 0x30` of NVN code-block header before that for graphics stages.
  Both constants must be checked against the NVN2 layout.

## Approach: lower SM86 to Maxwell in front of the existing translator

Recommended. Lichii (`src/shader_recompiler/frontend/ampere/lower_to_maxwell.h`, GPL-3.0, read for behaviour only) runs
four Switch 2 titles this way: it does not reinterpret the bytes, it emits new Maxwell instructions for each SM86 one,
inserts the scheduling word every three instructions, maps the uniform registers (`UR0..`) onto high general registers
(R216 and up), keeps an address map so branch targets and `BRX` tables survive the re-layout, and hands the result to
the unchanged yuzu Maxwell front end. The alternative, a native SM86 decoder inside Ryujinx's translator, would mean
new decoder structs and emitter signatures for every instruction (Ryujinx's emitters are bound to the 64-bit field
layout), many times the work for the same output.

Ours would be a C module with no dependency on the translator, usable both in `runtime/gfx_shader.c` (lower, then
request the translation of the Maxwell code; cache keys stay over the original bytes) and in an offline tool over the
whole corpus. Sources of truth that may be copied, with a header and a `THIRD_PARTY_NOTICES.md` row:

- Mesa's NAK compiler, `src/nouveau/compiler/nak/sm70_encode.rs` (MIT): the Volta/Turing/Ampere encoding, field by
  field, and the reference Lichii itself names for its encoding knowledge.
- Ryujinx's `Decoders/InstDecoders.cs` and `InstTable.cs` (MIT, already vendored): the Maxwell target encoding.

From Lichii, only behaviour: which SM86 forms Nintendo's compiler emits for these engines, the pitfalls it records in
`NX2_NOTES.md` (row 54: widen until the whole dump lowers; row 93: save and restore P6 around uniform guards; row 121:
translation time), and its test cases under `src/tests/video_core/ampere_*.cpp` as a checklist of forms (uniform
datapath `R2UR`/`UP2UR`/`ULDC`, `LDSM`, `IPA` modes, formatted surfaces, `CALL.REL.NOINC`, 64-bit `UIADD3` carry,
`TMML`, attribute `0x5c`).

## Order of work

1. **Build the translator** (`python tools/build_xlate.py` from the Windows venv), so a lowered shader can be checked
   at once. Confirm with MP4's census shaders if needed.
2. **Corpus tool**: port `refs/totk-pc-recomp/tools/shader_census.py` to the NVN2 archives (`Model.bfsha`, the
   `Layout2Shader`, `ApplicationPackage` sarc, `.vsetb`, fonts; `tools/shader_inventory.py` is MP4's Retro-format
   tool and is replaced, not extended). Output under `build/v131072/shaders/`: control and code per program, plus an
   opcode histogram (`word0 & 0xfff`) over all code. The histogram is the work list for step 4.
3. **Control section** in `gfx_shader.c`: one `ctl_parse()` giving size, code length, c1 offset, stage and flags, used
   by `req_init`, `build_code_hash`, the `.in` writer/reader, the async job copies and `vk_dump.c`; `CONTROL_SIZE`
   becomes the block's own size. Bump `KEY_VERSION`/`IN_VERSION` (the formats change). The `--probe-gfx`-style census
   self-test in `vk_pipeline.c` (`build/v121/shaders`) and `gfx_shader_probe.c` need the new corpus path.
4. **SM86 lowering**, corpus-driven: implement by opcode frequency, with the offline tool reporting the share of
   programs that lower and translate; stop at every unknown form rather than guessing. Hard parts in order: the
   address map (sched words every three instructions change every branch target; `BRX` tables live in c1), the
   uniform datapath, predicates and the P6 guard, then texture and surface forms.
5. **Run the game** with `OOT_TRACE=gfx`. Expect the first pixels, then new failures: NVN2 procs still stubbed
   (`nvn_procs_stub.inc` logs the first three calls of each; 160 return 0, only three were reached so far), state
   objects whose NVN2 layout differs, and whatever frame 3 was waiting for.

## After the shaders

Known and parked, none blocking: the C++ unwinder reads past `subsdk0` into the gap before `sdk` (`QueryMemory`
should report it free); `--probe-ipc`'s sdk heap constructor address is MP4's (`ipc.c` `ctors_143`); the
`--probe-services` override list; installer tables (`installer/layout.py`, `tools/make_release.py`,
`tools/prepare_shaders.py`); `tests/diff/func.py` and `tests/jumptables_check.py` not run; MP4's stale `fnprof_hit`
stub and `"\mp4"` escape to report upstream.

## Working facts

- Build: `/mnt/c/dev/metroid-prime-4-recomp/.venv-win/Scripts/python.exe tools/build_game.py` (257 s), output
  `build/v131072/oot.exe`. Lift: `recomp/gen.py --exefs build/extracted/v131072/exefs --out generated/v131072
  --modules main,subsdk0,sdk`.
- Run, always with WSLENV or every `OOT_*` variable is silently dropped:
  `export WSLENV=OOT_TRACE:OOT_VOLUME:OOT_APP_POOL:OOT_CRASH_PEEK:OOT_BREAK_AT:OOT_BREAK_BRIEF:OOT_BREAK_WATCH;
  OOT_VOLUME=0 OOT_TRACE=gfx build/v131072/oot.exe --seconds 15 > build/logs/run-N.log 2>&1`.
- Debug: `OOT_BREAK_AT=main+0x...`, `OOT_BREAK_WATCH`, `OOT_CRASH_PEEK`, `build/logs/nvn-proc-got.json` (GOT slot to
  proc name), `OOT_TRACE=svc,host,ipc,fs,gfx`.
- Extraction: `python3.14 tools/extract_nup.py --verify` (needs `toolchain/zbic/nso_unzbic` from `tools/build_zbic.py`).
- Lichii is at `/mnt/c/dev/Lichii` (GPL-3.0): `NX2_NOTES.md`, `src/shader_recompiler/frontend/ampere/`,
  `src/video_core/shader_environment.cpp`. Behaviour only, never code.
- Nothing from the game is committed; everything extracted or decompressed stays under `build/` (the decompressed
  bfsha of this session lived in the scratchpad and is gone; regenerate it from the RomFS image).

## Status (2026-11-06, lowering refusals of the first level)

A play session to the first level logged 192 refused draws, which were 33 distinct programs (`.fail.ctl` / `.fail.code`
files in the shader cache; `tools/sm86_dis.py` reads the code). Seven of them are not in the RomFS corpus the census
reads (a 3D fragment shader, two fragment shaders with IPA 0x526, four compute programs). Changes, each checked by
lowering, translating and `spirv-val` offline (no game run):

- TEX with bit 60 clear (128 of the 192 draws, one program): Ra holds the coordinates, Rb the LOD then the packed
  offsets (x, y, z in nibbles), as for TEX.B. Seen only on 3D fetches with LOD and offsets; other combinations are
  assumed to follow the same rule.
- IPA 0x526: Ra is a uniform register (UR4, always 0 there), ignored as the offset-free modes ignore Ra.
- The skinning vertex shaders (660 corpus programs: up to 247 live registers, no run of four free): a wide LDG with a
  uniform base keeps the base in its own destination, IMAD.WIDE writes Rd, Rd+1 directly or reuses the multiplier's
  scratch. Checked by interpreting old and new lowerings of random LDG / IMAD.WIDE sequences over symbolic registers
  (0 differences in 4000 programs; a deliberate bug is caught).
- RED with a forwarded uniform pair aligns its scratch pair; PLOP3 tables are matched only on the rows consistent with
  PT / !PT inputs; STG of wide zeros stores from zeroed scratch.
- Corpus: 14,306 of 14,363 lower (was 13,569 after the previous commit), every newly lowered vs/ps/cs program
  translates and validates. Left: mesh and task stages (12), UPLOP3 (13), PLOP3 reading a swapped predicate (10), the
  uniform-predicate budget of 10 vertex shaders, FCHK, IABS with an immediate, one TLD.

Update (2026-11-06, the Great Deku Tree scene): a real session (build/logs/run-51.log) had 24 distinct programs refused,
all compute kernels but one vertex shader. All 24 now lower, translate and pass `spirv-val` offline. What did it:
UPLOP3 (through the PLOP3 lowering); a PLOP3 whose real input is the predicate a uniform one is swapped into (copied to a
free predicate first); PLOP3 tables in any input order and through PSETP's second destination; a retry with the uniform
predicates swapped through a register when the free predicates run out; opcode 0x243 (F2IP.U8.F32, clamp and pack two
floats into bytes; the rounding numbering is an assumption), IABS from a constant, LEA with a subtracted addend, TLD with
texel offsets (2D: x and y in Ra, Rb the LOD then the offsets; seen on 40 fetches), and SURED (0x59f). The register that
holds the uniform predicates could not be register 0 before. Corpus: 14,332 of 14,363 lower; left are the mesh and
task stages (12), six compute programs stopped by one more opcode, ten vertex shaders short of a register for the
uniform predicates and three short of a uniform register pair.
