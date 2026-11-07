// Written for ocarina-of-time-recomp (MIT, Copyright (c) 2026 Jeff Aigner).
// SM86 (Ampere, the Switch 2's T239) shader programs lowered to Maxwell (SM5x) programs the vendored Ryujinx
// translator decodes. See sm86_lower.c for the layout of both and the provenance of the encodings.
#pragma once
#include <stdint.h>

typedef struct {
    uint8_t *code;             // Maxwell program: graphics: the 0x50-byte program header, then instructions in
                               // groups of three behind a (zero) scheduling word; compute: the groups alone
    uint32_t code_size;
    uint8_t *c1;               // the jump-table constants (constant bank 1) with their targets moved to the
                               // lowered program; the translator reads BRX targets from here
    uint32_t c1_len;
    uint32_t error_offset;     // byte offset of the SM86 instruction that could not be lowered, from the start
                               // of the code block (header included)
    char error[96];            // empty when the lowering succeeded
} Sm86Lowered;

// code and control are the NVN2 program as the game hands them to nvnProgramSetShaders (and as the archives hold
// them): control is the NVN2 control section (magic 0x98761234; its fields are read here), code the code block
// the control describes (graphics: 0x80-byte header, instructions, padding, the jump tables). Returns 1 and fills
// out on success; 0 with out->error set otherwise. Free with sm86_lower_free either way.
int sm86_lower(const uint8_t *code, uint32_t code_size, const uint8_t *control, uint32_t control_size, Sm86Lowered *out);
// Set by the host before any lowering (gfx_shader_set_tensor): 1 lowers an unconditional HMMA.16816.F16 of a compute
// program whose workgroup is whole subgroups to the translator's MMA marker (tools/xlate patch 0016), which becomes a
// cooperative matrix multiply on the tensor cores; 0 emulates it with shuffles and half FMAs.
extern int sm86_lower_tensor_cores;
// sm86_lower with the tensor-core choice made by the caller (tensor 0 or 1 instead of sm86_lower_tensor_cores).
int sm86_lower_ex(const uint8_t *code, uint32_t code_size, const uint8_t *control, uint32_t control_size, Sm86Lowered *out,
                  int tensor);
void sm86_lower_free(Sm86Lowered *out);

// Pointers held in storage buffers (sm86_lower.c lower_ldg). A compute program that loads a 64-bit pointer from a
// storage buffer whose address is a bank-0 constant (c0[desc]), at a constant offset, and then loads through it,
// cannot have the second load traced to a bound buffer by the translator (it read zero). Such a pointer load reads
// c0[SM86_PTR_C0 + 8 i] instead; the host fills those words with the pointers from guest memory at each dispatch and
// binds the buffer each points into (vk_exec_draw.c, "storage-held pointers"). The lowered c1 ends with the table:
// {u32 desc, u32 offset} for i = 0..count-1, then u32 count, u32 SM86_PTR_MAGIC. OOT_SM86_PTR=0: off.
#define SM86_PTR_C0 0x700u
#define SM86_PTR_MAX 32u
#define SM86_PTR_MAGIC 0x52545050u   // "PPTR"

// Task and mesh programs lowered as compute programs (sm86_lower.c lower_isbe; docs/phase3.md, "the mesh stages
// decoded"). Their stage buffer accesses become loads and stores into per-workgroup records in two storage buffers
// whose addresses the host puts in bank 0: task record t at TASK base + t * SM86_TASK_STRIDE (the mesh workgroup count
// at +4, the payload from +0x10), mesh record g at MESH base + g * SM86_MESH_STRIDE (the attribute lines from 0, the
// index map from SM86_MESH_MAP: the primitive count in its first word, byte indices from +4). A mesh workgroup g of a
// task draw is mesh i = g & MASK of task t = g >> SHIFT; one whose i is not below its task's count stores a primitive
// count of 0 and exits at once. Without a task program the host gives SHIFT 31, MASK ~0 and a task record with count
// ~0.
#define SM86_MESH_C0_MESH  0x600u   // {u64 address, u32 size} of the mesh records, as a storage table entry
#define SM86_MESH_C0_TASK  0x610u   // {u64 address, u32 size} of the task records
#define SM86_MESH_C0_MASK  0x620u   // u32
#define SM86_MESH_C0_SHIFT 0x624u   // u32
#define SM86_MESH_C0_COUNT 0x628u   // u32: a task workgroup at or past it exits at once (sm86_lower.c task_prologue); an
                                    // indirect draw's count, copied there on the GPU, under a dispatch the host bounds
#define SM86_MESH_C0_END   0x630u
// The stages' own storage buffer tables in bank 0 ({u64 address, u32 size} per slot), as the game's programs read them
// (a census of their bank-0 operands: vertex 0x110, fragment 0x510, task 0x670, mesh 0x770); the mesh table ends with
// the bank at 0x800 (9 slots). Storage-held pointers of these stages go to SM86_PTR_C0_MESH, below both.
#define SM86_TASK_TABLE    0x670u
#define SM86_MESH_TABLE    0x770u
#define SM86_PTR_C0_MESH   0x480u
// The pass-through vertex program of a mesh program (sm86_lower.c mesh_vs): the host hands the lowering the mesh
// program's code with a copy of its control section whose stage byte (+0x56) is 0 and whose u32 at +0x48 (zero in
// every program of the game) is SM86_MESH_VS_MAGIC. The lowering then writes a Maxwell vertex program for an
// instanced draw of 3 * max-primitives vertices per mesh workgroup (instance = workgroup): it reads the workgroup's
// index map and vertex lines from the mesh records, whose {address, size} the host puts at bank 0 +
// SM86_MESH_VS_TABLE (vertex storage slot 15), and stores each line to the output attribute the mesh header's map
// names; a primitive past the count gives a degenerate triangle (all outputs 0).
#define SM86_MESH_VS_MAGIC 0x5356534du   // "MSVS"
#define SM86_MESH_VS_TABLE 0x200u
#define SM86_MESH_VS_SLOT  15
// 0x1000: the indirect draws' task programs (1ddd3643, c7f0444a, bbcf8a53) write payload arrays at 0x18, 0x418 and
// 0x818 plus 4 * a lane index (to about 0x898); at 0x100 the tasks overwrote each other's payloads (2026-11-08).
#define SM86_TASK_STRIDE   0x1000u
#define SM86_MESH_STRIDE   0x8000u
#define SM86_MESH_STRIDE_LOG2 15u
#define SM86_TASK_STRIDE_LOG2 12u
#define SM86_MESH_MAP      0x7000u

// OOT_SM86_COUNT=<code hash, hex>:<offset>[=rN|=pN][,...] (diagnostic, sm86_lower.c count_emit): every thread of
// that compute program that arrives at the instruction at a listed byte offset adds 1 (=rN: register N; =pN: 1 << N
// when predicate N is set) to u32 k (k = the entry's place in the list) of a counter buffer, whose {u64 address, u32 size} the host puts at bank 0 + SM86_COUNT_C0 and
// binds to a host buffer it reads back and logs (vk_exec_draw.c "count:"). Between the global window (0x410..0x590)
// and the mesh path's words (0x600).
#define SM86_COUNT_C0  0x5f0u
#define SM86_COUNT_MAX 64u
// Nonzero for a program a diagnostic (OOT_SM86_PROBE, OOT_SM86_COUNT) changes: a tag of the setting, which the host
// mixes into that program's cache key, so the probed lowering never comes from the cache nor goes into the normal entry.
// Also nonzero, a fixed tag, for programs whose lowering changed without a KEY_VERSION bump (a BRXU, 2026-11-09).
uint64_t sm86_probe_tag(const uint8_t *code, uint32_t code_size);
// The OOT_SM86_COUNT entries (n of them) as labels for the log ("6a00=p3"), NULL when off.
const char (*sm86_count_points(uint32_t *n))[16];

// The control section's fields the host needs (offsets measured over the game's archives, tools/shader_census.py).
typedef struct {
    uint32_t size;             // the control section's own size (+0xd4; the archives repeat it at +0x30)
    uint32_t code_end;         // end of the instructions in the code block (+0x38), header included
    uint32_t c1_off, c1_len;   // the jump-table constants in the code block (+0x40, +0x3c)
    uint32_t code_size;        // the whole code block (+0x44)
    uint32_t stage;            // NVN's enum at byte +0x56: 0 vertex, 1 fragment, 2 geometry, 3 tess control,
                               // 4 tess eval, 5 compute, 6 task, 7 mesh
    uint32_t local[3];         // compute: the workgroup size (+0x58); task and mesh: {the low half of +0x58, 1, 1} (the
                               // high halves of +0x58..+0x63 hold the output limits there)
} Sm86Control;
// 0 when the magic or the bounds are wrong.
int sm86_control_parse(const uint8_t *control, uint32_t control_size, Sm86Control *c);
#define SM86_CONTROL_MAGIC 0x98761234u
#define SM86_CONTROL_MIN 0xe0u    // the fixed part; the jump-table records follow it
#define SM86_HEADER 0x80u         // the program header before a graphics stage's first instruction
