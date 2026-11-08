// From metroid-prime-4-recomp runtime/vk_exec_draw.c @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied with mp4 renamed to oot.
// From fe3h-pc-recomp runtime/vk_exec_draw.c @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied with fe3h renamed to mp4 only; the viewport orientation follows the window origin mode; storage images a shader stores into are recorded as GPU writes; texture type answers carry the integer class of the format.
// From totk-pc-recomp runtime/vk_exec_draw.c @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh); renamed to fe3h; separable programs: BindProgram with a subset of stages (FE3H); bound handles become pool slots (FE3H uses several descriptor pools).
// Draws, binds and dispatches of the Vulkan op-stream replay: the strong
// vk_exec_op_draw behind vk_exec.c's weak hook. The bind ops fill the VkExec slots; a draw opens the pass, resolves the
// bound program to translated shaders (the translator's answers come from the bound state), asks vk_pipeline.c for the
// pipeline, pushes (or allocates) set 0 as VkPipeInfo.bindings says and records the dynamic state, vertex and index
// buffers and the draw. A dispatch ends the pass and does the same for the compute shader. UPDATE_UB is a buffer copy
// from transient memory between full barriers.
//
// Per command buffer the replay keeps transient memory (support buffers, copied uniform ranges, generated indices),
// descriptor pools, image views and the storage-buffer ranges the GPU may write; all of it is freed, and the ranges are
// reported with vk_guest_written, from one completion callback once the submission's fence is done (the write watch
// does not see GPU writes). Nothing here locks: it runs on the submitting thread under vk_sync_submit's lock.
//
// Provenance: the op semantics (bind ops, program and slot conventions, draw and dispatch flow, vertex streams with a
// zeroed buffer for unbound ones) follow MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m:7409-7460,
// 9365-9488, 9489-9921, 10013-10079 @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou); the uniform-buffer spans, texture and
// sampler descriptor lookups and the dispatch flow follow LAS Recomp runtime/nvn/vulkan/exec.c:1851-1908, 2300-2516
// @ da35b83a (MIT, Copyright (c) 2026 Peiikai). The support buffer layout is Ryujinx's SupportBuffer.cs (MIT).
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "tex.h"
#include "nvn_hmap.h"
#include "sm86_lower.h"
#include "vk.h"
#include "vk_exec.h"

#define MAX_BINDINGS 128
#define MAX_DESC 2048   // a light-list texture array alone is 1024 (tools/xlate/Program.cs)
#define CHUNK_BYTES (4u << 20)
#define ZERO_BYTES 65536u
#define SUPPORT_BYTES 1408u   // Ryujinx SupportBuffer (std140): alpha_test, is_bgra[8], viewport_inverse, viewport_size, ...
#define NVN_STAGE_VS 0
#define NVN_STAGE_FS 1
#define NVN_STAGE_CS 5
#define ONESHOT_WAIT_MS 30000u   // prepare_draw: how long a one-shot draw waits for its pipeline before it is skipped

enum { W_PROGRAM, W_SHADER, W_PIPELINE, W_NO_VS, W_INDEX, W_UB, W_BIND, W_SSBO_ALIGN, W_SSBO_SLOT, W_RESTART, W_INDIRECT, W_MESH, W_COUNT };

static void warn_once(int key, const char *fmt, ...) {
    static volatile long seen[W_COUNT];
    if (__atomic_exchange_n(&seen[key], 1, __ATOMIC_RELAXED)) return;
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    rt_log("vk: %s (logged once)", msg);
}

static void *grow(void *p, int *cap, size_t elem) {
    *cap = *cap ? *cap * 2 : 8;
    p = realloc(p, elem * (size_t)*cap);
    if (!p) rt_fatal("vk: out of memory");
    return p;
}

// ---- resources of one command buffer, freed by its completion callback

typedef struct { VkBuffer buf; VmaAllocation alloc; uint8_t *map; uint64_t size, used; } Chunk;
typedef struct {
    VkImage image;
    VkFormat format;
    VkImageViewType type;
    uint32_t aspect, level0, levels, layer0, layers, swz;
} ViewKey;
_Static_assert(sizeof(ViewKey) == 40, "ViewKey is hashed as bytes: no padding");
typedef struct { uint64_t addr, size; } Range;
typedef struct {
    Chunk *chunks;
    int nchunks, cchunks;
    VkDescriptorPool *pools;
    int npools, cpools;
    Range *written;
    int nwritten, cwritten;
} CbRes;

static CbRes *cur;
static uint64_t cur_fence;

// Descriptor pools and full-size transient chunks go back to these lists when their command buffer completes, instead
// of being destroyed and created again for the next one: each creation is a driver memory allocation (on Windows an
// allocation, a map and a paging-fence wait on the CPU), about ten of them per frame inside the replay.
#define SPARE_POOLS 64
#define SPARE_CHUNKS 16
static SRWLOCK spare_lock = SRWLOCK_INIT;
static VkDescriptorPool spare_pool[SPARE_POOLS];
static Chunk spare_chunk[SPARE_CHUNKS];
static int nspare_pool, nspare_chunk;

// State of the command buffer being recorded that the Vulkan command buffer itself holds.
static struct {
    VkPipeline pipe[2];                  // bound per bind point: 0 graphics, 1 compute
    VkPipeState st;                      // the state vk_pipeline_set_dynamic last recorded
    int st_valid;
    VkBuffer vb[17];                     // vertex buffers last bound
    VkDeviceSize vo[17], vsz[17], vstr[17];
    int vcls[17];                        // their pool class and pool record, for the census
    void *vpool[17];
    NVNbufferRange vin[16];              // the ranges they were resolved from, under vk_mem_gen vgen
    uint64_t vgen;
    int vb_valid;
    uint8_t support[SUPPORT_BYTES];      // last support buffer and where it lives
    float sup_in[3];                     // the support_fill arguments it was filled from
    VkBuffer sup_buf;
    VkDeviceSize sup_off;
    int sup_valid;
    uint8_t c0[3][VK_C0_BYTES];          // last c0 bank per stage (vertex, fragment, compute) and where it lives
    VkBuffer c0_buf[3];
    VkDeviceSize c0_off[3];
    int c0_valid[3];
    VkBuffer c1_buf[3];                 // the lowered constant bank 1 of the shader c1_of[k] (OoT, ubo_info)
    VkDeviceSize c1_off[3], c1_size[3];
    const GfxShader *c1_of[3];
    struct { p_bind_buf ssbo[16], ubo[16]; uint32_t dim[3], binst; } c0_in[3];   // what vk_c0_fill read for c0[k]
    VkPipelineLayout ds_layout;          // graphics layout the sets below were last pushed or bound with
    VkDescriptorSet ds_bound[2];         // allocated set bound per set number (0: none or pushed since)
    VkDescriptorSet ds_last[2];          // last allocated and written set per set number, contents in DS
    VkDescriptorSetLayout ds_last_layout[2];
    int ds_pushed[2];                    // set number holds the push of DS.psig under ds_layout
    VkPipelineLayout pool_layout;        // graphics layout the pool set was last bound with (pool_bind)
    VkDescriptorSet pool_set;
    int res_valid;                       // the graphics sets bound hold what the inputs in RS give (bind_resources)
} B;

// Contents of B.ds_last: the set's packed descriptors (VkDescSlot), which with its layout determine every write. A draw
// whose allocated set would hold the same reuses the set instead of allocating and writing one (a per-draw
// cost). psig: the same for the last push per set number; a push equal to it under the same layout is not recorded
// again (the descriptors stay bound across pipeline binds of that layout).
#define DS_SIG_BYTES (MAX_DESC * sizeof(VkDescSlot))
static struct { uint32_t n[2], pn[2]; uint8_t sig[2][DS_SIG_BYTES], psig[2][DS_SIG_BYTES]; } DS;

// OOT_BIND_STATS counters (bind_stats).
static struct {
    int on;
    uint64_t draws, bind[2], desc[2], push, push_skip, set_new, set_reuse, ac_miss, same_prog, same_all, bind_skip;
} BS = {-1};

// The inputs of the last graphics bind_resources (bind_sig) and what its build counted for the traffic census, which a
// skipped build counts again. RS_BYTES bounds the bound-slot bytes a pipeline reads (RsList): per binding at most one
// c0 storage table, otherwise one bound range or handle per descriptor.
#define RS_BYTES (MAX_BINDINGS * sizeof(((VkExec *)0)->ssbo[0]) + MAX_DESC * sizeof(p_bind_buf))
typedef struct { int cls, use; uint64_t bytes; } RsCount;
static struct {
    int on;                              // OOT_BIND_SKIP: 1 skip (default), 2 check, 0 off
    uint32_t hn;                         // words of h the last signature filled; 0: none
    uint64_t h[24];                      // generations, feedback images, support inputs, base instance
    const VkPipeInfo *info;              // the pipeline whose RsList ranges body holds, as of bind op generation gen
    uint64_t gen;
    uint8_t body[RS_BYTES];
    int rec;                             // a graphics build is running: count_use records, scratch copies mark volatile
    int volatile_;                       // the build copied guest bytes into transient memory: a rebuild differs
    RsCount cnt[2 * VK_MCLS_N], pcnt[2 * VK_MCLS_N];
    int ncnt, pncnt, cnt_over;
    int cmds;                            // check mode: pushes, binds and set writes the build recorded
    uint32_t ckn;                        // check mode: the last build's packed descriptors (bytes in ck)
    uint8_t ck[DS_SIG_BYTES];
} RS = {-1};

// The bound-slot bytes of VkExec a pipeline's descriptors read, as ranges merged where they touch, built once per
// pipeline info: c0's storage table, the uniform and storage ranges of the bound slots and the texture and image
// handles. sup and c0: the support inputs and the base instance count too. p: storage buffers whose address the shader
// loads from a user uniform block; the address read from the block counts too. ok 0: never skipped (too many
// descriptors or such buffers).
#define RS_PTRS 4
typedef struct { uint32_t off, len; } RsRange;
typedef struct { int st; const GfxShaderRes *res; } RsPtr;
typedef struct { uint32_t n, bytes; uint8_t sup, c0, ok, np; RsPtr p[RS_PTRS]; RsRange r[]; } RsList;
static nvn_hmap rs_lists;                // VkPipeInfo pointer -> RsList, until shutdown
static const VkPipeInfo *rs_last;        // the last lookup
static const RsList *rs_last_l;

static uint32_t c0_dim[3], c0_binst;     // the dispatch size and base instance of the draw or dispatch being bound
static uint64_t bind_gen = 1;            // bumped by every bind op that writes ex->prog, ubo, ssbo, tex or img

static void cb_done(void *arg) {   // completion thread
    CbRes *r = arg;
    VkCtx *c = vk_ctx();
    for (int i = 0; i < r->nwritten; i++) vk_guest_written(r->written[i].addr, r->written[i].size);
    for (int i = 0; i < r->npools; i++) {
        vkResetDescriptorPool(c->device, r->pools[i], 0);
        AcquireSRWLockExclusive(&spare_lock);
        int kept = nspare_pool < SPARE_POOLS;
        if (kept) spare_pool[nspare_pool++] = r->pools[i];
        ReleaseSRWLockExclusive(&spare_lock);
        if (!kept) vkDestroyDescriptorPool(c->device, r->pools[i], NULL);
    }
    for (int i = 0; i < r->nchunks; i++) {
        AcquireSRWLockExclusive(&spare_lock);
        int kept = r->chunks[i].size == CHUNK_BYTES && nspare_chunk < SPARE_CHUNKS;
        if (kept) spare_chunk[nspare_chunk++] = r->chunks[i];
        ReleaseSRWLockExclusive(&spare_lock);
        if (!kept) vmaDestroyBuffer(c->vma, r->chunks[i].buf, r->chunks[i].alloc);
    }
    free(r->written), free(r->pools), free(r->chunks), free(r);
}

void vk_exec_draw_new_cb(void) { memset(&B, 0, sizeof B); }

// The buffer and offset stream st was last bound with (vk_dump compares them with a fresh resolve of the address).
int vk_exec_vb_bound(int st, VkBuffer *b, VkDeviceSize *o) {
    if (!B.vb_valid || st < 0 || st > 16) return 0;
    *b = B.vb[st], *o = B.vo[st];
    return 1;
}

static CbRes *cb_res(VkExec *ex) {
    if (cur && cur_fence == ex->fence) return cur;
    cur = calloc(1, sizeof *cur);
    if (!cur) rt_fatal("vk: out of memory");
    cur_fence = ex->fence;
    memset(&B, 0, sizeof B);
    vk_completion_add(ex->fence, cb_done, cur);
    return cur;
}

static uint8_t *scratch(VkExec *ex, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    CbRes *r = cb_res(ex);
    size = (size + 255) & ~255ull;
    Chunk *c = r->nchunks ? &r->chunks[r->nchunks - 1] : NULL;
    if (!c || c->used + size > c->size) {
        if (r->nchunks == r->cchunks) r->chunks = grow(r->chunks, &r->cchunks, sizeof *r->chunks);
        c = &r->chunks[r->nchunks];
        uint64_t n = size > CHUNK_BYTES ? size : CHUNK_BYTES;
        int reused = 0;
        if (n == CHUNK_BYTES) {
            AcquireSRWLockExclusive(&spare_lock);
            if (nspare_chunk) *c = spare_chunk[--nspare_chunk], reused = 1;
            ReleaseSRWLockExclusive(&spare_lock);
        }
        if (reused) {
            c->used = 0;
            r->nchunks++;
            *buf = c->buf;
            *off = 0;
            c->used = size;
            return c->map;
        }
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n,
                                 .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        VkResult res = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &c->buf, &c->alloc, &info);
        if (res != VK_SUCCESS) rt_fatal("vk: transient buffer of %llu bytes failed (%d)", (unsigned long long)n, (int)res);
        c->map = info.pMappedData, c->size = n, c->used = 0;
        r->nchunks++;
    }
    *buf = c->buf;
    *off = c->used;
    uint8_t *p = c->map + c->used;
    c->used += size;
    return p;
}

static void note_written(VkExec *ex, uint64_t addr, uint64_t size) {
    CbRes *r = cb_res(ex);
    for (int i = 0; i < r->nwritten; i++)
        if (r->written[i].addr == addr && r->written[i].size == size) return;
    if (r->nwritten == r->cwritten) r->written = grow(r->written, &r->cwritten, sizeof *r->written);
    r->written[r->nwritten++] = (Range){addr, size};
}

// ---- stand-ins for what a binding names but the game did not bind (zeros, a 1x1 image, the default sampler)

enum { DV_1D, DV_1D_ARRAY, DV_2D, DV_2D_ARRAY, DV_3D, DV_CUBE, DV_CUBE_ARRAY, DV_2D_MS, DV_2D_MS_ARRAY, DV_COUNT };

static struct {
    int ready;
    VkBuffer zbuf;
    VmaAllocation zalloc;
    VkBuffer dbuf;          // the discard buffer: written storage bindings with no backing (ssbo_info), so their
    VmaAllocation dalloc;   // stores never reach the zero buffer every unresolved read shares
    VkBufferView zview;
    VkSampler samp, samp_shadow;
    struct { VkImage image; VmaAllocation alloc; VkImageView view, sview; } img[DV_COUNT];
} Z;

static const struct { VkImageType t; VkImageViewType vt; uint32_t layers; VkSampleCountFlagBits s; VkImageCreateFlags f; } dinfo[DV_COUNT] = {
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_4_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_4_BIT, 0}};

// A 1x1 stand-in image of view type dinfo[i] in `fmt`, moved to GENERAL by the upload batch (which reaches the queue before
// the submission that uses it). *sview is the view a storage descriptor takes (a cube becomes a 2D array).
static void make_dummy(int i, VkFormat fmt, VkImage *image, VmaAllocation *alloc, VkImageView *view, VkImageView *sview) {
    VkCtx *c = vk_ctx();
    int ms = dinfo[i].s != VK_SAMPLE_COUNT_1_BIT;
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = dinfo[i].f, .imageType = dinfo[i].t,
                            .format = fmt, .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = dinfo[i].layers,
                            .samples = dinfo[i].s, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | (ms ? 0 : VK_IMAGE_USAGE_STORAGE_BIT),
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ia = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ia, image, alloc, NULL) != VK_SUCCESS) rt_fatal("vk: stand-in image %d failed", i);
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *image,
                                .viewType = dinfo[i].vt, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, dinfo[i].layers}};
    if (vkCreateImageView(c->device, &vi, NULL, view) != VK_SUCCESS) rt_fatal("vk: stand-in view failed");
    *sview = *view;
    if (!ms && (dinfo[i].vt == VK_IMAGE_VIEW_TYPE_CUBE || dinfo[i].vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)) {
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        if (vkCreateImageView(c->device, &vi, NULL, sview) != VK_SUCCESS) rt_fatal("vk: stand-in view failed");
    }
    VkImageMemoryBarrier2 bar = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                 .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                 .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .image = *image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, dinfo[i].layers}};
    VkCommandBuffer cb = vk_upload_begin();
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &bar};
    vkCmdPipelineBarrier2(cb, &di);
    vk_upload_end();
}

// Storage image stand-ins in the format a shader declares (GfxShaderRes.format: Ryujinx TextureFormat order); Vulkan wants
// the view format to match the declaration. Created on first use.
static const VkFormat tex_format[40] = {
    VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT,
    VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT,
    VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM,
    VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM,
    VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT,
    VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT,
    VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT,
    VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT,
    VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UINT_PACK32, VK_FORMAT_B10G11R11_UFLOAT_PACK32};
static struct { int used, di; VkFormat fmt; VkImage image; VmaAllocation alloc; VkImageView view, sview; } SD[48];

static VkImageView storage_dummy(int di, int code) {
    VkFormat fmt = code > 0 && code < 40 ? tex_format[code] : VK_FORMAT_R8G8B8A8_UNORM;
    if (fmt == VK_FORMAT_R8G8B8A8_UNORM || dinfo[di].s != VK_SAMPLE_COUNT_1_BIT) return Z.img[di].sview;
    for (int i = 0; i < 48; i++) {
        if (!SD[i].used) {
            SD[i].used = 1, SD[i].di = di, SD[i].fmt = fmt;
            make_dummy(di, fmt, &SD[i].image, &SD[i].alloc, &SD[i].view, &SD[i].sview);
            return SD[i].sview;
        }
        if (SD[i].di == di && SD[i].fmt == fmt) return SD[i].sview;
    }
    return Z.img[di].sview;
}

static void dummies_init(void) {
    if (Z.ready) return;
    VkCtx *c = vk_ctx();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = ZERO_BYTES,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &Z.zbuf, &Z.zalloc, &info) != VK_SUCCESS) rt_fatal("vk: zero buffer failed");
    memset(info.pMappedData, 0, ZERO_BYTES);
    {
        VkBufferCreateInfo db = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = ZERO_BYTES, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
        VmaAllocationCreateInfo da = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateBuffer(c->vma, &db, &da, &Z.dbuf, &Z.dalloc, NULL) != VK_SUCCESS) rt_fatal("vk: discard buffer failed");
    }
    VkBufferViewCreateInfo bv = {.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = Z.zbuf,
                                 .format = VK_FORMAT_R32G32B32A32_SFLOAT, .range = ZERO_BYTES};
    if (vkCreateBufferView(c->device, &bv, NULL, &Z.zview) != VK_SUCCESS) rt_fatal("vk: zero buffer view failed");
    samp_par sp;
    memset(&sp, 0, sizeof sp);
    sp.lod_max = 1000.0f;
    Z.samp = vk_sampler_for(&sp);
    sp.cmp_mode = 1, sp.cmp_func = NVN_DEPTH_FUNC_ALWAYS;
    Z.samp_shadow = vk_sampler_for(&sp);

    for (int i = 0; i < DV_COUNT; i++)
        make_dummy(i, VK_FORMAT_R8G8B8A8_UNORM, &Z.img[i].image, &Z.img[i].alloc, &Z.img[i].view, &Z.img[i].sview);
    Z.ready = 1;
}

static void twins_release(VkCtx *c);

static void pool_sets_release(VkCtx *c);

// Releases the stand-ins, the depth twins and the pool sets. The device must be idle; the game path leaves them to process exit like
// the pipelines.
static void mesh_release(VkCtx *c);
void vk_exec_draw_shutdown(void) {
    if (!Z.ready) return;
    VkCtx *c = vk_ctx();
    twins_release(c);
    pool_sets_release(c);
    mesh_release(c);
    for (int i = 0; i < DV_COUNT; i++) {
        if (Z.img[i].sview != Z.img[i].view) vkDestroyImageView(c->device, Z.img[i].sview, NULL);
        vkDestroyImageView(c->device, Z.img[i].view, NULL);
        vmaDestroyImage(c->vma, Z.img[i].image, Z.img[i].alloc);
    }
    for (int i = 0; i < 48 && SD[i].used; i++) {
        if (SD[i].sview != SD[i].view) vkDestroyImageView(c->device, SD[i].sview, NULL);
        vkDestroyImageView(c->device, SD[i].view, NULL);
        vmaDestroyImage(c->vma, SD[i].image, SD[i].alloc);
    }
    memset(SD, 0, sizeof SD);
    nvn_hm_clear(&rs_lists, free);
    rs_last = NULL, rs_last_l = NULL;
    for (int i = 0; i < nspare_pool; i++) vkDestroyDescriptorPool(c->device, spare_pool[i], NULL);
    for (int i = 0; i < nspare_chunk; i++) vmaDestroyBuffer(c->vma, spare_chunk[i].buf, spare_chunk[i].alloc);
    nspare_pool = nspare_chunk = 0;
    vk_usnap_shutdown();
    vkDestroyBufferView(c->device, Z.zview, NULL);
    vmaDestroyBuffer(c->vma, Z.zbuf, Z.zalloc);
    vmaDestroyBuffer(c->vma, Z.dbuf, Z.dalloc);
    memset(&Z, 0, sizeof Z);
}

// ---- resource types

static VkImageViewType res_view_type(uint16_t t) {
    int arr = t & 0x100;
    switch (t & 0xff) {
    case 1: return arr ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
    case 4: return VK_IMAGE_VIEW_TYPE_3D;
    case 5: return arr ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
    default: return arr ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    }
}

static int dummy_index(VkImageViewType vt, int ms) {
    switch (vt) {
    case VK_IMAGE_VIEW_TYPE_1D: return DV_1D;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return DV_1D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_3D: return DV_3D;
    case VK_IMAGE_VIEW_TYPE_CUBE: return DV_CUBE;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return DV_CUBE_ARRAY;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return ms ? DV_2D_MS_ARRAY : DV_2D_ARRAY;
    default: return ms ? DV_2D_MS : DV_2D;
    }
}

// GFX_TEX_* of a bound texture slot (the translator's tex_type / img_type answers).
static int gfx_type_of(const VkTexImage *img, const view_t *v, int hv) {
    if (hv && v->set_target) return vk_pipeline_tex_type(v->target);
    int ms = img->samples > 1;
    switch (img->view_type) {
    case VK_IMAGE_VIEW_TYPE_1D: return GFX_TEX_1D;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return GFX_TEX_1D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return ms ? GFX_TEX_2D_MS_ARRAY : GFX_TEX_2D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_3D: return GFX_TEX_3D;
    case VK_IMAGE_VIEW_TYPE_CUBE: return GFX_TEX_CUBE;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return GFX_TEX_CUBE_ARRAY;
    default: return ms ? GFX_TEX_2D_MS : GFX_TEX_2D;
    }
}

static VkComponentSwizzle swz_of(uint8_t s) {
    switch (s) {
    case NVN_TEXTURE_SWIZZLE_ZERO: return VK_COMPONENT_SWIZZLE_ZERO;
    case NVN_TEXTURE_SWIZZLE_ONE: return VK_COMPONENT_SWIZZLE_ONE;
    case NVN_TEXTURE_SWIZZLE_R: return VK_COMPONENT_SWIZZLE_R;
    case NVN_TEXTURE_SWIZZLE_G: return VK_COMPONENT_SWIZZLE_G;
    case NVN_TEXTURE_SWIZZLE_B: return VK_COMPONENT_SWIZZLE_B;
    default: return VK_COMPONENT_SWIZZLE_A;
    }
}

// A view in STENCIL mode samples the stencil plane (TextureViewSetDepthStencilMode); otherwise depth wins. The
// translator declares every sampled image float and the shader takes the raw bits, as for integer formats (nxnative
// nvn_mtl.m).
VkImageAspectFlags vk_exec_view_aspect(const VkTexImage *img, const view_t *v, int hv) {
    if (hv && v->set_dsmode && v->dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL && (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT))
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    return img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT     ? VK_IMAGE_ASPECT_DEPTH_BIT
           : img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                       : VK_IMAGE_ASPECT_COLOR_BIT;
}

// Image views of registered textures, kept for their image's lifetime: made per command buffer they were created again
// every frame (vkCreateImageView took about 8% of the fight replay on the 7900X). Used under the texture lock, as
// vk_exec_image_gone is (destroy_tex), which hands the views of a destroyed image to the deferred release with it.
typedef struct ViewNode { struct ViewNode *next; ViewKey k; VkImageView view; } ViewNode;
static nvn_hmap view_of;                 // ViewKey -> VkImageView
static nvn_hmap views_by_image;          // VkImage -> its first ViewNode

// An image view of a registered texture for a shader resource of type `rtype` (GfxShaderRes.type); VK_NULL_HANDLE when
// the image cannot serve it (the caller binds a stand-in). Texture lock held.
// 32-bit-per-texel colour formats an R32 storage view may alias (MUTABLE_FORMAT images: vk_texture.c).
static int colour32(VkFormat f) {
    switch (f) {
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32_SFLOAT:
        return 1;
    default: return 0;
    }
}
static VkImageView get_view(VkExec *ex, const VkTexImage *img, const view_t *v, int hv, uint16_t rtype, int storage) {
    VkImageViewType vt = res_view_type(rtype);
    int ms = (rtype & 0x200) != 0;
    if ((img->samples > 1) != ms || (storage && ms)) return VK_NULL_HANDLE;
    if (storage && !(img->usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
        static volatile LONG said;
        if (InterlockedIncrement(&said) <= 6)
            rt_log("vk_exec: storage binding of image %#llx (%ux%ux%u fmt %d) made without STORAGE usage: a dummy is bound",
                   (unsigned long long)img->addr, img->width, img->height, img->depth, (int)img->format);
        return VK_NULL_HANDLE;
    }
    int cube = vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    int ok = 0;
    switch (vt) {
    case VK_IMAGE_VIEW_TYPE_1D:
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: ok = img->type == VK_IMAGE_TYPE_1D; break;
    case VK_IMAGE_VIEW_TYPE_3D: ok = img->type == VK_IMAGE_TYPE_3D; break;
    case VK_IMAGE_VIEW_TYPE_CUBE:
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
        ok = img->type == VK_IMAGE_TYPE_2D && (img->view_type == VK_IMAGE_VIEW_TYPE_CUBE || img->view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY);
        break;
    default: ok = img->type == VK_IMAGE_TYPE_2D; break;
    }
    if (!ok) return VK_NULL_HANDLE;
    uint32_t l0 = hv && v->set_levels ? v->base_level : 0;
    if (l0 >= img->levels) return VK_NULL_HANDLE;
    uint32_t nl = hv && v->set_levels && v->nlevels ? v->nlevels : img->levels - l0;
    if (l0 + nl > img->levels) nl = img->levels - l0;
    if (storage) nl = 1;
    uint32_t y0 = 0, ny = 1;
    if (vt != VK_IMAGE_VIEW_TYPE_3D) {
        y0 = hv && v->set_layers ? v->min_layer : 0;
        if (y0 >= img->layers) return VK_NULL_HANDLE;
        uint32_t rest = hv && v->set_layers && v->nlayers ? v->nlayers : img->layers - y0;
        if (y0 + rest > img->layers) rest = img->layers - y0;
        if (vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_2D) ny = 1;
        else if (vt == VK_IMAGE_VIEW_TYPE_CUBE) ny = 6;
        else if (vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) ny = rest / 6 * 6;
        else ny = rest;
        if (!ny || y0 + ny > img->layers) return VK_NULL_HANDLE;
    }
    if (storage && cube) vt = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    uint32_t aspect = vk_exec_view_aspect(img, v, hv);
    uint8_t sw[4] = {NVN_TEXTURE_SWIZZLE_R, NVN_TEXTURE_SWIZZLE_G, NVN_TEXTURE_SWIZZLE_B, NVN_TEXTURE_SWIZZLE_A};
    if (!storage) {   // MNG nvn_mtl.m tex_swizzle_sel: view selectors replace the texture's, then the format remap
        memcpy(sw, hv && v->set_swizzle ? v->swz : img->swz, 4);
        if (img->alpha_one || img->chan[0])
            for (int i = 0; i < 4; i++) {
                if (img->alpha_one && sw[i] == NVN_TEXTURE_SWIZZLE_A) sw[i] = NVN_TEXTURE_SWIZZLE_ONE;
                else if (img->chan[0] && sw[i] >= NVN_TEXTURE_SWIZZLE_R && sw[i] <= NVN_TEXTURE_SWIZZLE_A)
                    sw[i] = img->chan[sw[i] - NVN_TEXTURE_SWIZZLE_R];
            }
    }
    uint32_t swz = (uint32_t)sw[0] | (uint32_t)sw[1] << 8 | (uint32_t)sw[2] << 16 | (uint32_t)sw[3] << 24;
    VkFormat vf = storage && img->storage_format ? img->storage_format : img->format;   // sRGB stored through UNORM
    // A sampled view whose format (the game's view, or the descriptor's when it follows another descriptor's image at
    // the same storage: vk_texture.c slot_resolve) is another 32-bit colour format reads the image's bytes in that
    // format, as Maxwell does. The upscaler samples the anti-aliasing's RGB9E5 texels in the R11G11B10F scene target
    // (2026-11-10). Both formats in the 32-bit class, so the MUTABLE_FORMAT image allows the view. OOT_VIEW_REFORMAT=0: off.
    if (!storage && hv && v->set_format && (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) && colour32(img->format)) {
        static int on = -1;
        if (on < 0) { const char *e = getenv("OOT_VIEW_REFORMAT"); on = !(e && e[0] == '0'); }
        TexVkFormat tf;
        if (on && tex_vk_format(v->format, &tf) && tf.vk != VK_FORMAT_UNDEFINED && tf.vk != img->format && colour32(tf.vk))
            vf = tf.vk;
    }
    // A storage view the game makes as R32UI / R32I / R32F of a texture of another 32-bit format: the kernel stores
    // texels it packed itself (2026-11-08: the frame's last colour kernel 1cb856c3 packs R11G11B10 into one word and
    // stores it through an R32UI view of the R11G11B10F scene target). Bound in the image's own format, each packed
    // word was converted as a float and the scene came out as clamped junk (the white screen). The translator declares
    // such images as float with an unknown format (storage = 2 here: GfxShaderRes.format 0), so an R32_SFLOAT view
    // carries the word unchanged; a declared integer format gets the integer view.
    if (storage && hv && v->set_format && (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) &&
        (v->format == NVN_FORMAT_R32UI || v->format == NVN_FORMAT_R32F || v->format == NVN_FORMAT_R32I)) {
        if (colour32(img->format) && img->format != VK_FORMAT_R32_UINT && img->format != VK_FORMAT_R32_SINT && img->format != VK_FORMAT_R32_SFLOAT)
            vf = storage == 2 || v->format == NVN_FORMAT_R32F ? VK_FORMAT_R32_SFLOAT
                 : v->format == NVN_FORMAT_R32UI           ? VK_FORMAT_R32_UINT
                                                           : VK_FORMAT_R32_SINT;
    }
    ViewKey k = {img->image, vf, vt, aspect, l0, nl, y0, ny, swz};
    if (!view_of.klen) nvn_hm_init(&view_of, sizeof(ViewKey)), nvn_hm_init(&views_by_image, sizeof(VkImage));
    VkImageView found = (VkImageView)nvn_hm_get(&view_of, &k);
    if (found) return found;
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img->image, .viewType = vt,
                                .format = vf,
                                .components = {swz_of(sw[0]), swz_of(sw[1]), swz_of(sw[2]), swz_of(sw[3])},
                                .subresourceRange = {aspect, l0, nl, y0, ny}};
    VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                      .usage = storage ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT};
    if (img->block_views || img->storage_format || img->storage_r32) ci.pNext = &use;   // the image's COLOR_ATTACHMENT usage is its block views'
                                                                     // (vk_texture.c); an sRGB image's STORAGE usage is its UNORM view's
    VkImageView view;
    if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS) return VK_NULL_HANDLE;
    ViewNode *n = malloc(sizeof *n);
    if (!n) rt_fatal("vk: out of memory");
    *n = (ViewNode){nvn_hm_get(&views_by_image, &img->image), k, view};
    nvn_hm_put(&views_by_image, &img->image, n);
    nvn_hm_put(&view_of, &k, (void *)view);
    return view;
}

// An attachment view of a registered texture, kept with its sampled views for the image's lifetime: made per command
// buffer, every render target's views were created again each frame (vkCreateImageView under vk_exec_pass_begin, 2%
// of the fight replay on the 7900X). Attachment keys carry a swizzle word no sampled key has (those hold four NVN
// selectors, each below 8). The texture lock guards the tables, as for get_view; vk_exec_pass_begin may already hold it.
VkImageView vk_exec_target_view(const VkTexImage *img, VkFormat format, VkImageViewType type, uint32_t aspect,
                                uint32_t level, uint32_t base, uint32_t layers, int blocks) {
    ViewKey k = {img->image, format, type, aspect, level, 1, base, layers, blocks ? 0xfffffffeu : 0xffffffffu};
    vk_texture_lock();
    if (!view_of.klen) nvn_hm_init(&view_of, sizeof(ViewKey)), nvn_hm_init(&views_by_image, sizeof(VkImage));
    VkImageView view = (VkImageView)nvn_hm_get(&view_of, &k);
    if (!view) {
        VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img->image,
                                    .viewType = type, .format = format, .subresourceRange = {aspect, level, 1, base, layers}};
        VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                          .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
        if (blocks || img->storage_r32 || img->storage_format) ci.pNext = &use;   // not the BC image's sampling and transfer
                                                                               // usages, nor a storage usage by extension
        if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS)
            rt_fatal("vk: render target view (format %d) could not be created", (int)format);
        ViewNode *n = malloc(sizeof *n);
        if (!n) rt_fatal("vk: out of memory");
        *n = (ViewNode){nvn_hm_get(&views_by_image, &img->image), k, view};
        nvn_hm_put(&views_by_image, &img->image, n);
        nvn_hm_put(&view_of, &k, (void *)view);
    }
    vk_texture_unlock();
    return view;
}

void vk_exec_image_gone(VkImage image) {
    vk_exec_bc_gone(image);
    if (!views_by_image.klen) return;
    for (ViewNode *n = nvn_hm_del(&views_by_image, &image), *next; n; n = next) {
        next = n->next;
        nvn_hm_del(&view_of, &n->k);
        vk_defer_view(n->view);
        free(n);
    }
}

// ---- program to shaders

// (program, stage, answers) -> shader, 4-way set-associative, most recently used first. A miss costs a SHA-256 of the
// whole shader code (gfx_shader.c req_init), slow on CPUs without SHA instructions: a field frame looks up ~3000 and
// the 2048-slot direct-mapped table this replaced missed 3% of them on conflicts.
#define SHADER_SETS 2048
#define SHADER_WAYS 4
typedef struct { uint64_t serial; int stage; int used; GfxShaderAnswers a; const GfxShader *s; } ShaderEnt;
static ShaderEnt shader_tab[SHADER_SETS][SHADER_WAYS];

// OoT: NVN2 control sections (sm86_lower.h): the stage at byte +0x56, the code block's size at +0x44.
static const GfxShader *C1CUR[6];                           // the shader shader_for last resolved per NVN stage

static int find_stage_scan(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    for (uint32_t i = 0; i < p->nstage && i < 6; i++) {
        const uint8_t *c = p->sd[i].control;
        if (!c || !p->sd[i].data) continue;
        if (c[0x56] != nvn_stage) continue;
        memcpy(len, c + 0x44, 4);
        *ctl = c, *code = (const uint8_t *)(uintptr_t)p->sd[i].data;
        return 1;
    }
    return 0;
}

// Which shader of a program serves each NVN stage, and its code length, per program serial (new at every
// nvnProgramSetShaders): the scan reads two words of every stage's control block, cold guest cache lines, and runs
// several times a draw. Direct-mapped; a collision only scans again.
#define STAGEM_N 512
static struct { uint64_t serial; int8_t idx[6]; uint32_t len[6]; } stagem[STAGEM_N];

int vk_exec_find_stage(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    if (!p->serial || (unsigned)nvn_stage >= 6) return find_stage_scan(p, nvn_stage, ctl, code, len);
    typeof(stagem[0]) *m = &stagem[p->serial * 0x9E3779B97F4A7C15ull >> 55];
    if (m->serial != p->serial) {
        m->serial = p->serial;
        memset(m->idx, -1, sizeof m->idx);
        for (uint32_t i = 0; i < p->nstage && i < 6; i++) {
            const uint8_t *c = p->sd[i].control;
            if (!c || !p->sd[i].data) continue;
            uint32_t st = c[0x56];
            if (st < 6 && m->idx[st] < 0) m->idx[st] = (int8_t)i, memcpy(&m->len[st], c + 0x44, 4);
        }
    }
    int i = m->idx[nvn_stage];
    if (i < 0) return 0;
    *ctl = p->sd[i].control, *code = (const uint8_t *)(uintptr_t)p->sd[i].data, *len = m->len[nvn_stage];
    return 1;
}

// Texture and image types of a stage's first 32 slots, each kept with the handle and the generation and slot token it
// was looked up under: a slot is looked up again, a hash probe, only when the handle changed or its slot may resolve
// otherwise (slot_still).
static struct { uint64_t tex[32], img[32], ttok[32], itok[32]; uint32_t tgen[32], igen[32]; uint8_t tt[32], it[32]; } AC[6];

// vk_texture_slot_token of the slot a texture or image handle names; an empty handle always resolves to nothing.
static uint64_t token_of(uint64_t h) { return h ? vk_texture_slot_token((int)(h & 0xfffff)) : 1; }

// Whether what handle h resolved to under the global generation *egen and its slot token etok (both read before that
// lookup) still holds at global generation gen: nothing at all changed since, or not its slot (the token, read only when
// the global generation moved, about every 16 draws). A token hit moves *egen to gen.
static int slot_still(uint32_t *egen, uint64_t etok, uint64_t h, uint32_t gen) {
    if (*egen == gen) return 1;
    uint64_t t = token_of(h);
    if (!t || t != etok) return 0;
    *egen = gen;
    return 1;
}
static int shader_memo_on(void);

// The slots of (program, stage) whose types its translation reads (gfx_shader_slot_mask), learned from its first
// shader; ~0u while unknown. The types of other slots select the same shader, so answers_for leaves them zero instead
// of looking up all 64 handles. Direct-mapped by serial and stage; a collision only relearns.
#define SLOTM_N 4096
static struct { uint64_t serial; int st; uint32_t m; } slotm[SLOTM_N];

static uint32_t slotm_idx(uint64_t serial, int st) { return (uint32_t)((serial * 8 + (uint64_t)st) * 0x9E3779B97F4A7C15ull >> 52); }

static uint32_t slot_mask_of(uint64_t serial, int st) {
    const typeof(slotm[0]) *e = &slotm[slotm_idx(serial, st)];
    return serial && e->serial == serial && e->st == st ? e->m : ~0u;
}

static void slot_mask_learn(uint64_t serial, int st, const GfxShader *s) {
    if (!serial || slot_mask_of(serial, st) != ~0u) return;
    slotm[slotm_idx(serial, st)] = (typeof(slotm[0])){serial, st, gfx_shader_slot_mask(s)};
}

// The numeric class of a Vulkan format: GFX_TEX_UINT, GFX_TEX_SINT or 0 (float, normalised, depth, compressed).
static uint8_t format_class(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8_UINT: case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_B8G8R8A8_UINT: case VK_FORMAT_A8B8G8R8_UINT_PACK32: case VK_FORMAT_A2R10G10B10_UINT_PACK32:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16_UINT:
    case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32B32_UINT:
    case VK_FORMAT_R32G32B32A32_UINT:
        return GFX_TEX_UINT;
    case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8_SINT: case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_B8G8R8A8_SINT: case VK_FORMAT_A8B8G8R8_SINT_PACK32: case VK_FORMAT_A2R10G10B10_SINT_PACK32:
    case VK_FORMAT_A2B10G10R10_SINT_PACK32: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16_SINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32_SINT:
    case VK_FORMAT_R32G32B32A32_SINT:
        return GFX_TEX_SINT;
    default: return 0;
    }
}

// The slot's texture type for the translator: its dimensionality (GFX_TEX_*) and, for an integer format, the class flag
// (GFX_TEX_UINT / GFX_TEX_SINT): the translator declares integer samplers for those (this game samples its R32UI light
// list; a float sampler of an integer image is invalid in Vulkan and read it wrong).
static uint8_t type_of_handle(uint64_t h) {   // texture lock held
    view_t v;
    int hv;
    const VkTexImage *img = h ? vk_texture_by_id_locked((int)(h & 0xfffff), &v, &hv) : NULL;
    if (!img) return 0;
    VkFormat f = img->format;
    if (hv && v.set_format) {
        TexVkFormat vf;
        if (tex_vk_format(v.format, &vf) && vf.vk != VK_FORMAT_UNDEFINED) f = vf.vk;
    }
    uint8_t cls = format_class(f);
    if (cls) {
        static volatile LONG said;
        if (InterlockedIncrement(&said) <= 5)
            rt_log("vk_exec: texture slot %d (%ux%u vk format %d) answered as an integer texture (%s)", (int)(h & 0xfffff), img->width,
                   img->height, (int)f, cls == GFX_TEX_UINT ? "uint" : "sint");
    }
    return (uint8_t)(gfx_type_of(img, &v, hv) | cls);
}

// The answers of stage st for the slots in mask; the types of other slots stay zero.
static void answers_for(VkExec *ex, int st, const uint8_t *ctl, uint32_t mask, GfxShaderAnswers *a) {
    memset(a, 0, sizeof *a);
    if (st == NVN_STAGE_VS) vk_pipeline_answers(&ex->pipe, a, NULL);
    else if (st == NVN_STAGE_FS) vk_pipeline_answers(&ex->pipe, NULL, a);
    else {
        // The workgroup size as sm86_control_parse reads it: for task and mesh programs only the low half of +0x58 (the
        // high halves of +0x58..+0x63 hold their output limits). Read raw, a mesh program got 32 x 1 x 128 = 4096
        // threads, four times Vulkan's limit, and faulted the GPU on every mesh draw (2026-11-07, run-68, --probe-cs).
        Sm86Control c;
        if (sm86_control_parse(ctl, SM86_CONTROL_MIN, &c))
            for (int k = 0; k < 3; k++) a->local[k] = (uint16_t)(c.local[k] ? c.local[k] : 1);
        else
            for (int k = 0; k < 3; k++) {
                uint32_t l;
                memcpy(&l, ctl + 0x58 + 4 * k, 4);
                a->local[k] = (uint16_t)(l ? l : 1);
            }
    }
    // The shared memory the kernel may use: the NVN2 control section's field for it is not identified yet (none of
    // its words matches the kernels' STS/LDS/LDSM use), and 0 loses every shared store, so every compute program
    // gets Maxwell's and Vulkan's common limit; it costs occupancy, not correctness (2026-11-06, docs/phase3.md).
    if (st == NVN_STAGE_CS) a->shared_bytes = 0xc000;
    typeof(AC[0]) *c = &AC[st];
    // The generation and tokens are read before the lookups (one that creates an entry changes them) and unlocked: a
    // registration racing with them is ordered after this draw, and a slot stored under older ones is only looked up
    // again.
    uint32_t gen = vk_texture_slots_gen();
    int locked = 0, check = shader_memo_on() == 2;
    for (uint32_t m = mask; m; m &= m - 1) {
        int slot = __builtin_ctz(m);
        uint64_t h = ex->tex[st][slot], hi = ex->img[st][slot];
        int tmiss = c->tex[slot] != h || !c->ttok[slot] || !slot_still(&c->tgen[slot], c->ttok[slot], h, gen);
        int imiss = c->img[slot] != hi || !c->itok[slot] || !slot_still(&c->igen[slot], c->itok[slot], hi, gen);
        uint64_t tt = tmiss ? token_of(h) : 0, it = imiss ? token_of(hi) : 0;
        if ((tmiss || imiss || check) && !locked) vk_texture_lock(), locked = 1, BS.ac_miss++;
        if (tmiss) c->tt[slot] = type_of_handle(h), c->tex[slot] = h, c->ttok[slot] = tt, c->tgen[slot] = gen;
        if (imiss) c->it[slot] = type_of_handle(hi), c->img[slot] = hi, c->itok[slot] = it, c->igen[slot] = gen;
        if (check && (type_of_handle(h) != c->tt[slot] || type_of_handle(hi) != c->it[slot]))   // OOT_SHADER_MEMO=check
            rt_fatal("vk_exec: kept texture types of stage %d slot %d differ from a lookup", st, slot);
        a->tex_type[slot] = c->tt[slot], a->img_type[slot] = c->it[slot];
    }
    if (locked) vk_texture_unlock();
}

// *present: the program has this stage. Returns NULL when it has not or the translation failed. pending (graphics
// stages, vk_pipeline_async_on): a shader not translated yet is queued and NULL returned with *pending = 1. await: a
// shader queued earlier is waited for (gfx_shader_await) instead.
static const GfxShader *shader_for(VkExec *ex, int st, int *present, int *pending, int await) {
    const uint8_t *ctl, *code;
    uint32_t len;
    *present = 0;
    if (pending) *pending = 0;
    if (!ex->prog || !vk_exec_find_stage(ex->prog, st, &ctl, &code, &len)) return NULL;
    *present = 1;
    GfxShaderAnswers a;
    uint64_t serial = ex->prog->serial;
    answers_for(ex, st, ctl, slot_mask_of(serial, st), &a);
    for (int i = 0; i < 32; i++)
        if (a.tex_type[i] & (GFX_TEX_UINT | GFX_TEX_SINT)) {
            static volatile LONG said;
            if (InterlockedIncrement(&said) <= 5)
                rt_log("vk_exec: stage %d slot %d answered type %#x (mask %#x) for program %llu", st, i, a.tex_type[i],
                       slot_mask_of(serial, st), (unsigned long long)serial);
            break;
        }
    ShaderEnt *set = NULL;
    if (serial) {
        uint64_t h = (serial * 0x9E3779B97F4A7C15ull ^ (uint64_t)st) ^ nvn_hm_hash(&a, sizeof a);   // a is zero-padded
        set = shader_tab[(h ^ h >> 29) & (SHADER_SETS - 1)];
        for (int w = 0; w < SHADER_WAYS && set[w].used; w++) {
            ShaderEnt *e = &set[w];
            if (e->serial != serial || e->stage != st || memcmp(&e->a, &a, sizeof a)) continue;
            if (w) {   // to the front
                ShaderEnt hit = *e;
                memmove(&set[1], &set[0], w * sizeof *set);
                set[0] = hit;
            }
            return C1CUR[st] = set[0].s;
        }
    }
    int gs = st == NVN_STAGE_VS ? GFX_STAGE_VS : st == NVN_STAGE_FS ? GFX_STAGE_FS : GFX_STAGE_CS;
    const GfxShader *s = await ? gfx_shader_await(gs, ctl, code, len, &a)
                       : pending && vk_pipeline_async_on() ? gfx_shader_get_async(gs, ctl, code, len, &a, pending)
                                                           : gfx_shader_get(gs, ctl, code, len, NULL, &a);
    if (pending && *pending) return NULL;
    if (s) slot_mask_learn(serial, st, s), C1CUR[st] = s;
    if (!s) warn_once(W_SHADER, "a stage %d shader of program %llu failed to translate: draws with it are skipped", st,
                      (unsigned long long)serial);
    else if (set) {   // in front, the least recently used way drops out
        memmove(&set[1], &set[0], (SHADER_WAYS - 1) * sizeof *set);
        set[0] = (ShaderEnt){.serial = serial, .stage = st, .used = 1, .a = a, .s = s};
    }
    return s;
}

// ---- descriptors

static int nvn_stage_of(int gfx_stage) {
    return gfx_stage == GFX_STAGE_VS ? NVN_STAGE_VS : gfx_stage == GFX_STAGE_FS ? NVN_STAGE_FS : NVN_STAGE_CS;
}

static VkDescriptorBufferInfo zero_info(uint64_t max) {
    return (VkDescriptorBufferInfo){Z.zbuf, 0, max < ZERO_BYTES ? max : ZERO_BYTES};
}

// vk_mem_count, also kept per (class, use) while a graphics build runs (RS).
static void count_use(int cls, int use, uint64_t n) {
    vk_mem_count(cls, use, n);
    if (!RS.rec) return;
    for (int i = 0; i < RS.ncnt; i++)
        if (RS.cnt[i].cls == cls && RS.cnt[i].use == use) {
            RS.cnt[i].bytes += n;
            return;
        }
    if (RS.ncnt < (int)(sizeof RS.cnt / sizeof RS.cnt[0])) RS.cnt[RS.ncnt++] = (RsCount){cls, use, n};
    else RS.cnt_over = 1;
}

// A buffer descriptor for [addr, addr+size) of guest memory: its backing (a pool's VRAM mirror when it has one) when
// aligned, a copy in transient memory when the backing is host memory with an unaligned offset. 0 when neither works.
// written: the shader stores into it (a mirror's range goes back to guest memory at the end of the submission).
static void xfb_before_draw(VkExec *ex);
typedef struct ProbeJob ProbeJob;
static ProbeJob *probe_begin(VkExec *ex, int draw);
static void probe_end(VkExec *ex, ProbeJob *j);   // below: begins a capture before the first draw of a bracket
static int bind_range(VkExec *ex, uint64_t addr, uint64_t size, uint64_t align, uint64_t max, int use, int written,
                      VkDescriptorBufferInfo *out) {
    VkMemSpan sp;
    if (!addr || !vk_mem_resolve(addr, 1, &sp)) return 0;
    uint64_t n = size && size < sp.avail ? size : sp.avail;
    if (n > max) n = max;
    if (!n) return 0;
    if (sp.offset % align == 0) {
        VK_MEM_USE(sp);
        *out = (VkDescriptorBufferInfo){sp.buffer, sp.offset, n};
        if (written) vk_mem_gpu_wrote(&sp, n), vk_usnap_drop(addr, n);
        count_use(sp.cls, use, n);
        VK_MEM_BOUND(sp);
        return 1;
    }
    if (sp.alloc) return 0;
    VkBuffer b;
    VkDeviceSize o;
    RS.volatile_ = 1;
    memcpy(scratch(ex, n, &b, &o), (const void *)(uintptr_t)addr, n);
    vk_mem_count(sp.cls, VK_MUSE_CPU_READ, n);
    vk_mem_count(VK_MCLS_SCRATCH, use, n);
    *out = (VkDescriptorBufferInfo){b, o, n};
    return 1;
}

static void c0_table(uint8_t *at, const p_bind_buf b[16]) {
    for (int r = 0; r < 16; r++) {
        uint64_t a = b[r].addr;
        uint32_t n = (uint32_t)b[r].size;
        memcpy(at + 16 * r, &a, 8);
        memcpy(at + 16 * r + 8, &n, 4);
    }
}

// The driver constant bank c0 as GLSLC-compiled shaders read it. Storage-buffer tables {u64 address, u32 size} every
// 16 bytes at 0x110 (vertex), 0x510 (fragment), 0x310 (compute): Ryujinx's GlobalToStorage fallback loads each
// candidate base from here and compares the global pointer against it. Compute: dispatch size at 0x200, the uniform
// buffer table at 0x210 + 16 r. Graphics: the largest point size (Maxwell's limit, GLSLC clamps gl_PointSize to it) at
// 0x610, the base instance at 0x644. Other stages: zeros. Unbound slots read zero, since the replay keeps them zero.
// Layout: nxnative dm/gfx/nvn/nvn_mtl.m (MIT).
void vk_c0_fill(uint8_t bank[VK_C0_BYTES], int stage, const p_bind_buf ssbo[16], const p_bind_buf ubo[16],
                const uint32_t dispatch[3], uint32_t base_instance) {
    memset(bank, 0, VK_C0_BYTES);
    int base = stage == NVN_STAGE_VS ? 0x110 : stage == NVN_STAGE_FS ? 0x510 : stage == NVN_STAGE_CS ? 0x310 : -1;
    if (base < 0) return;
    c0_table(bank + base, ssbo);
    if (stage == NVN_STAGE_CS) {
        memcpy(bank + 0x200, dispatch, 12);
        c0_table(bank + 0x210, ubo);
        return;
    }
    const float max_point = 2047.0f;
    memcpy(bank + 0x610, &max_point, 4);
    memcpy(bank + 0x644, &base_instance, 4);
}

static int c0_on(void) {   // OOT_FIX_C0=0: c0 reads zeros instead of the driver constants
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("OOT_FIX_C0");
    }
    return on;
}

// NVN2 (SM86) compute shaders read constant bank 0 as: 0x10 reserved bytes (the dispatch size here), the constants
// pushed with nvnCommandBufferPushComputeConstants, then the driver's tables of the bound resources in slot order:
// storage buffers {u64 address, u32 size, u32 0} for control[0xb2] slots, texture handles (u64) for control[0xb3]
// slots, image handles (u64) for control[0xb6] slots (measured on the upscaler's kernels, docs/phase3.md
// 2026-11-06: a kernel with a 0x54-byte push, 3 storage slots, 4 texture slots and 2 image slots reads its storage
// slot 2 at 0x90, textures 0 and 3 at 0xa0 and 0xb8, image 1 at 0xc8; one with no push reads slots 0 and 1 at 0x10
// and 0x20). The push is consumed by the next dispatch.
static int nvn2_cs_layout(const VkExec *ex, uint32_t *base, uint32_t *ns, uint32_t *nt, uint32_t *ni) {
    const uint8_t *ctl, *code;
    uint32_t len;
    if (!ex->prog || !vk_exec_find_stage(ex->prog, NVN_STAGE_CS, &ctl, &code, &len)) return 0;
    *ns = ctl[0xb2], *nt = ctl[0xb3], *ni = ctl[0xb6];
    // The tables end at the u16 at control +0xba (2026-11-06: eb67c232 1 storage + 2 textures ends at 0x40 and reads
    // its storage slot at 0x20; 5d415995 5 storage, no push, ends at 0x60; c82d3cba ends at 0xd8): the shader's own
    // layout. The game may push more bytes than the shader declares, and a base from the pushed length then moved the
    // tables (the level's compute textures read no handle). The pushed length is the fallback.
    uint32_t end = (uint32_t)ctl[0xba] | (uint32_t)ctl[0xbb] << 8, tables = 16 * *ns + 8 * *nt + 8 * *ni;
    if (end >= 0x10 + tables && end <= VK_C0_BYTES) *base = end - tables;
    else *base = 0x10 + ((ex->push_cc_len + 15u) & ~15u);
    return 1;
}

// ---- mesh draws (op_draw_mesh): the task and mesh programs run as compute programs (sm86_lower.c lower_isbe) on
// records in one device-local buffer, the task records first. While M.on, the compute bank 0 also carries the record
// table entries and the split (sm86_lower.h SM86_MESH_C0_*), and those two storage bindings are the records. The
// addresses in bank 0 are stand-ins: the translated program subtracts the entry's address from the one it computes, so
// only the binding matters.
#define MESH_FAKE_TASK 0x7e0000000000ull
#define MESH_FAKE_MESH 0x7f0000000000ull
static struct {
    int on;
    int drawing;      // the pass-through draw: vertex storage slot 15 is the mesh records
    uint32_t table;   // SM86_TASK_TABLE or SM86_MESH_TABLE: the running stage's storage table in bank 0
    uint32_t shift;
    uint32_t count;   // task workgroups that run (bank 0 SM86_MESH_C0_COUNT); an indirect draw's comes from ind
    VkMemSpan ind;    // an indirect draw's arguments (ind.buffer set): word 0 is its count, copied on the GPU
    int mi;           // an indirect draw run from its real count (op_draw_mesh): dispatch and draw take their counts from MI.buf + mi_off
    int map;          // its mesh workgroups are a packed list (mesh_compact; bank 0 SM86_MESH_C0_MAPPED)
    VkDeviceSize mi_off;
    VkBuffer buf;
    VmaAllocation mem;
    VkDeviceSize cap, task_size, mesh_size;   // the mesh records follow the task records
} M;
// The counts of the mesh-only indirect draws (op_draw_mesh): {x, 1, 1, 0} then a VkDrawIndirectCommand, per draw.
static struct { VkBuffer buf; VmaAllocation mem; uint32_t next; } MI;
enum { MI_SLOTS = 16384, MI_MAX_GROUPS = 5120 };   // the game's lists stop at 0x21c00 items, 4320 groups of 32

static uint32_t ptr_count(int st);
static uint64_t ptr_value(const VkExec *ex, int st, uint32_t i);
static struct { uint64_t addr, size; } GW[VK_GW_N];   // the global window (gw_select, below)
static uint32_t gw_n;
// getenv takes the CRT's lock and walks the environment: on a path run per draw it was 7% of the render thread by Saria (2026-11-09).
#define ENV_ONCE(name) ({ static const char *v_; static int got_; if (!got_) v_ = getenv(name), got_ = 1; v_; })
static int cnt_on(void);
static void mesh_c0(const VkExec *ex, uint8_t bank[VK_C0_BYTES]) {
    for (uint32_t k = 0; k < 16 && M.table + 16 * k + 16 <= VK_C0_BYTES; k++) {   // the stage's storage slots
        uint64_t a = ex->ssbo[NVN_STAGE_CS][k].addr;
        uint32_t n = (uint32_t)ex->ssbo[NVN_STAGE_CS][k].size;
        memcpy(bank + M.table + 16 * k, &a, 8), memcpy(bank + M.table + 16 * k + 8, &n, 4);
    }
    for (uint32_t i = 0, np = ptr_count(NVN_STAGE_CS); i < np && i < SM86_PTR_MESH_MAX; i++) {
        uint64_t p = ptr_value(ex, NVN_STAGE_CS, i);
        memcpy(bank + SM86_PTR_C0_MESH + 8 * i, &p, 8);
    }
    uint64_t a = MESH_FAKE_MESH;
    uint32_t sz = (uint32_t)M.mesh_size, shift = M.shift, mask = shift >= 31 ? ~0u : (1u << shift) - 1;
    memcpy(bank + SM86_MESH_C0_MESH, &a, 8), memcpy(bank + SM86_MESH_C0_MESH + 8, &sz, 4);
    a = MESH_FAKE_TASK, sz = (uint32_t)M.task_size;
    memcpy(bank + SM86_MESH_C0_TASK, &a, 8), memcpy(bank + SM86_MESH_C0_TASK + 8, &sz, 4);
    memcpy(bank + SM86_MESH_C0_MASK, &mask, 4), memcpy(bank + SM86_MESH_C0_SHIFT, &shift, 4);
    memcpy(bank + SM86_MESH_C0_COUNT, &M.count, 4);
    uint32_t mapped = M.map ? ~0u : 0u;
    memcpy(bank + SM86_MESH_C0_MAPPED, &mapped, 4);
    for (uint32_t k = 0; k < gw_n; k++) {   // gw_select_mesh
        uint32_t wsz = (uint32_t)GW[k].size;
        memcpy(bank + VK_GW_C0 + 16 * k, &GW[k].addr, 8), memcpy(bank + VK_GW_C0 + 16 * k + 8, &wsz, 4);
    }
    if (cnt_on()) {   // OOT_SM86_COUNT in a task or mesh program
        uint64_t ca = 1ull << 32;
        uint32_t cs = 4 * SM86_COUNT_MAX;
        memcpy(bank + SM86_COUNT_C0, &ca, 8), memcpy(bank + SM86_COUNT_C0 + 8, &cs, 4);
    }
}

static uint32_t ptr_count(int st);
static uint64_t ptr_value(const VkExec *ex, int st, uint32_t i);

// The global window (tools/xlate/patches/0019): a compute program's memory access whose address the translator
// cannot trace to a constant-buffer word (a pointer it loaded from memory at a run-time index) is checked on the GPU
// against VK_GW_N {u64 address, u32 size} entries at bank 0 + VK_GW_C0, each bound as a storage buffer; gw_select
// fills them before the dispatch with the buffers the program's pointers can reach. 2026-11-07: the frame's first
// culling kernel (e6d5432f) counts the visible objects with atomics through pointers it reads from its object table,
// into the group counts of the culling dispatches that follow; all 49 of its such accesses were dropped, every count
// stayed 0, nothing was culled in, and the indirect mesh draws (the ground's colour) drew nothing.

static int cnt_on(void);
static void shader_hex(const GfxShader *s, char hex[17]);
static void nvn2_c0_fill_cs(const VkExec *ex, uint8_t bank[VK_C0_BYTES]) {
    memset(bank, 0, VK_C0_BYTES);
    memcpy(bank, c0_dim, 12);
    memcpy(bank + 0x10, ex->push_cc, ex->push_cc_len < VK_C0_BYTES - 0x10 ? ex->push_cc_len : VK_C0_BYTES - 0x10);
    uint32_t base, ns, nt, ni;
    if (!nvn2_cs_layout(ex, &base, &ns, &nt, &ni)) return;
    // The storage-held pointers the lowering moved to c0 (sm86_lower.h SM86_PTR_C0), unless the tables reach there.
    uint32_t np = ptr_count(NVN_STAGE_CS);
    if (np && base + 16 * ns + 8 * nt + 8 * ni <= SM86_PTR_C0)
        for (uint32_t i = 0; i < np; i++) {
            uint64_t p = ptr_value(ex, NVN_STAGE_CS, i);
            memcpy(bank + SM86_PTR_C0 + 8 * i, &p, 8);
        }
    uint32_t o = base;
    for (uint32_t k = 0; k < ns && k < 16 && o + 16 <= VK_C0_BYTES; k++, o += 16) {
        uint32_t size = (uint32_t)ex->ssbo[NVN_STAGE_CS][k].size;
        memcpy(bank + o, &ex->ssbo[NVN_STAGE_CS][k].addr, 8);
        memcpy(bank + o + 8, &size, 4);
    }
    for (uint32_t t = 0; t < nt && t < 128 && o + 8 <= VK_C0_BYTES; t++, o += 8) memcpy(bank + o, &ex->tex[NVN_STAGE_CS][t], 8);
    for (uint32_t i = 0; i < ni && i < 32 && o + 8 <= VK_C0_BYTES; i++, o += 8) memcpy(bank + o, &ex->img[NVN_STAGE_CS][i], 8);
    if (!M.on && o <= VK_GW_C0)   // gw_select: none when the tables reach the window
        for (uint32_t k = 0; k < gw_n; k++) {
            uint32_t sz = (uint32_t)GW[k].size;
            memcpy(bank + VK_GW_C0 + 16 * k, &GW[k].addr, 8), memcpy(bank + VK_GW_C0 + 16 * k + 8, &sz, 4);
        }
    if (!M.on && o <= SM86_COUNT_C0 && cnt_on()) {   // OOT_SM86_COUNT: any aligned address; the count is its offset
        uint64_t a = 1ull << 32;
        uint32_t sz = 4 * SM86_COUNT_MAX;
        memcpy(bank + SM86_COUNT_C0, &a, 8), memcpy(bank + SM86_COUNT_C0 + 8, &sz, 4);
    }
    // Maxwell's tables as well, past the NVN2 layout: the translator's GlobalToStorage fallback searches the storage
    // table at 0x310 for the base of a global pointer it could not trace to a constant (the kernels' stores through
    // the address at bank 0 + 0x10), and the uniform table at 0x210 for the blocks read as storage.
    c0_table(bank + 0x310, ex->ssbo[NVN_STAGE_CS]);
    c0_table(bank + 0x210, ex->ubo[NVN_STAGE_CS]);
    // OOT_BANK_WATCH=<lo hex>:<hi hex> (diagnostic): each compute program whose bank 0 holds a 64-bit value in that range,
    // once per program, offset and value: which kernel is given a buffer the game later reads.
    static uint64_t bw_lo, bw_hi;
    static int bw_on = -1;
    if (bw_on < 0) { const char *e = getenv("OOT_BANK_WATCH"); bw_on = e && sscanf(e, "%llx:%llx", (unsigned long long *)&bw_lo, (unsigned long long *)&bw_hi) == 2; }
    if (bw_on) {
        static struct { char hx[17]; uint32_t off; uint64_t v; } seen[512];
        static int nseen;
        char hx[17];
        shader_hex(C1CUR[NVN_STAGE_CS], hx);
        for (uint32_t b = 0; b + 8 <= VK_C0_BYTES; b += 8) {
            uint64_t v;
            memcpy(&v, bank + b, 8);
            if (v < bw_lo || v >= bw_hi) continue;
            int known = 0;
            for (int i = 0; i < nseen && !known; i++) known = seen[i].off == b && seen[i].v == v && !strcmp(seen[i].hx, hx);
            if (known || nseen >= 512) continue;
            strcpy(seen[nseen].hx, hx), seen[nseen].off = b, seen[nseen].v = v, nseen++;
            rt_log("vk_exec: OOT_BANK_WATCH: %s c0+%#x = %#llx (frame %llu)", hx, b, (unsigned long long)v, (unsigned long long)gfx_stats().frames);
        }
    }
}

// The NVN slot a bank-0 byte offset falls in under that layout: *kind 0 storage buffer, 1 texture, 2 image; -1 outside.
static int nvn2_c0_slot(const VkExec *ex, uint32_t off, int *kind) {
    uint32_t base, ns, nt, ni;
    if (!nvn2_cs_layout(ex, &base, &ns, &nt, &ni) || off < base) return -1;
    off -= base;
    if (off < 16 * ns) return *kind = 0, (int)(off / 16);
    off -= 16 * ns;
    if (off < 8 * nt) return *kind = 1, (int)(off / 8);
    off -= 8 * nt;
    if (off < 8 * ni) return *kind = 2, (int)(off / 8);
    return -1;
}

static int nvn2_cs(const VkExec *ex, int st) {   // an SM86 compute program is bound: its bank 0 is the NVN2 one
    const uint8_t *ctl, *code;
    uint32_t len;
    return st == NVN_STAGE_CS && ex->prog && vk_exec_find_stage(ex->prog, NVN_STAGE_CS, &ctl, &code, &len) && ctl[0x56] == 5;
}

// The buffers bound as storage (or read as indirect arguments) so far, by the ranges the game bound (overlapping ones
// merged), sorted: the places a pointer in memory can name. Not the backing's extent: one backing is 2.4 GiB, and a
// window entry is bound as written, so a mirrored pool copies the whole range back after every submission (the first
// run with whole backings fell to 1 fps). A stale entry costs nothing: gw_add resolves the address again.
#define GWR_N 1024
#define GW_MAX_ENTRY (64ull << 20)
#define GW_SCAN (16u << 10)   // bytes of each uniform block gw_select reads for pointers (CPU time per dispatch)
#define GW_SCAN_SB (4u << 10) // of each storage buffer: often write-combined memory, slow for the CPU to read
#define GW_SCAN_FRAMES 32     // a scan's result is kept this many frames (pointer tables rarely change)
static struct { uint64_t base, end; } GWR[GWR_N];
static uint32_t gwr_n;
static int gwr_find(uint64_t v) {   // index of the entry holding v, or -1
    uint32_t lo = 0, hi = gwr_n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (GWR[mid].end <= v) lo = mid + 1;
        else if (GWR[mid].base > v) hi = mid;
        else return (int)mid;
    }
    return -1;
}
static void gwr_note(uint64_t addr, uint64_t size) {   // size 0: unknown (up to 4 MiB of the backing)
    if (!addr || nvn_gpu_only(addr) || gwr_n == GWR_N) return;   // full: no churn on every binding
    int f = gwr_find(addr);
    if (f >= 0 && (size == 0 || addr + size <= GWR[f].end)) return;   // known
    VkMemSpan sp;
    if (!vk_mem_resolve(addr, 1, &sp) || !sp.pool) return;
    if (size == 0) size = 4u << 20;
    if (size > sp.avail) size = sp.avail;
    if (size > GW_MAX_ENTRY) size = GW_MAX_ENTRY;
    uint64_t b = addr, e = addr + size;
    uint32_t w = 0;   // drop the entries the new range overlaps, merged into it
    for (uint32_t i = 0; i < gwr_n; i++) {
        if (GWR[i].end > b && GWR[i].base < e) {
            if (GWR[i].base < b) b = GWR[i].base;
            if (GWR[i].end > e) e = GWR[i].end;
            continue;
        }
        GWR[w++] = GWR[i];
    }
    gwr_n = w;
    if (gwr_n == GWR_N || e - b > GW_MAX_ENTRY) return;
    uint32_t i = gwr_n;
    while (i > 0 && GWR[i - 1].base > b) GWR[i] = GWR[i - 1], i--;
    GWR[i].base = b, GWR[i].end = e, gwr_n++;
}
static void gw_add(uint64_t addr, uint64_t size) {
    VkMemSpan sp;
    if (gw_n >= VK_GW_N || !addr || nvn_gpu_only(addr) || !size || !vk_mem_resolve(addr, 1, &sp)) return;
    if (size > sp.avail) size = sp.avail;
    if (size > GW_MAX_ENTRY) size = GW_MAX_ENTRY;
    for (uint32_t k = 0; k < gw_n; k++)
        if (addr <= GW[k].addr + GW[k].size && GW[k].addr <= addr + size) {   // overlapping or adjacent: one entry
            uint64_t b = addr < GW[k].addr ? addr : GW[k].addr, e = addr + size > GW[k].addr + GW[k].size ? addr + size : GW[k].addr + GW[k].size;
            VkMemSpan bs;   // the union must stay in one backing: bank 0 must not claim more than the descriptor binds
            if (e - b <= GW_MAX_ENTRY && vk_mem_resolve(b, 1, &bs) && bs.avail >= e - b) {
                GW[k].addr = b, GW[k].size = e - b;
                return;
            }
            if (addr >= GW[k].addr && addr + size <= GW[k].addr + GW[k].size) return;   // covered
            continue;
        }
    GW[gw_n].addr = addr, GW[gw_n].size = size, gw_n++;
}

static void mwatch_gw(const VkExec *ex);

// What a scan of one buffer found: the registry ranges its 64-bit values point into. Kept per buffer for GW_SCAN_FRAMES
// frames: scanning per dispatch cost 28 ms a frame (the frame fell to 30 fps), most of it reading write-combined guest
// memory. A uniform block also takes the inline updates of the submission made since its last scan (gw_scan).
typedef struct {
    uint64_t addr, n, frame, fence;
    uint32_t upd_done;
    uint8_t nf, nraw;
    struct { uint64_t b, s; } f[VK_GW_N];   // the registry ranges found
    uint64_t raw[VK_GW_N];                  // the pointer values themselves (gw_select follows them one hop)
} GwScan;
static GwScan GWS[512];
static uint64_t gw_miss;   // OOT_TRACE=perf
static double gw_upd;
static inline double gw_now(void);
static int mwatch_on(void);
static void shader_hex(const GfxShader *s, char hex[17]);
static void gws_found(GwScan *c, uint64_t v) {
    if (v < 0x100000000ull || v >= RT_GPU_VA_BASE || c->nf >= VK_GW_N) return;
    int e = gwr_find(v);
    VkMemSpan sp;
    // Not bound as compute storage yet: e6d5432f's pointer table names buffers only pointers reach (and the indirect
    // mesh draws' arguments). Taken when the value is 16-byte aligned and its backing is at most 256 MiB, which keeps
    // out integer pairs such as 0x200000000 that land in the 1.4 GiB heap.
    if (e < 0 && !(v & 15) && vk_mem_resolve(v, 1, &sp) && sp.pool && sp.offset + sp.avail <= (256ull << 20)) {
        gwr_note(v, sp.avail);
        e = gwr_find(v);
    }
    if (e < 0) return;
    int known = 0;
    for (int i = 0; i < c->nraw && !known; i++) known = c->raw[i] == v;
    if (!known && c->nraw < VK_GW_N) c->raw[c->nraw++] = v;
    for (int i = 0; i < c->nf; i++)
        if (c->f[i].b == GWR[e].base) return;
    c->f[c->nf].b = GWR[e].base, c->f[c->nf].s = GWR[e].end - GWR[e].base, c->nf++;
}
static void gw_scan_updates(const VkExec *ex, GwScan *c);   // below UPD
static const GwScan *gw_scan(const VkExec *ex, uint64_t addr, uint64_t n, int ubo) {
    uint64_t frame = gfx_stats().frames, h = (addr ^ n << 48) * 0x9E3779B97F4A7C15ull;   // page-aligned addresses: mix, take the top bits
    h = ((h ^ h >> 29) * 0xBF58476D1CE4E5B9ull) >> 55;
    GwScan *c = &GWS[h];
    if (c->addr != addr || c->n != n || frame - c->frame >= GW_SCAN_FRAMES) {
        gw_miss++;
        memset(c, 0, sizeof *c);
        c->addr = addr, c->n = n, c->frame = frame;
        const uint64_t *q = ubo ? vk_usnap_peek(addr, n) : NULL;   // a uniform block as of the submit, else memory
        VkMemSpan sp;
        if (!q && vk_mem_resolve(addr, n, &sp)) q = (const uint64_t *)(uintptr_t)addr;
        for (uint64_t i = 0; q && i < n / 8 && c->nf < VK_GW_N; i++) gws_found(c, q[i]);
    }
    double t = gw_now();
    if (ubo) gw_scan_updates(ex, c);
    gw_upd += gw_now() - t;
    return c;
}

// The window's entries for the dispatch about to be bound, when its program has them: the buffers named by the 64-bit
// values in the storage buffers it reads, its uniform blocks and its pushed constants (each value is looked up among
// the buffers bound as storage so far), then its own storage buffers.
static void gw_select_inner(VkExec *ex, const VkPipeInfo *info);
static double gw_part[4];   // OOT_TRACE=perf: seconds in the parts of gw_select_inner
static inline double gw_now(void) {
    LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f); return (double)c.QuadPart / (double)f.QuadPart; }
static void gw_select(VkExec *ex, const VkPipeInfo *info) {
    gw_n = 0;
    // OOT_GW (default on since 2026-11-08, after the culling and the indirect draws it feeds worked in ~10 clean runs;
    // OOT_GW=0 turns it off): off, every access
    // through the window reads 0 and its stores are dropped, as before the window.
    static int on = -1;
    if (on < 0) on = rt_knob("GW", "1");
    if (!on) return;
    static int log_users = -1;   // OOT_GW_USERS=1 (diagnostic): each program that gets window entries, once
    if (log_users < 0) log_users = rt_knob("GW_USERS", "0");
    if (log_users) {
        gw_select_inner(ex, info);
        static char seen[256][17];
        static int nseen;
        char hx[17];
        shader_hex(C1CUR[NVN_STAGE_CS], hx);
        int known = 0;
        for (int i = 0; i < nseen && !known; i++) known = !strcmp(seen[i], hx);
        if (gw_n && !known && nseen < 256) {
            strcpy(seen[nseen++], hx);
            rt_log("vk_exec: OOT_GW_USERS: %s gets %u window entries (first at frame %llu)", hx, gw_n,
                   (unsigned long long)gfx_stats().frames);
        }
        return;
    }
    if (!(rt_trace & RT_TRACE_PERF)) return gw_select_inner(ex, info);
    static LARGE_INTEGER f;   // OOT_TRACE=perf: its CPU time, logged every 600 frames
    static double acc;
    static uint64_t calls, used, last;
    LARGE_INTEGER a, b;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&a);
    gw_select_inner(ex, info);
    QueryPerformanceCounter(&b);
    acc += (double)(b.QuadPart - a.QuadPart) / (double)f.QuadPart, calls++, used += gw_n != 0;
    uint64_t fr = gfx_stats().frames;
    if (fr >= last + 600) {
        double d = (double)(fr - last) / 1000.0;
        rt_log("perf: gw_select: %.3f ms a frame over %llu dispatches a frame, %llu with entries a frame (registry %u); parts %.3f %.3f %.3f %.3f ms",
               1000.0 * acc / (double)(fr - last), (unsigned long long)(calls / (fr - last)),
               (unsigned long long)(used / (fr - last)), gwr_n, gw_part[0] / d, gw_part[1] / d, gw_part[2] / d, gw_part[3] / d);
        rt_log("perf: gw_select: scan misses %llu a frame, inline-update scans %.3f ms a frame", (unsigned long long)(gw_miss / (fr - last)), gw_upd / d);
        gw_miss = 0, gw_upd = 0;
        acc = 0, calls = used = 0, last = fr, memset(gw_part, 0, sizeof gw_part);
    }
}
// The window for a task or mesh stage dispatch (mesh_dispatch; the stage's resources are in the compute slots then).
// 2026-11-08: the indirect draws' task program c7f0444a reads its meshlet list through 8 loads no trace reaches; with
// the window empty they read 0, its loop bound became 0x07ffffff and the GPU hung (run-135, found with OOT_SM86_COUNT).
// A stage with storage-held pointers gets it too: their table (SM86_PTR_C0_MESH) lies above the window.
static void gw_select_mesh(VkExec *ex, const VkPipeInfo *info) {
    gw_n = 0;
    static int on = -1;
    if (on < 0) on = rt_knob("GW", "1");
    if (!on) return;
    int uses = 0;
    for (uint32_t i = 0; i < info->n_bindings && !uses; i++) {
        const GfxShaderRes *r = info->bindings[i].res;
        uint32_t off = r ? (uint32_t)r->handle * 4 : 0;
        uses = info->bindings[i].kind == GFX_RES_SSBO && r && r->cbuf_slot == 0 && off >= VK_GW_C0 && off < VK_GW_C0 + 16 * VK_GW_N;
    }
    if (!uses) return;
    for (uint32_t k = 0; k < 16; k++) gwr_note(ex->ssbo[NVN_STAGE_CS][k].addr, ex->ssbo[NVN_STAGE_CS][k].size);
    uint64_t hop[64];
    int nhop = 0;
    // Only the uniform blocks the program reads can hand it a pointer: scanning all 18 for every stage of every mesh draw was 28% of the render
    // thread's active time by Saria (2026-11-09), as most blocks are new addresses each frame and so miss the scan cache.
    uint32_t ubo_used = 0;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const GfxShaderRes *r = info->bindings[i].res;
        if (info->bindings[i].kind == GFX_RES_UBO && r && r->cbuf_slot >= 3 && r->cbuf_slot - 3 < 18) ubo_used |= 1u << (r->cbuf_slot - 3);
    }
    for (uint32_t k = 0; k < 16 + 18 && gw_n < VK_GW_N; k++) {
        int ubo = k >= 16;
        if (ubo && !(ubo_used >> (k - 16) & 1)) continue;
        const p_bind_buf *b = ubo ? &ex->ubo[NVN_STAGE_CS][k - 16] : &ex->ssbo[NVN_STAGE_CS][k];
        if (!b->addr || (b->addr & 7) || nvn_gpu_only(b->addr)) continue;
        VkMemSpan sp;
        if (!vk_mem_resolve(b->addr, 8, &sp)) continue;
        uint64_t n = b->size ? b->size : sp.avail;
        if (n > sp.avail) n = sp.avail;
        if (n > (ubo ? GW_SCAN : GW_SCAN_SB)) n = ubo ? GW_SCAN : GW_SCAN_SB;
        const GwScan *c = gw_scan(ex, b->addr, n & ~7ull, ubo);
        for (int i = 0; i < c->nf && gw_n < VK_GW_N; i++) gw_add(c->f[i].b, c->f[i].s);
        for (int i = 0; i < c->nraw && nhop < 64; i++) hop[nhop++] = c->raw[i];
    }
    for (int h = 0; h < nhop && gw_n < VK_GW_N; h++) {
        VkMemSpan sp;
        if ((hop[h] & 7) || !vk_mem_resolve(hop[h], 8, &sp)) continue;
        const GwScan *c = gw_scan(ex, hop[h], (sp.avail < 256 ? sp.avail : 256) & ~7ull, 0);
        for (int i = 0; i < c->nf && gw_n < VK_GW_N; i++) gw_add(c->f[i].b, c->f[i].s);
    }
    for (uint32_t k = 0; k < 16 && gw_n < VK_GW_N; k++)
        if (ex->ssbo[NVN_STAGE_CS][k].addr)
            gw_add(ex->ssbo[NVN_STAGE_CS][k].addr, ex->ssbo[NVN_STAGE_CS][k].size ? ex->ssbo[NVN_STAGE_CS][k].size : 4u << 20);
    mwatch_gw(ex);
}

// OOT_GW_SKIP=<input id prefixes> (default none): programs that get no window. 374126bf and
// 358beb27 were skipped from 2026-11-08: their stores were traced to bank-3 words that hold floats (docs/phase3.md),
// and with the window the game crashed on a bad object index in 3 of 12 runs. Translator patch 0021 sends such stores
// to the window after the wrong candidate, and without the window these kernels never write their per-object state,
// so the shadow culling got no work (2026-11-09). OOT_GW_SKIP=374126bf,358beb27 restores the old default.
static int gw_skipped(void) {
    static char list[256];
    static int init;
    if (!init) {
        // The default used to list 18 programs, a workaround for the crash at main+0xc938d0, which came with their window.
        // Its cause was the slot reservation of the foliage kernels (lower_atomg), so nothing is skipped by default now.
        const char *e = getenv("OOT_GW_SKIP");
        snprintf(list, sizeof list, "%s", e && strcmp(e, "none") ? e : "");
        init = 1;
    }
    char hx[17];
    shader_hex(C1CUR[NVN_STAGE_CS], hx);
    for (const char *p = list; *p;) {
        size_t k = strcspn(p, ",");
        if (k >= 4 && k <= 16 && !strncmp(hx, p, k)) return 1;
        p += k + (p[k] == ',');
    }
    return 0;
}

static void gw_select_inner(VkExec *ex, const VkPipeInfo *info) {
    if (gw_skipped()) return;
    double t0 = gw_now(), t1;
    int uses = 0;
    uint32_t written = 0, base, ns, nt, ni;
    if (!nvn2_cs_layout(ex, &base, &ns, &nt, &ni)) return;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const GfxShaderRes *r = info->bindings[i].res;
        if (info->bindings[i].kind != GFX_RES_SSBO || !r || r->cbuf_slot != 0) continue;
        uint32_t off = (uint32_t)r->handle * 4;
        int kind, s;
        if (off >= VK_GW_C0 && off < VK_GW_C0 + 16 * VK_GW_N) uses = 1;
        else if ((s = nvn2_c0_slot(ex, off, &kind)) >= 0 && kind == 0 && s < 32 && (r->flags & 1)) written |= 1u << s;
    }
    t1 = gw_now(), gw_part[0] += t1 - t0, t0 = t1;
    if (!uses || base + 16 * ns + 8 * nt + 8 * ni > VK_GW_C0) return;
    for (uint32_t k = 0; k < 16 && k < ns; k++) gwr_note(ex->ssbo[NVN_STAGE_CS][k].addr, ex->ssbo[NVN_STAGE_CS][k].size);
    for (uint32_t i = 0; i < ex->push_cc_len / 8 && gw_n < VK_GW_N; i++) {   // the push
        uint64_t v;
        memcpy(&v, ex->push_cc + 8 * i, 8);
        int e = v >= 0x100000000ull && v < RT_GPU_VA_BASE ? gwr_find(v) : -1;
        if (e >= 0) gw_add(GWR[e].base, GWR[e].end - GWR[e].base);
    }
    t1 = gw_now(), gw_part[1] += t1 - t0, t0 = t1;
    uint64_t hop[64];
    int nhop = 0;
    // Its storage buffers that it only reads (the tables at their heads), then its uniform blocks: e6d5432f loads its
    // counters' pointers from an array in uniform block 1 at a run-time index (LDC.64 c[4][R]), which no
    // constant-offset trace reaches.
    for (uint32_t k = 0; k < 16 + 18 && gw_n < VK_GW_N; k++) {
        int ubo = k >= 16;
        const p_bind_buf *b = ubo ? &ex->ubo[NVN_STAGE_CS][k - 16] : &ex->ssbo[NVN_STAGE_CS][k];
        if ((!ubo && (k >= ns || (written >> k & 1))) || !b->addr || (b->addr & 7) || nvn_gpu_only(b->addr)) continue;
        VkMemSpan sp;
        if (!vk_mem_resolve(b->addr, 8, &sp)) continue;
        uint64_t n = b->size ? b->size : sp.avail;
        if (n > sp.avail) n = sp.avail;
        if (n > (ubo ? GW_SCAN : GW_SCAN_SB)) n = ubo ? GW_SCAN : GW_SCAN_SB;
        const GwScan *c = gw_scan(ex, b->addr, n & ~7ull, ubo);
        for (int i = 0; i < c->nf && gw_n < VK_GW_N; i++) gw_add(c->f[i].b, c->f[i].s);
        for (int i = 0; i < c->nraw && nhop < 64; i++) hop[nhop++] = c->raw[i];
    }
    // One more hop: small entries found so far are pointer records themselves. e6d5432f reads a record's address from
    // its uniform block, a pointer from the record, and its object count at that pointer + 4 (threads past it exit).
    for (int h = 0; h < nhop && gw_n < VK_GW_N; h++) {   // 256 bytes at each pointer found above
        VkMemSpan sp;
        if ((hop[h] & 7) || !vk_mem_resolve(hop[h], 8, &sp)) continue;
        const GwScan *c = gw_scan(ex, hop[h], (sp.avail < 256 ? sp.avail : 256) & ~7ull, 0);
        for (int i = 0; i < c->nf && gw_n < VK_GW_N; i++) gw_add(c->f[i].b, c->f[i].s);
    }
    t1 = gw_now(), gw_part[2] += t1 - t0, t0 = t1;
    for (uint32_t k = 0; k < 16 && k < ns && gw_n < VK_GW_N; k++)
        gw_add(ex->ssbo[NVN_STAGE_CS][k].addr, ex->ssbo[NVN_STAGE_CS][k].size ? ex->ssbo[NVN_STAGE_CS][k].size : 4u << 20);
    mwatch_gw(ex);
    gw_part[3] += gw_now() - t0;
    // OOT_GW_TRACE=<input id prefix> (diagnostic): for that kernel, every 300 frames, what its first checks read on the
    // CPU side: uniform block 0 at 0x650..0x6f0 (e6d5432f: per-type thresholds and the count its threads stop at) and
    // each pointer in uniform block 1 followed twice, with the word at + 4 (the per-type object count).
    const char *tr = ENV_ONCE("OOT_GW_TRACE");
    char hex[17];
    shader_hex(C1CUR[NVN_STAGE_CS], hex);
    static uint64_t last_tr;
    uint64_t fr = gfx_stats().frames;
    if (tr && *tr && !strncmp(hex, tr, strlen(tr)) && fr >= last_tr + 300) {
        last_tr = fr;
        const p_bind_buf *u0 = &ex->ubo[NVN_STAGE_CS][0], *u1 = &ex->ubo[NVN_STAGE_CS][1];
        uint32_t w[17] = {0};
        const uint8_t *s0 = u0->addr && u0->size >= 0x6f4 ? vk_usnap_peek(u0->addr, 0x6f4) : NULL;
        if (s0) memcpy(w, s0 + 0x650, sizeof w);
        rt_log("gwtrace: f%llu %s ubo0 %#llx+%#llx (%s) 0x650..0x6f0: %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u",
               (unsigned long long)fr, hex, (unsigned long long)u0->addr, (unsigned long long)u0->size, s0 ? "snapshot" : "none",
               w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15], w[16]);
        const uint8_t *s1 = u1->addr && u1->size ? vk_usnap_peek(u1->addr, u1->size < 0x200 ? u1->size : 0x200) : NULL;
        for (uint32_t i = 0; s1 && i < (u1->size < 0x200 ? u1->size : 0x200) / 8; i++) {
            uint64_t p1, p2 = 0, v2 = 0;
            memcpy(&p1, s1 + 8 * i, 8);
            VkMemSpan sp;
            if (p1 < 0x100000000ull || p1 >= RT_GPU_VA_BASE || !vk_mem_resolve(p1, 8, &sp)) continue;
            memcpy(&p2, (const void *)(uintptr_t)p1, 8);
            if (p2 >= 0x100000000ull && p2 < RT_GPU_VA_BASE && vk_mem_resolve(p2, 8, &sp)) memcpy(&v2, (const void *)(uintptr_t)p2, 8);
            int in1 = 0, in2 = 0;
            for (uint32_t k = 0; k < gw_n; k++) {
                in1 |= p1 >= GW[k].addr && p1 < GW[k].addr + GW[k].size;
                in2 |= p2 >= GW[k].addr && p2 < GW[k].addr + GW[k].size;
            }
            rt_log("gwtrace:   ubo1[%#x] %#llx%s -> %#llx%s -> words %u %u", 8 * i, (unsigned long long)p1, in1 ? " (window)" : "",
                   (unsigned long long)p2, in2 ? " (window)" : "", (uint32_t)v2, (uint32_t)(v2 >> 32));
        }
    }
}

// c0 from replay scratch; a bank equal to the stage's previous one reuses its range, so it costs no copy.
static VkDescriptorBufferInfo c0_info(VkExec *ex, int st) {
    int k = st == NVN_STAGE_VS ? 0 : st == NVN_STAGE_FS ? 1 : 2;
    // vk_c0_fill reads the storage slots, and the uniform slots and dispatch size for compute or the base instance for
    // graphics: equal inputs give the same bank, so neither the fill nor the 2 KiB compare is needed.
    typeof(B.c0_in[0]) in;
    memset(&in, 0, sizeof in);
    memcpy(in.ssbo, ex->ssbo[st], sizeof in.ssbo);
    if (st == NVN_STAGE_CS) memcpy(in.ubo, ex->ubo[st], sizeof in.ubo), memcpy(in.dim, c0_dim, sizeof in.dim);
    else in.binst = c0_binst;
    int nvn2 = nvn2_cs(ex, st);   // its bank also depends on the push and the handles: compared by bytes below
    if (!nvn2 && B.c0_valid[k] && !memcmp(&in, &B.c0_in[k], sizeof in))
        return (VkDescriptorBufferInfo){B.c0_buf[k], B.c0_off[k], VK_C0_BYTES};
    B.c0_in[k] = in;
    uint8_t tmp[VK_C0_BYTES];
    if (nvn2) nvn2_c0_fill_cs(ex, tmp);
    else vk_c0_fill(tmp, st, ex->ssbo[st], ex->ubo[st], c0_dim, c0_binst);
    if (M.on && st == NVN_STAGE_CS) mesh_c0(ex, tmp);   // op_draw_mesh clears c0_valid, so no stale bank is reused
    if (!B.c0_valid[k] || memcmp(tmp, B.c0[k], VK_C0_BYTES)) {
        memcpy(scratch(ex, VK_C0_BYTES, &B.c0_buf[k], &B.c0_off[k]), tmp, VK_C0_BYTES);
        memcpy(B.c0[k], tmp, VK_C0_BYTES);
        B.c0_valid[k] = 1;
    }
    return (VkDescriptorBufferInfo){B.c0_buf[k], B.c0_off[k], VK_C0_BYTES};
}

// The buffer range limits, read once: the device's properties are a cold cache line on every binding of every draw.
static struct { uint64_t ubo_max, ubo_align, sbo_max, sbo_align; } LIM;

static void limits_init(void) {
    const VkPhysicalDeviceLimits *l = &vk_ctx()->props.limits;
    LIM.ubo_align = l->minUniformBufferOffsetAlignment, LIM.sbo_max = l->maxStorageBufferRange;
    LIM.sbo_align = l->minStorageBufferOffsetAlignment;
    __atomic_store_n(&LIM.ubo_max, l->maxUniformBufferRange, __ATOMIC_RELEASE);
}

static VkDescriptorBufferInfo ubo_info(VkExec *ex, int st, int slot) {
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    uint64_t max = LIM.ubo_max, align = LIM.ubo_align;
    VkDescriptorBufferInfo bi;
    if (slot == 0 && c0_on()) return c0_info(ex, st);
    if (slot == 1 && ex->prog) {
        // The compiler's constant bank 1 (the jump tables, in the code block): the lowered program's copy, whose
        // branch targets were moved with the code (gfx_shader_c1), uploaded once per shader and stage.
        const GfxShader *s = C1CUR[st];
        uint32_t n = 0;
        const uint8_t *c1 = s ? gfx_shader_c1(s, &n) : NULL;
        if (c1 && n) {
            int k = st == NVN_STAGE_VS ? 0 : st == NVN_STAGE_FS ? 1 : 2;
            if (B.c1_of[k] != s || !B.c1_buf[k]) {
                uint64_t size = (n + 255) & ~255ull;
                uint8_t *dst = scratch(ex, size, &B.c1_buf[k], &B.c1_off[k]);
                memcpy(dst, c1, n), memset(dst + n, 0, size - n);
                B.c1_of[k] = s, B.c1_size[k] = size;
            }
            return (VkDescriptorBufferInfo){B.c1_buf[k], B.c1_off[k], B.c1_size[k]};
        }
    } else if (slot >= 3 && slot - 3 < 18) {
        const p_bind_buf *b = &ex->ubo[st][slot - 3];
        if (b->size && vk_usnap_find(b->addr, b->size < max ? b->size : max, align, &bi)) {   // as of the submit call
            count_use(VK_MCLS_SCRATCH, VK_MUSE_UBO, bi.range);
            return bi;
        }
        if (bind_range(ex, b->addr, b->size, align, max, VK_MUSE_UBO, 0, &bi)) return bi;
    }
    return zero_info(max);
}

// The NVN slot of a storage binding. Ryujinx numbers a shader's storage buffers in the order it meets them
// (GfxShaderRes.slot); the NVN slot is the entry of the driver constant buffer c0 that holds the buffer's address
// (cbuf_slot 0, handle = word offset): 16-byte entries at byte 0x110 (vertex), 0x510 (fragment) and 0x310 (compute).
// A compute address at c0 0x210..0x30f is uniform block (offset - 0x210) / 16 read as storage: *ubo is set. -1: the
// address is not in c0 (the shader loads it from a user uniform block: ubo_ptr) or the stage has no known table.
// Mapping: MacNativeGames games/totk/tools/totk_xlate/Service.cs:206-222 @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou).
static int ssbo_slot(const GfxShaderRes *r, int st, int *ubo) {
    *ubo = 0;
    if (r->cbuf_slot != 0) return -1;
    if (M.on && st == NVN_STAGE_CS) {   // op_draw_mesh: the task or mesh stage's own table
        uint32_t o = (uint32_t)r->handle * 4;
        if (o >= M.table && o < M.table + 0x100 && (o - M.table) % 16 == 0 && o + 16 <= VK_C0_BYTES) return (int)((o - M.table) / 16);
    }
    int off = r->handle * 4, base = st == NVN_STAGE_VS ? 0x110 : st == NVN_STAGE_FS ? 0x510 : st == NVN_STAGE_CS ? 0x310 : -1;
    if (st == NVN_STAGE_CS && off >= 0x210 && off < 0x310 && (off - 0x210) % 16 == 0) {
        *ubo = 1;
        return (off - 0x210) / 16;
    }
    return base >= 0 && off >= base && off < base + 0x100 && (off - base) % 16 == 0 ? (off - base) / 16 : -1;
}

// 1 when the shader loads the storage buffer's address from a user uniform block (cbuf_slot 3.., NVN uniform slot
// cbuf_slot - 3, byte offset handle * 4). The Depths terrain reads its vertices through one, about 500 draws a frame;
// with a zero buffer they collapse and the ground is missing. MacNativeGames resolves these the same way (NVN_RB_SBUFU,
// games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m @ b75be8cf).
static int ubo_ptr_res(const GfxShaderRes *r) {
    return r->cbuf_slot >= 3 && r->cbuf_slot - 3 < 18;
}

// The inline uniform updates of the submission being replayed, newest last (a ring of UPD_N). op_update_ub writes them
// into the buffer's GPU copy; guest memory gets them only from the write-back when the submission ends. The terrain
// rewrites its pointer block this way before every draw, all at one address, so ubo_ptr reads the latest update that
// covers the pointer, else memory. Payloads stay in the recorded command buffer for the whole replay.
#define UPD_N 4096
static struct {
    uint64_t fence;                      // the submission the entries belong to
    uint32_t n;                          // updates noted in it
    struct { uint64_t lo, hi; const uint8_t *data; } u[UPD_N];
} UPD;

static void upd_note(const VkExec *ex, const p_update_ub *u) {
    if (UPD.fence != ex->fence) UPD.fence = ex->fence, UPD.n = 0;
    UPD.u[UPD.n++ % UPD_N] = (typeof(UPD.u[0])){u->dst, u->dst + u->size, u->data};
}

// 1 when an update of this submission supplied [a, a+n).
static int upd_read(const VkExec *ex, uint64_t a, void *out, size_t n) {
    if (UPD.fence != ex->fence) return 0;
    for (uint32_t i = UPD.n, end = UPD.n > UPD_N ? UPD.n - UPD_N : 0; i-- > end;) {
        const typeof(UPD.u[0]) *e = &UPD.u[i % UPD_N];
        if (a >= e->hi || a + n <= e->lo) continue;
        if (a < e->lo || a + n > e->hi) return 0;   // partly covered: memory
        memcpy(out, e->data + (a - e->lo), n);
        return 1;
    }
    return 0;
}

// gw_scan: the 64-bit values the inline updates of this submission made since the block's last scan wrote into it.
static void gw_scan_updates(const VkExec *ex, GwScan *c) {
    if (UPD.fence != ex->fence) return;
    if (c->fence != ex->fence) c->fence = ex->fence, c->upd_done = 0;
    uint32_t from = c->upd_done, end = UPD.n;
    if (end - from > UPD_N) from = end - UPD_N;
    for (uint32_t i = from; i < end && c->nf < VK_GW_N; i++) {
        const typeof(UPD.u[0]) *e = &UPD.u[i % UPD_N];
        uint64_t lo = e->lo > c->addr ? e->lo : c->addr, hi = e->hi < c->addr + c->n ? e->hi : c->addr + c->n;
        for (uint64_t p = (lo + 7) & ~7ull; p + 8 <= hi && c->nf < VK_GW_N; p += 8) {
            uint64_t v;
            memcpy(&v, e->data + (p - e->lo), 8);
            gws_found(c, v);
        }
    }
    c->upd_done = end;
}

// That address as the bound block holds it at this point of the replay; 0 when the block is unbound or too short. The
// block has no size field.
static uint64_t ubo_ptr(const VkExec *ex, int st, const GfxShaderRes *r) {
    const p_bind_buf *u = &ex->ubo[st][r->cbuf_slot - 3];
    uint64_t off = (uint64_t)r->handle * 4, a = 0;
    VkMemSpan sp;
    if (!u->addr || u->size < off + 8 || upd_read(ex, u->addr + off, &a, 8)) return a;
    if (vk_mem_resolve(u->addr + off, 8, &sp)) memcpy(&a, (const void *)(uintptr_t)(u->addr + off), 8);
    return a;
}

// ---- storage-held pointers (sm86_lower.h SM86_PTR_C0)
// The lowered program's table at the end of its c1: entry i names the bank-0 offset of a storage buffer's address
// (desc) and an offset into that buffer, where the program reads a 64-bit pointer it then loads through. The
// pointer goes into c0[SM86_PTR_C0 + 8 i] (nvn2_c0_fill_cs), and a storage binding whose address the translator
// traced to that word is bound at the pointer, up to the end of its backing (ssbo_info), as ubo_ptr does for
// pointers in uniform blocks. The pointer is read as the submission left it (an inline update) or from memory.
static const uint8_t *ptr_table(int st, uint32_t *count) {
    const GfxShader *s = st >= 0 && st < 6 ? C1CUR[st] : NULL;
    uint32_t n = 0, tail[2];
    const uint8_t *c1 = s ? gfx_shader_c1(s, &n) : NULL;
    *count = 0;
    if (!c1 || n < 8) return NULL;
    memcpy(tail, c1 + n - 8, 8);
    if (tail[1] != SM86_PTR_MAGIC || tail[0] == 0 || tail[0] > SM86_PTR_MAX || 8 + 8 * tail[0] > n) return NULL;
    *count = tail[0];
    return c1 + n - 8 - 8 * tail[0];
}

static uint32_t ptr_count(int st) {
    uint32_t n;
    ptr_table(st, &n);
    return n;
}

static uint64_t ptr_value(const VkExec *ex, int st, uint32_t i) {
    uint32_t n, e[2];
    const uint8_t *t = ptr_table(st, &n);
    if (!t || i >= n) return 0;
    memcpy(e, t + 8 * i, 8);
    int kind = -1, s;
    if (M.on && st == NVN_STAGE_CS)   // op_draw_mesh: the stage's own table
        s = e[0] >= M.table && (e[0] - M.table) % 16 == 0 ? (int)((e[0] - M.table) / 16) : -1, kind = 0;
    else s = nvn2_c0_slot(ex, e[0], &kind);
    if (s < 0 || kind != 0 || s >= 16) return 0;
    const p_bind_buf *b = &ex->ssbo[st][s];
    uint64_t a = b->addr + e[1], v = 0;
    VkMemSpan sp;
    int how = 0;
    if (!b->addr || (b->size && b->size < (uint64_t)e[1] + 8)) how = 3;
    else if (upd_read(ex, a, &v, 8)) how = 1;
    else if (vk_mem_resolve(a, 8, &sp)) memcpy(&v, (const void *)(uintptr_t)a, 8), how = 2;
    static uint64_t seen[64];   // log each distinct (place, pointer) once
    static int nseen;
    uint64_t key = (a * 0x9E3779B97F4A7C15ull) ^ v;
    int known = 0;
    for (int k = 0; k < nseen && !known; k++) known = seen[k] == key;
    if (!known && nseen < 64) {
        seen[nseen++] = key;
        rt_log("vk_exec: storage-held pointer %u: c0+%#x slot %d (%#llx+%#llx) at +%#x -> %#llx (%s)", i, e[0], s,
               (unsigned long long)b->addr, (unsigned long long)b->size, e[1], (unsigned long long)v,
               how == 1 ? "inline update" : how == 2 ? "memory" : how == 3 ? "out of the buffer" : "unmapped");
    }
    return v;
}

// The pointer entry a bank-0 storage binding of an NVN2 compute program names; -1 for the others.
static int ptr_res(const VkExec *ex, int st, const GfxShaderRes *r) {
    int mesh = M.on && st == NVN_STAGE_CS;
    uint32_t off = (uint32_t)r->handle * 4, pb = mesh ? SM86_PTR_C0_MESH : SM86_PTR_C0;
    if (r->cbuf_slot != 0 || off < pb || off >= pb + 8 * (mesh ? SM86_PTR_MESH_MAX : SM86_PTR_MAX) || (off - pb) % 8) return -1;
    return (int)((off - pb) / 8) < (int)ptr_count(st) && (M.on || nvn2_cs(ex, st)) ? (int)((off - pb) / 8) : -1;
}

// OOT_MESH_WATCH=1 (diagnostic): who writes the indirect mesh draws' arguments. op_draw_mesh adds each argument
// address to the watch list; a written storage binding that covers one, and a buffer fill over one, is logged with
// the kernel, as is each written binding that has no backing (the discard buffer), once per kernel and bank offset.
static uint64_t mwatch[64];
static volatile LONG nmwatch;
static int mwatch_on(void) {
    static int on = -1;
    if (on < 0) on = getenv("OOT_MESH_WATCH") != NULL;
    return on;
}
static void mwatch_add(uint64_t a) {
    if (!mwatch_on()) return;
    LONG n = nmwatch;
    for (LONG i = 0; i < n; i++) if (mwatch[i] == a) return;
    if (n < 64) mwatch[n] = a, nmwatch = n + 1;
}
static void shader_hex(const GfxShader *s, char hex[17]) {
    strcpy(hex, "?");
    if (!s) return;
    const uint8_t *id = gfx_shader_input_id(s);
    for (int i = 0; i < 8; i++) snprintf(hex + 2 * i, 3, "%02x", id[i]);
}
// OOT_WATCH_RANGE=<lo hex>:<hi hex> (diagnostic): every GPU write that may land in [lo, hi) (written storage binding,
// window entry, copy, fill) is logged once per writer, kernel and bank offset (2026-11-08: the game crashes on an object
// record in 0x29bc00000.. with OOT_GW=1).
static void wrange_check(const char *what, int st, uint32_t off, uint64_t addr, uint64_t size) {
    static int init, nr;
    static uint64_t lo[8], hi[8];
    if (!init) {   // <lo>:<hi>[,<lo>:<hi>...]
        const char *e = getenv("OOT_WATCH_RANGE");
        init = 1;
        for (; e && *e && nr < 8; e = strchr(e, ',') ? strchr(e, ',') + 1 : "")
            if (sscanf(e, "%llx:%llx", (unsigned long long *)&lo[nr], (unsigned long long *)&hi[nr]) == 2) nr++;
    }
    int hit = 0;
    for (int i = 0; i < nr && !hit; i++) hit = addr && addr < hi[i] && (!size || addr + size > lo[i]);
    if (!hit) return;
    static uint64_t seen[512];
    static int nseen;
    static volatile LONG lines;
    char hex[17];
    shader_hex(st >= 0 && st < 6 ? C1CUR[st] : NULL, hex);
    uint64_t key = (addr * 0x9E3779B97F4A7C15ull) ^ ((st >= 0 && st < 6) ? (uintptr_t)C1CUR[st] : 0) ^ off ^ (uintptr_t)what;
    for (int k = 0; k < nseen; k++) if (seen[k] == key) return;
    if (nseen < 512) seen[nseen++] = key;
    if (InterlockedIncrement(&lines) <= 300)
        rt_log("wrange: f%llu %s by %s (stage %d, c0+%#x) at %#llx+%#llx", (unsigned long long)gfx_stats().frames, what, hex,
               st, off, (unsigned long long)addr, (unsigned long long)size);
}
void vk_exec_mesh_watch(const char *what, int st, uint32_t off, uint64_t addr, uint64_t size) {   // also vk_exec_xfer.c clear_buf
    wrange_check(what, st, off, addr, size);
    if (!mwatch_on()) return;
    static uint64_t seen[256];
    static int nseen;
    static volatile LONG lines;
    char hex[17];
    shader_hex(st >= 0 && st < 6 ? C1CUR[st] : NULL, hex);
    for (LONG i = 0; i < nmwatch; i++) {
        uint64_t w = mwatch[i];
        if (addr && w >= addr && (size == 0 || w < addr + size)) {
            uint64_t key = (w * 0x9E3779B97F4A7C15ull) ^ (st >= 0 && st < 6 ? (uintptr_t)C1CUR[st] : 0) ^ off;
            int known = 0;
            for (int k = 0; k < nseen && !known; k++) known = seen[k] == key;
            if (known) continue;
            if (nseen < 256) seen[nseen++] = key;
            if (InterlockedIncrement(&lines) <= 200)
                rt_log("meshwatch: f%llu %s by %s (stage %d, c0+%#x) at %#llx+%#llx covers indirect args %#llx",
                       (unsigned long long)gfx_stats().frames, what, hex, st, off, (unsigned long long)addr,
                       (unsigned long long)size, (unsigned long long)w);
        }
    }
}
// OOT_MESH_WATCH=<frame> (above 1): every dispatch of that frame is logged in order with its storage slots.
static void mwatch_seq(const VkExec *ex, const GfxShader *cs, const char *dims) {
    static long long at = -1;
    if (at < 0) { const char *e = getenv("OOT_MESH_WATCH"); at = e ? atoll(e) : 0; }
    if (at <= 1 || (long long)gfx_stats().frames != at) return;
    char hex[17], s[1200];
    size_t o = 0;
    shader_hex(cs, hex);
    for (int k = 0; k < 16 && o < sizeof s - 60; k++) {
        const p_bind_buf *b = &ex->ssbo[NVN_STAGE_CS][k];
        if (!b->addr) continue;
        VkMemSpan sp;
        uint64_t pb = 0, pe = 0;
        if (vk_mem_resolve(b->addr, 1, &sp)) pb = b->addr - sp.offset, pe = b->addr + sp.avail;
        o += (size_t)snprintf(s + o, sizeof s - o, " %d=%#llx+%#llx[pool %#llx-%#llx%s]", k, (unsigned long long)b->addr,
                              (unsigned long long)b->size, (unsigned long long)pb, (unsigned long long)pe, sp.pool ? "" : " dev");
    }
    s[o] = 0;
    rt_log("meshseq: %s %s:%s", hex, dims, s);
}

// OOT_MESH_WATCH: the global window's entries (gw_select), per kernel at its first dispatch with any and when their
// count changes; and every dispatch of the OOT_MESH_WATCH frame.
static void mwatch_gw(const VkExec *ex) {
    (void)ex;
    if (!mwatch_on()) return;
    static struct { const void *cs; uint32_t n; } seen[256];
    static int nseen;
    const void *cs = C1CUR[NVN_STAGE_CS];
    static long long at = -1;
    if (at < 0) { const char *e = getenv("OOT_MESH_WATCH"); at = e ? atoll(e) : 0; }
    int i = 0;
    while (i < nseen && seen[i].cs != cs) i++;
    if (i < nseen && seen[i].n == gw_n && (long long)gfx_stats().frames != at) return;
    if (i == nseen && nseen < 256) nseen++;
    if (i < 256) seen[i].cs = cs, seen[i].n = gw_n;
    char hex[17], s[900];
    size_t o = 0;
    shader_hex(cs, hex);
    for (uint32_t k = 0; k < gw_n && o < sizeof s - 40; k++)
        o += (size_t)snprintf(s + o, sizeof s - o, " %#llx+%#llx", (unsigned long long)GW[k].addr, (unsigned long long)GW[k].size);
    s[o] = 0;
    rt_log("meshwatch: f%llu global window of %s: %u entries:%s (registry %u)", (unsigned long long)gfx_stats().frames, hex, gw_n, s, gwr_n);
}

static uint64_t ubo_ptr(const VkExec *ex, int st, const GfxShaderRes *r);
static void mwatch_discard(const VkExec *ex, int st, const GfxShaderRes *r) {
    if (!mwatch_on()) return;
    static uint64_t seen[256];
    static int nseen;
    uint64_t key = (st >= 0 && st < 6 ? (uintptr_t)C1CUR[st] : 0) ^ ((uint64_t)r->cbuf_slot << 40) ^ (uint64_t)r->handle * 4;
    for (int k = 0; k < nseen; k++) if (seen[k] == key) return;
    if (nseen >= 256) return;
    seen[nseen++] = key;
    char hex[17];
    shader_hex(st >= 0 && st < 6 ? C1CUR[st] : NULL, hex);
    char why[200] = "";
    uint32_t off = (uint32_t)r->handle * 4, base = 0, ns = 0, nt = 0, ni = 0;
    if (r->cbuf_slot >= 3 && r->cbuf_slot - 3 < 18) {
        const p_bind_buf *u = &ex->ubo[st][r->cbuf_slot - 3];
        snprintf(why, sizeof why, "uniform slot %d = %#llx+%#llx, pointer %#llx", r->cbuf_slot - 3, (unsigned long long)u->addr,
                 (unsigned long long)u->size, (unsigned long long)ubo_ptr(ex, st, r));
    } else if (r->cbuf_slot == 0 && st == NVN_STAGE_CS && nvn2_cs_layout(ex, &base, &ns, &nt, &ni)) {
        uint64_t v = 0;
        if (off >= 0x10 && off + 8 <= 0x10 + ex->push_cc_len) memcpy(&v, ex->push_cc + off - 0x10, 8);
        snprintf(why, sizeof why, "tables at %#x (%u storage, %u textures, %u images), push %#x bytes, pushed value %#llx", base,
                 ns, nt, ni, ex->push_cc_len, (unsigned long long)v);
    }
    rt_log("meshwatch: f%llu written storage of %s (stage %d%s) at c%d+%#x has no backing: discarded; %s",
           (unsigned long long)gfx_stats().frames, hex, st, M.on ? ", mesh draw" : "", r->cbuf_slot, off, why);
}

// OOT_SM86_COUNT (diagnostic, sm86_lower.h): the counter buffer the probed program's counts go to, host-visible and
// coherent, read as it is (the counts only grow) and logged every 120 frames with the deltas since the last log and
// the number of dispatches that bound it.
static struct { int on; VkBuffer buf; VmaAllocation mem; volatile uint32_t *map; uint32_t last[SM86_COUNT_MAX]; uint64_t binds, frame; } CNT;
static int cnt_on(void) {
    if (CNT.on) return CNT.on > 0;
    uint32_t n;
    CNT.on = -1;
    if (!sm86_count_points(&n)) return 0;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4 * SM86_COUNT_MAX,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                  .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &CNT.buf, &CNT.mem, &info) != VK_SUCCESS) return 0;
    CNT.map = info.pMappedData;
    memset((void *)CNT.map, 0, 4 * SM86_COUNT_MAX);
    rt_log("count: OOT_SM86_COUNT on, %u points", n);
    return CNT.on = 1;
}
// On a device loss (vk_device.c): the counters as the GPU left them (host-coherent memory outlives the device), so
// a hang shows where its threads were.
void vk_exec_count_dump(void) {
    if (CNT.on <= 0 || !CNT.map) return;
    uint32_t n;
    const char (*at)[16] = sm86_count_points(&n);
    char line[64 * 40], *o = line;
    for (uint32_t k = 0; k < n && o < line + sizeof line - 40; k++) o += sprintf(o, " @%s:%u", at[k], CNT.map[k] - CNT.last[k]);
    rt_log("count: at the device loss, since f%llu: %s", (unsigned long long)CNT.frame, line);
}
static void cnt_log(void) {
    if (!cnt_on()) return;
    uint64_t fr = gfx_stats().frames;
    if (fr < CNT.frame + 120) return;
    uint32_t n;
    const char (*at)[16] = sm86_count_points(&n);
    char line[64 * 40], *o = line;
    for (uint32_t k = 0; k < n && o < line + sizeof line - 40; k++) {
        uint32_t v = CNT.map[k];
        o += sprintf(o, " @%s:%u", at[k], v - CNT.last[k]);
        CNT.last[k] = v;
    }
    rt_log("count: f%llu..%llu, %llu dispatches bound it:%s", (unsigned long long)CNT.frame, (unsigned long long)fr,
           (unsigned long long)CNT.binds, line);
    CNT.frame = fr, CNT.binds = 0;
}

static VkDescriptorBufferInfo ssbo_info(VkExec *ex, int st, const GfxShaderRes *r, int written) {
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    if (st == NVN_STAGE_CS && r->cbuf_slot == 0 && (uint32_t)r->handle * 4 == SM86_COUNT_C0 && cnt_on()) {
        CNT.binds++;
        return (VkDescriptorBufferInfo){CNT.buf, 0, 4 * SM86_COUNT_MAX};
    }
    uint64_t max = LIM.sbo_max, align = LIM.sbo_align;
    VkDescriptorBufferInfo bi;
    int ubo, slot = ssbo_slot(r, st, &ubo);
    p_bind_buf up = {0};
    const p_bind_buf *b = NULL;
    int pi, gw = 0;
    if (M.drawing && st == NVN_STAGE_VS && r->cbuf_slot == 0 && (uint32_t)r->handle * 4 == SM86_MESH_VS_TABLE)
        return (VkDescriptorBufferInfo){M.buf, M.task_size, M.mesh_size};   // mesh_draw: the mesh records
    if (M.on && st == NVN_STAGE_CS && r->cbuf_slot == 0) {   // op_draw_mesh: the records
        uint32_t off = (uint32_t)r->handle * 4;
        if (off == SM86_MESH_C0_TASK) return (VkDescriptorBufferInfo){M.buf, 0, M.task_size};
        if (off == SM86_MESH_C0_MESH) return (VkDescriptorBufferInfo){M.buf, M.task_size, M.mesh_size};
        if (off >= VK_GW_C0 && off < VK_GW_C0 + 16 * VK_GW_N) {   // the window (gw_select_mesh), else zeros
            uint32_t k = (off - VK_GW_C0) / 16;
            if (k >= gw_n || (off - VK_GW_C0) % 16)
                return written ? (VkDescriptorBufferInfo){Z.dbuf, 0, max < ZERO_BYTES ? max : ZERO_BYTES} : zero_info(max);
            up.addr = GW[k].addr, up.size = GW[k].size, b = &up, gw = 1;
        }
    }
    if (b) {   // a window entry of a task or mesh stage, chosen above
    } else if (slot >= 0 && slot < 16) b = ubo ? &ex->ubo[st][slot] : &ex->ssbo[st][slot];
    else if (ubo_ptr_res(r)) up.addr = ubo_ptr(ex, st, r), b = &up;   // size 0: up to the end of the backing
    else if ((pi = ptr_res(ex, st, r)) >= 0) up.addr = ptr_value(ex, st, (uint32_t)pi), b = &up;   // the same
    else if (r->cbuf_slot == 0 && nvn2_cs(ex, st)) {                  // NVN2: the slot the bank-0 entry belongs to
        int kind, s2 = nvn2_c0_slot(ex, (uint32_t)r->handle * 4, &kind);
        uint32_t off = (uint32_t)r->handle * 4, base, ns, nt, ni;
        if (s2 < 0 && off >= VK_GW_C0 && off < VK_GW_C0 + 16 * VK_GW_N) {   // the global window (gw_select)
            uint32_t k = (off - VK_GW_C0) / 16;
            if (k >= gw_n || (off - VK_GW_C0) % 16)
                return written ? (VkDescriptorBufferInfo){Z.dbuf, 0, max < ZERO_BYTES ? max : ZERO_BYTES} : zero_info(max);
            up.addr = GW[k].addr, up.size = GW[k].size, b = &up, gw = 1;
        } else if (s2 >= 0 && kind == 0 && s2 < 16) b = &ex->ssbo[st][s2];
        else if (off >= 0x10 && off + 8 <= 0x10 + ex->push_cc_len && nvn2_cs_layout(ex, &base, &ns, &nt, &ni) && off + 8 <= base) {
            // a pointer the game pushed with the constants (PushComputeConstants), before the tables: the buffer at it,
            // to the end of its backing (2026-11-07: ace1cec0 loops over a list at the pointer in its push at 0x60;
            // bound to the zero buffer before)
            memcpy(&up.addr, ex->push_cc + off - 0x10, 8);
            b = &up;
        }
    }
    if (b) {
        // vk_caps.c: shaders translated for sbuf_align address the buffer from its address rounded down to it.
        uint64_t lead = vk_caps.sbuf_align ? b->addr & (vk_caps.sbuf_align - 1) : 0;
        // A window entry is bound as written for every store the program might make through it, so its whole range
        // would go back to guest memory after the submission (a mirrored pool) and come back up as dirty pages: 70 MiB
        // each way per frame, the frame CPU-bound at 28 fps. The GPU's copy stays the one in use; the CPU does not read
        // these (counters and lists the GPU builds), and what it writes there still goes up.
        if (bind_range(ex, b->addr - lead, b->size ? b->size + lead : 0, align, max, ubo ? VK_MUSE_UBO : VK_MUSE_SSBO,
                       written && !gw, &bi)) {
            if (st == NVN_STAGE_CS && !gw) gwr_note(b->addr, b->size);   // a place pointers in memory can name (gw_select)
            if (gw && ENV_ONCE("OOT_GW_TRACE")) {   // the window binding as made (diagnostic)
                static uint64_t lastf;
                static int nthis;
                uint64_t fr = gfx_stats().frames;
                char hx[17];
                shader_hex(C1CUR[st], hx);
                if (fr != lastf) lastf = fr, nthis = 0;
                if (!strncmp(hx, ENV_ONCE("OOT_GW_TRACE"), strlen(ENV_ONCE("OOT_GW_TRACE"))) && fr % 300 < 2 && nthis++ < 24)
                    rt_log("gwtrace: f%llu %s window c0+%#x -> %#llx+%#llx bound at buffer offset %#llx range %#llx (lead %#llx, written %d)",
                           (unsigned long long)fr, hx, (uint32_t)r->handle * 4, (unsigned long long)b->addr, (unsigned long long)b->size,
                           (unsigned long long)bi.offset, (unsigned long long)bi.range, (unsigned long long)lead, written);
            }
            if (written && !gw && bi.buffer != VK_NULL_HANDLE) {
                note_written(ex, b->addr - lead, bi.range);
                vk_exec_mesh_watch("storage write", st, (uint32_t)r->handle * 4, b->addr - lead, bi.range);
            }
            if (written && gw) vk_exec_mesh_watch("window entry (may write)", st, (uint32_t)r->handle * 4, b->addr - lead, bi.range);
            return bi;
        }
        if (b->addr) {
            // Which program and binding (the first 8): an "address" made of two floats is a pointer read from the
            // wrong place (2026-11-09: 0x3f12864e3e51b3a4 in the level).
            static volatile LONG said;
            if (InterlockedIncrement(&said) <= 8) {
                char hx[17];
                shader_hex(C1CUR[st], hx);
                VkMemSpan s1;   // what its first byte resolves to: contiguous bytes from there, and the class
                int r1 = vk_mem_resolve(b->addr, 1, &s1);
                rt_log("vk: storage buffer %#llx+%#llx (stage %d, c%d+%#x, program %s) is not backed or not aligned: a zero "
                       "buffer is bound (first byte: %s, %#llx contiguous, class %d)", (unsigned long long)b->addr,
                       (unsigned long long)b->size, st, r->cbuf_slot, (uint32_t)r->handle * 4, hx, r1 ? "backed" : "not backed",
                       (unsigned long long)s1.avail, s1.cls);
            }
        }
    } else {
        warn_once(W_SSBO_SLOT, "a storage buffer whose address is at c%d+%#x (stage %d) is not replayed: a zero buffer is bound",
                  r->cbuf_slot, r->handle * 4, st);
    }
    // A written binding with no backing stores into the discard buffer: the zero buffer every unresolved read shares
    // must stay zero (a store into it made counts and list lengths garbage for every other unresolved reader).
    if (written) {
        mwatch_discard(ex, st, r);
        return (VkDescriptorBufferInfo){Z.dbuf, 0, max < ZERO_BYTES ? max : ZERO_BYTES};
    }
    return zero_info(max);
}

// ---- feedback loops (maxwell.md 1.9, vk.h VkCtx.feedback)
// TotK's G-buffer and terrain shaders sample targets of the pass that renders them, at their own pixel, after earlier
// draws of the pass wrote them (albedo, normals, R32F linear depth). Maxwell reads see those writes; in Vulkan the read
// is undefined without a barrier, and on the RX 9070 it showed as dark screen-aligned blocks on the island ground
// and hid Tulin's sage spirit. Before a draw whose fragment shader samples an attachment of
// the open pass that a draw or clear of the pass wrote since the pass began or the last barrier:
// - VK_FEEDBACK_LAYOUT: the image is learned (VkTexImage.fb_learned; it has the feedback usage since its creation), the
//   pass restarts once with it in ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT, and from then on its passes attach it in that
//   layout, its descriptors carry that layout, the draws that sample it use a pipeline with the feedback flag
//   (VkPipeState.feedback) and a by-region barrier precedes such a draw.
// - VK_FEEDBACK_RESTART: the pass ends and restarts behind the full barrier (vk_exec_pass_begin).
static int in_fbl(const VkExec *ex, VkImage img) {
    for (int i = 0; i < ex->nfbl; i++)
        if (ex->fbl_img[i] == img) return 1;
    return 0;
}

// The image and the levels [l0, l1) and layers [a0, a1) a texture's view covers.
typedef struct { VkImage image; uint32_t l0, l1, a0, a1; } SubRange;

static SubRange sub_of(const VkTexImage *img, const view_t *v, int hv) {
    SubRange r = {img->image, 0, img->levels, 0, img->layers};
    if (hv && v->set_levels) r.l0 = v->base_level, r.l1 = v->base_level + (v->nlevels ? v->nlevels : img->levels);
    if (hv && v->set_layers) r.a0 = v->min_layer, r.a1 = v->min_layer + (v->nlayers ? v->nlayers : img->layers);
    return r;
}

// Attachment slots of the open pass (bit 8 depth) whose subresource r covers.
static uint32_t attached_as(const VkExec *ex, const SubRange *r) {
    uint32_t m = 0;
    if (!r->image) return 0;
    for (uint32_t i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if ((i < 8 && i >= ex->pass_ncolor) || t->image != r->image) continue;
        if (t->level < r->l0 || t->level >= r->l1) continue;
        if (!t->is3d && (t->base_layer + t->layers <= r->a0 || t->base_layer >= r->a1)) continue;
        m |= 1u << i;
    }
    return m;
}

// OOT_TEX_MEMO (tex_info, feedback_scan): 0 off, 1 on, 2 (=check) computes every hit again and stops on a difference.
static int td_mode(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_value("OOT_TEX_MEMO");
        on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
    }
    return on;
}

// The subresource a texture handle samples, per handle, kept while its slot resolves the same (slot_still): a draw's
// texture bindings change the whole-scan memo below, but the handles they bind were mostly looked up before, in this
// pass or an earlier one. Direct-mapped.
#define FM_N 8192
static struct { uint64_t h, tok; uint32_t gen; SubRange r; } FM[FM_N];

static SubRange sampled_sub(uint64_t h) {   // texture lock held
    view_t v;
    int hv;
    const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), &v, &hv);
    return img ? sub_of(img, &v, hv) : (SubRange){0};
}

// Attachments of the open pass the fragment shader fs samples; kept while the shader, the texture bindings and the
// pass stay the same.
static uint32_t feedback_scan(VkExec *ex, const GfxShader *fs) {
    if (ex->fb_fs == fs && ex->fb_serial == ex->tex_serial && ex->fb_pass == ex->pass_serial) return ex->fb_sampled;
    const GfxShaderRes *res;
    int n = gfx_shader_resources(fs, GFX_RES_TEX, &res);
    uint32_t m = 0, gen = vk_texture_slots_gen();   // before any lookup
    int memo = td_mode(), locked = 0;
    for (int r = 0; r < n; r++)
        for (int k = 0; k < (res[r].count ? res[r].count : 1); k++) {
            int slot = res[r].slot + k;
            uint64_t h = slot >= 0 && slot < 128 ? ex->tex[NVN_STAGE_FS][slot] : 0;
            if (!h) continue;
            typeof(FM[0]) *f = &FM[(h * 0x9E3779B97F4A7C15ull) >> 51];
            if (memo && f->h == h && slot_still(&f->gen, f->tok, h, gen)) {
                if (memo == 2) {
                    if (!locked) vk_texture_lock(), locked = 1;
                    SubRange c = sampled_sub(h);
                    if (memcmp(&c, &f->r, sizeof c))
                        rt_fatal("vk_exec: feedback memo of handle %#llx differs from a lookup", (unsigned long long)h);
                }
                m |= attached_as(ex, &f->r);
                continue;
            }
            uint64_t tok = token_of(h);   // before the lookup
            if (!locked) vk_texture_lock(), locked = 1;
            SubRange sr = sampled_sub(h);
            m |= attached_as(ex, &sr);
            if (tok) *f = (typeof(*f)){.h = h, .tok = tok, .gen = gen, .r = sr};
            else f->h = 0;
        }
    if (locked) vk_texture_unlock();
    ex->fb_fs = fs, ex->fb_serial = ex->tex_serial, ex->fb_pass = ex->pass_serial, ex->fb_sampled = m;
    return m;
}

// Before the pipeline lookup of a draw with fragment shader fs: learns and restarts as the mode says. Returns the
// sampled attachments in the feedback layout (the pipeline's feedback flag and feedback_barrier).
static uint32_t feedback_prepare(VkExec *ex, const GfxShader *fs) {
    int mode = vk_ctx()->feedback;
    if (mode == VK_FEEDBACK_OFF || !fs || !ex->in_pass) return 0;
    uint32_t m = feedback_scan(ex, fs);
    uint32_t d = m & ex->fb_dirty;
    if (d) vk_exec_frame.fb_draws++;
    if (d & ~(mode == VK_FEEDBACK_LAYOUT ? ex->fbl_mask : 0)) {
        if (mode == VK_FEEDBACK_LAYOUT)
            for (uint32_t i = 0; i < 9; i++) {
                const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
                if ((d >> i & 1) && t->img && t->img->fb_usage) ((VkTexImage *)t->img)->fb_learned = 1;
            }
        vk_exec_pass_end(ex);
        vk_exec_pass_begin(ex);   // learned targets now in the feedback layout; the full barrier cleared fb_dirty
        vk_exec_frame.fb_restarts++;
    }
    return mode == VK_FEEDBACK_LAYOUT ? m & ex->fbl_mask : 0;
}

// By-region barrier inside the pass: earlier attachment writes become visible to this draw's fragment shader reads.
static void feedback_barrier(VkExec *ex) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                           VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
                           .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded++;
    ex->fb_dirty = 0;
    vk_exec_frame.fb_barriers++;
}

// Marks the attachments the draw with state s writes: colour targets with a nonzero effective write mask, depth with
// depth writes or the stencil test on.
static void feedback_written(VkExec *ex, const VkPipeState *s) {
    if (vk_ctx()->feedback == VK_FEEDBACK_OFF) return;
    if (ex->depth.image && !(ex->fb_dirty & 0x100) && ((s->ds.dt && s->ds.dw) || s->ds.st)) ex->fb_dirty |= 0x100;
    for (uint32_t i = 0; i < ex->pass_ncolor; i++)
        if (ex->color[i].image && !(ex->fb_dirty >> i & 1) && vk_pipeline_target_wmask(s, (int)i)) ex->fb_dirty |= 1u << i;
}

static void tex_info_full(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                          VkDescriptorImageInfo *out, VkImage *image);

// Sampled-texture descriptors per (handle, resource type, separate), kept while nothing they derive from changed: the
// slot's resolution and texture entry (slot_still) and the samplers (vk_sampler_gen); the layout is set again on every
// hit from the images in the feedback layout (in_fbl of the image kept with it). The first binding of a slot in a submission (vk_texture_epoch) still makes the slot lookup
// for its storage check; a re-upload writes the same image, so the descriptor stays. Draws bind the same textures
// thousands of times a frame (15,000 lookups a frame in the monsters fight); a hit skips the slot lookup, the view key
// with its hash probe and the sampler lookup. Storage images, which note their binding per submission, are
// not kept. Direct-mapped; texture lock held. OOT_TEX_MEMO=0 turns it off; =check computes every hit again and stops
// on a difference.
#define TD_N 8192
static struct {
    uint64_t h, tok;
    uint32_t gen, epoch, sgen;
    uint16_t type;
    uint8_t separate, valid;
    VkImage image;
    VkDescriptorImageInfo out;
} TD[TD_N];

static void tex_info_lookup(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                            VkDescriptorImageInfo *out) {
    int td_on = td_mode();
    VkImage image;
    if (!h || storage || !td_on) return tex_info_full(ex, r, h, storage, separate, out, &image);
    uint32_t gen = vk_texture_slots_gen(), ep = vk_texture_epoch(), sg = vk_sampler_gen();   // before any lookup
    typeof(TD[0]) *m = &TD[(h * 0x9E3779B97F4A7C15ull ^ (uint64_t)r->type << 7 ^ (uint64_t)separate) >> 51];
    if (m->valid && m->h == h && m->type == r->type && m->separate == separate && m->sgen == sg &&
        slot_still(&m->gen, m->tok, h, gen)) {
        if (m->epoch != ep) {   // the slot's first binding in this submission: only its storage check is due
            vk_texture_by_id_locked((int)(h & 0xfffff), NULL, NULL);
            m->epoch = ep;
        }
        *out = m->out;
        out->imageLayout = m->image && ex->nfbl && in_fbl(ex, m->image) ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                                                                          : VK_IMAGE_LAYOUT_GENERAL;
        if (td_on == 2) {
            VkDescriptorImageInfo full;
            tex_info_full(ex, r, h, storage, separate, &full, &image);
            if (full.sampler != out->sampler || full.imageView != out->imageView || full.imageLayout != out->imageLayout)
                rt_fatal("vk_exec: texture descriptor memo of handle %#llx type %#x differs from a lookup",
                         (unsigned long long)h, r->type);
        }
        return;
    }
    uint64_t tok = token_of(h);   // before the lookup
    tex_info_full(ex, r, h, storage, separate, out, &image);
    *m = (typeof(*m)){.h = h, .tok = tok, .gen = gen, .epoch = ep, .sgen = sg, .type = r->type,
                      .separate = (uint8_t)separate, .valid = 1, .image = image, .out = *out};
}

static void tex_info_full(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                          VkDescriptorImageInfo *out, VkImage *image) {
    int ms = (r->type & 0x200) != 0;
    *image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler samp = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    if (h) {
        view_t v;
        int hv;
        const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), &v, &hv);
        // A game R32 storage view writes the bytes the image of another descriptor at the same address is read
        // through (the scene target): store into that image (vk_texture_r32_sibling_locked).
        if (img && storage && hv && v.set_format &&
            (v.format == NVN_FORMAT_R32UI || v.format == NVN_FORMAT_R32F || v.format == NVN_FORMAT_R32I) && colour32(img->format))
        {
            const VkTexImage *was = img;
            img = vk_texture_r32_sibling_locked(img);
            static volatile LONG said;
            if (InterlockedIncrement(&said) <= 12)
                rt_log("vk_exec: R32 storage view of tex %d at %#llx (%ux%u vk fmt %d usage %#x owned %d) -> %s image (vk fmt %d usage %#x owned %d)",
                       (int)(h & 0xfffff), (unsigned long long)was->addr, was->width, was->height, (int)was->format,
                       (unsigned)was->usage, was->gpu_owned, img == was ? "the same" : "sibling", (int)img->format,
                       (unsigned)img->usage, img->gpu_owned);
        }
        if (img) view = get_view(ex, img, &v, hv, r->type, storage ? (r->format ? 1 : 2) : 0), *image = img->image;
        if (img && storage) vk_texture_storage_bound_locked(img);
        if (img && !storage && ex->nfbl && in_fbl(ex, img->image)) layout = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
        if (!storage && !separate) {
            // A depth texture sampled without a compare reads one texel on the Switch, whatever the sampler's filter:
            // blending neighbouring depths puts reconstructed positions in mid-air at object edges (shadow masks).
            int sid = NVN_HANDLE_SAMP(h);
            int point = img && (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && !(r->type & 0x400) &&
                        vk_exec_view_aspect(img, &v, hv) != VK_IMAGE_ASPECT_STENCIL_BIT;
            samp = point ? vk_sampler_point_by_id_locked(sid) : vk_sampler_by_id_locked(sid);
        }
    }
    if (!view) {
        int di = dummy_index(res_view_type(r->type), ms);
        view = storage ? storage_dummy(di, r->format) : Z.img[di].view;
    }
    if (!samp) samp = r->type & 0x400 ? Z.samp_shadow : Z.samp;
    *out = (VkDescriptorImageInfo){separate || storage ? VK_NULL_HANDLE : samp, view, layout};
}

// ---- depth twins: colour images under a compare sampler in a compute shader
// TotK's compute passes sample an R32_SFLOAT texture through a shadow sampler. Maxwell compares any single-channel
// float or unorm texel; Vulkan only formats with SAMPLED_IMAGE_DEPTH_COMPARISON (VUID-vkCmdDispatch-None-06479), which
// AMD's drivers do not give R32_SFLOAT, so the result there is undefined. Such an image gets a depth twin of the same
// size and bits (R32_SFLOAT -> D32_SFLOAT, R16_UNORM -> D16_UNORM), copied through a buffer right before each dispatch
// that samples it (a dispatch is outside any render pass), and the dispatch samples the twin. Draws keep the colour
// image: a copy cannot go inside their pass. Design: Savage Wind Recomp runtime/nvn/vulkan/exec.c depth_twin_update
// (https://github.com/Revivedx/SavageWindRecomp, MIT); the code is ours. OOT_DEPTH_TWIN=0: off.
#define TWINS 8
typedef struct { VkImage src, img; VmaAllocation mem; VkImageView view[2]; uint32_t w, h, layers; VkFormat fmt; } Twin;
static Twin twins[TWINS];
static unsigned twin_next;
static int twin_dispatch;                // op_dispatch*: the resources being bound are a dispatch's
static VkBuffer twin_buf;
static VmaAllocation twin_buf_mem;
static VkDeviceSize twin_buf_cap;

static void twin_free(void *arg) {   // completion thread, after the last submission that may sample it
    Twin *t = arg;
    VkCtx *c = vk_ctx();
    for (int k = 0; k < 2; k++)
        if (t->view[k]) vkDestroyImageView(c->device, t->view[k], NULL);
    vmaDestroyImage(c->vma, t->img, t->mem);
    free(t);
}

typedef struct { VkBuffer buf; VmaAllocation mem; } TwinBuf;
static void twin_buf_free(void *arg) {   // completion thread, after the last submission that may copy through it
    TwinBuf *b = arg;
    vmaDestroyBuffer(vk_ctx()->vma, b->buf, b->mem);
    free(b);
}

static VkFormat twin_format(const VkTexImage *img) {
    static int on = -1;
    if (on < 0) on = rt_knob("DEPTH_TWIN", "1");
    if (!on || (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || img->samples > 1 || img->depth > 1) return VK_FORMAT_UNDEFINED;
    VkFormat df = img->format == VK_FORMAT_R32_SFLOAT ? VK_FORMAT_D32_SFLOAT
                : img->format == VK_FORMAT_R16_UNORM  ? VK_FORMAT_D16_UNORM : VK_FORMAT_UNDEFINED;
    if (!df) return df;
    static int compares[2] = {-1, -1};   // per source format: the driver compares it as it is
    int i = df == VK_FORMAT_D16_UNORM;
    if (compares[i] < 0) {
        VkFormatProperties3 p3 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
        VkFormatProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &p3};
        vkGetPhysicalDeviceFormatProperties2(vk_ctx()->phys, img->format, &p2);
        compares[i] = (p3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT) != 0;
        rt_log("vk_exec: %s under a compare sampler: %s", i ? "R16_UNORM" : "R32_SFLOAT",
               compares[i] ? "compared by the driver" : "sampled through a depth twin in compute passes");
    }
    return compares[i] ? VK_FORMAT_UNDEFINED : df;
}

static void twins_release(VkCtx *c) {
    for (int i = 0; i < TWINS; i++) {
        for (int k = 0; k < 2; k++)
            if (twins[i].view[k]) vkDestroyImageView(c->device, twins[i].view[k], NULL);
        if (twins[i].img) vmaDestroyImage(c->vma, twins[i].img, twins[i].mem);
    }
    memset(twins, 0, sizeof twins);
    if (twin_buf) vmaDestroyBuffer(c->vma, twin_buf, twin_buf_mem);
    twin_buf = VK_NULL_HANDLE, twin_buf_mem = NULL, twin_buf_cap = 0;
}

// The twin of img holding img's current texels, the copies recorded into ex->cb; NULL when img needs none or it cannot
// be made.
static Twin *twin_update(VkExec *ex, const VkTexImage *img) {
    VkFormat df = twin_format(img);
    if (!df) return NULL;
    VkCtx *c = vk_ctx();
    uint32_t bpp = df == VK_FORMAT_D32_SFLOAT ? 4 : 2, layers = img->layers ? img->layers : 1;
    VkDeviceSize bytes = (VkDeviceSize)img->width * img->height * layers * bpp;
    if (bytes > (64u << 20)) return NULL;
    if (bytes > twin_buf_cap) {   // one copy buffer, grown; the old one goes after the submission being recorded
        VkDeviceSize want = twin_buf_cap ? twin_buf_cap * 2 : 1u << 16;
        while (want < bytes) want *= 2;
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = want,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VkBuffer nb;
        VmaAllocation nm;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &nb, &nm, NULL) != VK_SUCCESS) return NULL;
        if (twin_buf) {
            TwinBuf *old = malloc(sizeof *old);
            if (!old) rt_fatal("vk: out of memory (depth twin)");
            *old = (TwinBuf){twin_buf, twin_buf_mem};
            vk_completion_add(vk_sync_issued() + 1, twin_buf_free, old);
        }
        twin_buf = nb, twin_buf_mem = nm, twin_buf_cap = want;
    }
    Twin *t = NULL;
    for (int i = 0; i < TWINS && !t; i++)
        if (twins[i].img && twins[i].src == img->image && twins[i].w == img->width && twins[i].h == img->height &&
            twins[i].layers == layers && twins[i].fmt == df)
            t = &twins[i];
    if (!t) {
        t = &twins[twin_next++ % TWINS];
        if (t->img) {   // the submission being recorded may still sample the replaced twin
            Twin *old = malloc(sizeof *old);
            if (!old) rt_fatal("vk: out of memory (depth twin)");
            *old = *t;
            vk_completion_add(vk_sync_issued() + 1, twin_free, old);
        }
        *t = (Twin){.src = img->image, .w = img->width, .h = img->height, .layers = layers, .fmt = df};
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = df,
                                .extent = {img->width, img->height, 1}, .mipLevels = 1, .arrayLayers = layers,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ci, &ai, &t->img, &t->mem, NULL) != VK_SUCCESS) {
            *t = (Twin){0};
            return NULL;
        }
        for (int k = 0; k < 2; k++) {
            VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->img,
                                        .viewType = k ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, .format = df,
                                        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, k ? layers : 1}};
            if (vkCreateImageView(c->device, &vi, NULL, &t->view[k]) != VK_SUCCESS) t->view[k] = VK_NULL_HANDLE;
        }
        static int nlog;
        if (nlog++ < 8)
            rt_log("vk_exec: depth twin %ux%ux%u %s for a compute pass's compare sampler", img->width, img->height,
                   layers, df == VK_FORMAT_D32_SFLOAT ? "D32_SFLOAT" : "D16_UNORM");
    }
    // The source's earlier writes and the twin's earlier readers before the copies; the twin's old texels are dropped.
    VkMemoryBarrier2 m0 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = t->img, .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, layers}};
    VkDependencyInfo d0 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m0,
                           .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(ex->cb, &d0);
    VkBufferImageCopy src = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers},
                             .imageExtent = {img->width, img->height, 1}};
    vkCmdCopyImageToBuffer(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, twin_buf, 1, &src);
    VkMemoryBarrier2 m1 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
    VkDependencyInfo d1 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m1};
    vkCmdPipelineBarrier2(ex->cb, &d1);
    VkBufferImageCopy dst = {.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, layers},
                             .imageExtent = {img->width, img->height, 1}};
    vkCmdCopyBufferToImage(ex->cb, twin_buf, t->img, VK_IMAGE_LAYOUT_GENERAL, 1, &dst);
    // The twin's texels before the dispatch; the source's next writers and the buffer's next copy after these copies.
    VkMemoryBarrier2 m2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo d2 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m2};
    vkCmdPipelineBarrier2(ex->cb, &d2);
    ex->recorded += 5;
    return t;
}

// The image descriptor of slot h; in a dispatch, a compare-sampled colour image the driver cannot compare is swapped
// for its depth twin.
static void tex_info(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate, VkDescriptorImageInfo *out) {
    tex_info_lookup(ex, r, h, storage, separate, out);
    if (!twin_dispatch || storage || separate || !h || !(r->type & 0x400)) return;
    VkImageViewType vt = res_view_type(r->type);
    if (vt != VK_IMAGE_VIEW_TYPE_2D && vt != VK_IMAGE_VIEW_TYPE_2D_ARRAY) return;
    const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), NULL, NULL);
    Twin *t = img ? twin_update(ex, img) : NULL;
    VkImageView v = t ? t->view[vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY] : VK_NULL_HANDLE;
    if (v) out->imageView = v, out->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
}

// ---- pool sets (gfx.h GFX_POOL_SET)
// A shader that samples through a handle it loads from memory indexes arrays standing for the bound texture and
// sampler pools (the translator's BindlessElimination). Writing those arrays per draw would be some 100,000
// descriptors (OoT's texture pool holds 33068 entries), so each bound pool pair gets one persistent set, updated after
// bind: element k of a view type's binding holds texture id k of the pool, or the stand-in of that view type; element
// k of the sampler binding holds sampler id k.
// Only ids the GPU sampled get their texture. Every pool texture access stores 1 at its id in the set's feedback buffer
// (gfx_shader.c add_pool_feedback), a device-local buffer that each submission which bound the set copies into a
// host-cached one at its end (vk_exec_pool_feedback); pool_sync reads that every FB_MS (uncached reads of the
// device-local buffer through the BAR took 20 ms per sync): an id seen there joins the wanted list and is
// resolved from then on. Creating the texture of every live id instead took more than the 12 GB of VRAM on the level
// (images are made on first lookup and ASTC decodes to RGBA8). A newly sampled texture so shows the stand-in for a
// frame or two. For wanted ids, pool_sync rewrites only elements whose slot resolves otherwise (vk_texture_slot_token)
// and repeats the per-submit storage check of the lookup for CPU-written textures, which a per-draw binding made.
// A binding is filled the first time a pipeline indexes it.
// Not covered yet: a pool texture that is also an attachment of the open pass (the feedback scan sees bound slots
// only), the depth-without-compare point sampler of bound slots (tex_info_lookup), and giving textures back once the
// GPU stops sampling them (the feedback is never cleared).
#define POOL_SETS 4
#define POOL_FLUSH 1024
#define FB_MS 30.0
typedef struct {
    int tbase, sbase, ntex, nsamp;           // slot bases and lengths of the texture and sampler pool
    VkDescriptorPool dp;
    VkDescriptorSet set;
    VkBuffer fb, rb;                         // sampling feedback: one uint per texture id, and its host-cached copy
    VmaAllocation fb_alloc, rb_alloc;
    const uint32_t *rb_map;
    uint64_t used;                           // the last submission (fence) that bound the set
    double fb_read_ms;                       // when it was last read
    uint32_t filled;                         // GFX_POOL_* bindings written in full
    uint32_t tgen, sgen, epoch;              // vk_texture_slots_gen, vk_sampler_gen, vk_texture_epoch at the last sync
    uint8_t *wanted;                         // per texture id: the GPU sampled it
    int32_t *list;                           // the wanted ids, nwanted of them
    int nwanted;
    uint64_t *tok;                           // per texture id: its slot token at its last resolve (0: never registered)
    uint8_t *recheck;                        // per texture id: 1 CPU-written (storage check per submit), 2 unresolved
    VkImageView *view[GFX_POOL_BINDINGS];    // per image binding and id: the view the set holds
    VkSampler *samp;                         // per sampler id: the sampler the set holds
} PoolSet;
static PoolSet PS[POOL_SETS];
static int nps;
static struct { uint64_t syncs, scans, resolves, writes, binds, fb_reads; double ms; } PST;   // OOT_TRACE=perf

static const uint16_t pool_rtype[GFX_POOL_BINDINGS] = {0, 1, 3, 4, 5, 0x101, 0x103, 0x105};   // GfxShaderRes.type

static double now_ms_qpc(void) {
    static double inv;
    LARGE_INTEGER t;
    if (!inv) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        inv = 1000.0 / (double)f.QuadPart;
    }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * inv;
}

static struct { VkWriteDescriptorSet w[POOL_FLUSH]; VkDescriptorImageInfo ii[POOL_FLUSH]; uint32_t n; } PW;
static void pw_flush(void) {
    if (PW.n) vkUpdateDescriptorSets(vk_ctx()->device, PW.n, PW.w, 0, NULL);
    PST.writes += PW.n;
    PW.n = 0;
}
static void pw_add(VkDescriptorSet set, uint32_t b, uint32_t el, VkImageView v, VkSampler smp) {
    if (PW.n == POOL_FLUSH) pw_flush();
    PW.ii[PW.n] = (VkDescriptorImageInfo){smp, v, VK_IMAGE_LAYOUT_GENERAL};
    PW.w[PW.n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = b,
                                        .dstArrayElement = el, .descriptorCount = 1,
                                        .descriptorType = b == GFX_POOL_SAMPLER ? VK_DESCRIPTOR_TYPE_SAMPLER
                                                                                : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                        .pImageInfo = &PW.ii[PW.n]};
    PW.n++;
}

static VkImageView pool_dummy(int b) { return Z.img[dummy_index(res_view_type(pool_rtype[b]), 0)].view; }

// The set of the pool pair bound at this point of the stream; NULL when the device has no pool layout or the table
// is full. Texture lock held.
static PoolSet *pool_set_of(const VkExec *ex) {
    VkDescriptorSetLayout sl = vk_pipeline_pool_layout();
    if (!sl) return NULL;
    int tb = nvn_tpool_slot(ex->tex_pool, 0), sb = nvn_spool_slot(ex->samp_pool, 0);
    for (int i = 0; i < nps; i++)
        if (PS[i].tbase == tb && PS[i].sbase == sb) return &PS[i];
    if (nps == POOL_SETS) {
        warn_once(W_BIND, "more than %d texture and sampler pool pairs: pool-indexed samples of the others read stand-ins", POOL_SETS);
        return NULL;
    }
    PoolSet *p = &PS[nps];
    memset(p, 0, sizeof *p);
    p->tbase = tb, p->sbase = sb;
    p->ntex = ex->tex_pool && ex->tex_pool->n > 0 ? ex->tex_pool->n : GFX_POOL_TEXTURES;
    p->nsamp = ex->samp_pool && ex->samp_pool->n > 0 ? ex->samp_pool->n : GFX_POOL_SAMPLERS;
    if (p->ntex > GFX_POOL_TEXTURES || p->nsamp > GFX_POOL_SAMPLERS)
        rt_log("vk_exec: a pool of %d textures and %d samplers: ids past %d and %d read stand-ins", p->ntex, p->nsamp,
               GFX_POOL_TEXTURES, GFX_POOL_SAMPLERS);
    if (p->ntex > GFX_POOL_TEXTURES) p->ntex = GFX_POOL_TEXTURES;
    if (p->nsamp > GFX_POOL_SAMPLERS) p->nsamp = GFX_POOL_SAMPLERS;
    VkCtx *c = vk_ctx();
    VkDescriptorPoolSize sz[3] = {{VK_DESCRIPTOR_TYPE_SAMPLER, GFX_POOL_SAMPLERS},
                                  {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, (GFX_POOL_BINDINGS - 1) * GFX_POOL_TEXTURES},
                                  {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, .maxSets = 1,
                                     .poolSizeCount = 3, .pPoolSizes = sz};
    if (vkCreateDescriptorPool(c->device, &pi, NULL, &p->dp) != VK_SUCCESS) rt_fatal("vk: the pool set's descriptor pool failed");
    VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = p->dp,
                                      .descriptorSetCount = 1, .pSetLayouts = &sl};
    if (vkAllocateDescriptorSets(c->device, &ai, &p->set) != VK_SUCCESS) rt_fatal("vk: the pool set could not be allocated");
    // The feedback buffer, device-local (the shaders store into it at every pool sample), zeroed by the first copy into
    // it; its readback copy, host-cached.
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = GFX_POOL_TEXTURES * 4,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo fa = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VmaAllocationCreateInfo ra = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &fa, &p->fb, &p->fb_alloc, NULL) != VK_SUCCESS ||
        vmaCreateBuffer(c->vma, &bi, &ra, &p->rb, &p->rb_alloc, &info) != VK_SUCCESS || !info.pMappedData)
        rt_fatal("vk: the pool set's feedback buffers failed");
    memset(info.pMappedData, 0, GFX_POOL_TEXTURES * 4);
    vmaFlushAllocation(c->vma, p->rb_alloc, 0, VK_WHOLE_SIZE);
    p->rb_map = info.pMappedData;
    VkCommandBuffer ucb = vk_upload_begin();   // outside any pass, ahead of this submission in queue order
    vkCmdFillBuffer(ucb, p->fb, 0, VK_WHOLE_SIZE, 0);
    vk_upload_end();
    VkDescriptorBufferInfo dbi = {p->fb, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet fw = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = p->set, .dstBinding = GFX_POOL_FEEDBACK,
                               .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi};
    vkUpdateDescriptorSets(c->device, 1, &fw, 0, NULL);
    p->tok = calloc(GFX_POOL_TEXTURES, sizeof *p->tok);
    p->recheck = calloc(GFX_POOL_TEXTURES, 1);
    p->wanted = calloc(GFX_POOL_TEXTURES, 1);
    p->list = calloc(GFX_POOL_TEXTURES, sizeof *p->list);
    if (!p->tok || !p->recheck || !p->wanted || !p->list) rt_fatal("vk: out of memory");
    rt_log("vk_exec: pool set %d: texture pool at slot %d (%d ids), sampler pool at slot %d (%d ids)", nps, tb, p->ntex, sb,
           p->nsamp);
    nps++;
    return p;
}

// Element id of the image bindings in mask: the view of the texture the slot resolves to now. Texture lock held.
static void pool_resolve(VkExec *ex, PoolSet *p, int id, uint32_t mask) {
    view_t v;
    int hv;
    const VkTexImage *img = vk_texture_live_by_id_locked(p->tbase + id, &v, &hv);
    p->recheck[id] = img ? (img->mirrored && !img->gpu_owned) : p->tok[id] ? 2 : 0;   // 2: stale or not yet creatable
    PST.resolves++;
    for (int b = 1; b < GFX_POOL_BINDINGS; b++) {
        if (!(mask >> b & 1)) continue;
        VkImageView view = img ? get_view(ex, img, &v, hv, pool_rtype[b], 0) : VK_NULL_HANDLE;
        if (!view) view = pool_dummy(b);
        if (view != p->view[b][id]) p->view[b][id] = view, pw_add(p->set, (uint32_t)b, (uint32_t)id, view, VK_NULL_HANDLE);
    }
}

// Brings the set up to the slots for the bindings in mask (filling new ones first). Texture lock held.
static void pool_sync(VkExec *ex, PoolSet *p, uint32_t mask) {
    uint32_t tg = vk_texture_slots_gen(), sg = vk_sampler_gen(), ep = vk_texture_epoch();   // before any lookup
    uint32_t add = mask & ~p->filled;
    double t0 = now_ms_qpc();
    int read_fb = t0 - p->fb_read_ms >= FB_MS;
    if (!add && !read_fb && tg == p->tgen && sg == p->sgen && ep == p->epoch) return;
    PST.syncs++;
    int all = 0;   // every wanted id again: a new image binding
    for (int b = 0; b < GFX_POOL_BINDINGS; b++) {
        if (!(add >> b & 1)) continue;
        int n = b == GFX_POOL_SAMPLER ? GFX_POOL_SAMPLERS : GFX_POOL_TEXTURES;
        VkDescriptorImageInfo *ii = malloc(sizeof *ii * (size_t)n);
        void *shadow = calloc((size_t)n, sizeof(VkImageView));
        if (!ii || !shadow) rt_fatal("vk: out of memory");
        VkImageView dv = b == GFX_POOL_SAMPLER ? VK_NULL_HANDLE : pool_dummy(b);
        for (int i = 0; i < n; i++) {
            ii[i] = (VkDescriptorImageInfo){b == GFX_POOL_SAMPLER ? Z.samp : VK_NULL_HANDLE, dv, VK_IMAGE_LAYOUT_GENERAL};
            if (b == GFX_POOL_SAMPLER) ((VkSampler *)shadow)[i] = Z.samp;
            else ((VkImageView *)shadow)[i] = dv;
        }
        VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = p->set, .dstBinding = (uint32_t)b,
                                  .descriptorCount = (uint32_t)n,
                                  .descriptorType = b == GFX_POOL_SAMPLER ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                  .pImageInfo = ii};
        vkUpdateDescriptorSets(vk_ctx()->device, 1, &w, 0, NULL);
        free(ii);
        if (b == GFX_POOL_SAMPLER) p->samp = shadow, p->sgen = 0;   // every sampler id below
        else p->view[b] = shadow, all = 1;
        p->filled |= 1u << b;
    }
    uint32_t imgs = p->filled & ~(1u << GFX_POOL_SAMPLER);
    int first_new = p->nwanted;
    if (read_fb) {   // ids the GPU sampled, as of the last completed copy, join the wanted list
        p->fb_read_ms = t0;
        PST.fb_reads++;
        vmaInvalidateAllocation(vk_ctx()->vma, p->rb_alloc, 0, VK_WHOLE_SIZE);
        for (int id = 0; id < p->ntex; id++)
            if (p->rb_map[id] && !p->wanted[id]) p->wanted[id] = 1, p->list[p->nwanted++] = id;
    }
    if (imgs) {
        int new_epoch = ep != p->epoch, scan = all || tg != p->tgen || new_epoch;
        if (scan) PST.scans++;
        for (int i = scan ? 0 : first_new; i < p->nwanted; i++) {
            int id = p->list[i];
            uint64_t t = vk_texture_slot_token(p->tbase + id);   // before the lookup it guards
            if (i >= first_new || all || t != p->tok[id] || (p->recheck[id] == 2 && new_epoch)) {
                p->tok[id] = t;
                pool_resolve(ex, p, id, imgs);
            } else if (p->recheck[id] == 1 && new_epoch) {
                vk_texture_by_id_locked(p->tbase + id, NULL, NULL);   // the storage check of a CPU-written texture
            }
        }
    }
    if (p->samp && sg != p->sgen)
        for (int id = 0; id < p->nsamp; id++) {
            VkSampler smp = vk_sampler_by_id_locked(p->sbase + id);
            if (!smp) smp = Z.samp;
            if (smp != p->samp[id]) p->samp[id] = smp, pw_add(p->set, GFX_POOL_SAMPLER, (uint32_t)id, VK_NULL_HANDLE, smp);
        }
    pw_flush();
    p->tgen = tg, p->sgen = sg, p->epoch = ep;
    double t1 = now_ms_qpc();
    PST.ms += t1 - t0;
    static double logged;
    if ((rt_trace & RT_TRACE_PERF) && t1 - logged >= 5000.0 && (logged = t1))
        rt_log("vk_exec: pool sets: %d ids wanted; %llu syncs, %llu feedback reads, %llu scans, %llu resolves, "
               "%llu descriptor writes, %llu binds, %.1f ms", p->nwanted, (unsigned long long)PST.syncs,
               (unsigned long long)PST.fb_reads, (unsigned long long)PST.scans, (unsigned long long)PST.resolves,
               (unsigned long long)PST.writes, (unsigned long long)PST.binds, PST.ms);
}

// Before a draw or dispatch whose shaders index the pools: syncs the bound pair's set and binds it at GFX_POOL_SET.
// A graphics bind holds while later binds use the same layout (B.pool_layout); compute binds every time, as the ASTC
// and FSR passes push their own compute sets into the same command buffers.
static void pool_bind(VkExec *ex, const VkPipeInfo *info) {
    int gfx = info->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS;
    vk_texture_lock();
    PoolSet *p = pool_set_of(ex);
    if (p) pool_sync(ex, p, info->pool_mask);
    vk_texture_unlock();
    if (p) p->used = ex->fence;
    if (!p || (gfx && B.pool_layout == info->layout && B.pool_set == p->set)) return;
    vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, GFX_POOL_SET, 1, &p->set, 0, NULL);
    PST.binds++;
    RS.cmds++;
    if (gfx) B.pool_layout = info->layout, B.pool_set = p->set;
}

// At the end of a submission, after its closing barrier: each pool set it bound copies its feedback into the readback
// buffer, which the host reads once the copy completed (a read before sees the previous copy: ids join a little later).
void vk_exec_pool_feedback(VkExec *ex) {
    int n = 0;
    for (int i = 0; i < nps; i++) {
        PoolSet *p = &PS[i];
        if (p->used != ex->fence) continue;
        VkBufferCopy r = {0, 0, (VkDeviceSize)p->ntex * 4};
        vkCmdCopyBuffer(ex->cb, p->fb, p->rb, 1, &r);
        n++;
    }
    if (!n) return;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                           .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded += n + 1;
}

static void pool_sets_release(VkCtx *c) {
    for (int i = 0; i < nps; i++) {
        vkDestroyDescriptorPool(c->device, PS[i].dp, NULL);
        vmaDestroyBuffer(c->vma, PS[i].fb, PS[i].fb_alloc);
        vmaDestroyBuffer(c->vma, PS[i].rb, PS[i].rb_alloc);
        free(PS[i].tok), free(PS[i].recheck), free(PS[i].samp), free(PS[i].wanted), free(PS[i].list);
        for (int b = 0; b < GFX_POOL_BINDINGS; b++) free(PS[i].view[b]);
    }
    memset(PS, 0, sizeof PS);
    nps = 0;
}

static VkDescriptorSet alloc_set(VkExec *ex, VkDescriptorSetLayout layout) {
    CbRes *r = cb_res(ex);
    VkCtx *c = vk_ctx();
    for (int attempt = 0; attempt < 2; attempt++) {
        if (r->npools) {
            VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                              .descriptorPool = r->pools[r->npools - 1], .descriptorSetCount = 1,
                                              .pSetLayouts = &layout};
            VkDescriptorSet set;
            VkResult res = vkAllocateDescriptorSets(c->device, &ai, &set);
            if (res == VK_SUCCESS) return set;
            if (res != VK_ERROR_OUT_OF_POOL_MEMORY && res != VK_ERROR_FRAGMENTED_POOL) rt_fatal("vk: descriptor set allocation failed (%d)", (int)res);
        }
        static const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16384}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256}};
        VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 256,
                                         .poolSizeCount = sizeof sizes / sizeof sizes[0], .pPoolSizes = sizes};
        if (r->npools == r->cpools) r->pools = grow(r->pools, &r->cpools, sizeof *r->pools);
        VkDescriptorPool pool = VK_NULL_HANDLE;
        AcquireSRWLockExclusive(&spare_lock);
        if (nspare_pool) pool = spare_pool[--nspare_pool];
        ReleaseSRWLockExclusive(&spare_lock);
        if (!pool && vkCreateDescriptorPool(c->device, &pi, NULL, &pool) != VK_SUCCESS) rt_fatal("vk: descriptor pool failed");
        r->pools[r->npools++] = pool;
    }
    rt_fatal("vk: descriptor set allocation failed twice");
    return VK_NULL_HANDLE;
}

static void support_in(const VkExec *ex, float in[3]) {   // what support_fill reads: viewport size, point size
    in[0] = ex->in_pass && ex->has_vp ? ex->vp[0][2] : (float)ex->pass_w;
    in[1] = ex->in_pass && ex->has_vp ? ex->vp[0][3] : (float)ex->pass_h;
    in[2] = ex->point_size > 0 ? ex->point_size : 1.0f;
}

static void support_fill(uint8_t *b, float w, float h, float point_size) {
    memset(b, 0, SUPPORT_BYTES);
    float inv[4] = {w > 0 ? 2.0f / w : 0, h > 0 ? 2.0f / h : 0, 0, 0}, size[4] = {w, h, 0, 0};
    memcpy(b + 144, inv, 16);
    memcpy(b + 160, size, 16);
    const float one = 1.0f;
    memcpy(b + 1392, &point_size, 4);   // point_size (translator patch 0008): vertex shaders that write none output it
    for (int i = 0; i < 73; i++) memcpy(b + 192 + 16 * i, &one, 4);   // render scale 1: the host renders at guest size
}

// OOT_BIND_STATS=1: bindings and descriptors per draw of each stage (Task 2.2's choice of the pushed stage), and how
// often a draw repeats the previous one, logged every 20000 draws: pushes recorded and skipped, allocated sets written
// and reused, stage texture-type recomputations (answers_for), draws with the previous draw's program, bound state
// and textures (shaders and pipeline would be the same) and those that also have its buffers and viewport (all
// descriptors would be the same). BS is declared beside DS.
static int bind_stats_on(void) {
    if (BS.on < 0) {
        const char *e = cfg_get("OOT_BIND_STATS");
        BS.on = e && *e && strcmp(e, "0");
    }
    return BS.on;
}

static void bind_stats(const VkPipeInfo *info) {
    if (!bind_stats_on() || info->bind_point != VK_PIPELINE_BIND_POINT_GRAPHICS) return;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        int fs = info->n_sets > 1 ? b->set : !(b->stages & VK_SHADER_STAGE_VERTEX_BIT);
        BS.bind[fs]++, BS.desc[fs] += b->count ? b->count : 1;
    }
    if (++BS.draws % 20000 == 0) {
        double d = (double)BS.draws;
        rt_log("vk_exec: per draw over %llu draws: vertex %.2f bindings %.2f descriptors, fragment %.2f bindings %.2f descriptors",
               (unsigned long long)BS.draws, BS.bind[0] / d, BS.desc[0] / d, BS.bind[1] / d, BS.desc[1] / d);
        rt_log("vk_exec: per draw: pushes %.3f skipped %.3f, sets written %.3f reused %.3f, texture types %.3f; "
               "same program and state as the draw before %.3f, also same buffers %.3f; descriptor builds skipped %.3f",
               BS.push / d, BS.push_skip / d, BS.set_new / d, BS.set_reuse / d, BS.ac_miss / d, BS.same_prog / d,
               BS.same_all / d, BS.bind_skip / d);
    }
}

// The bound state a draw reads, compared with the previous draw's (OOT_BIND_STATS only).
static void repeat_stats(const VkExec *ex, const VkPipeInfo *info) {
    static struct {
        const prog_t *prog;
        VkPipeState pipe;
        uint64_t tex[2][128], img[2][32];
        const VkPipeInfo *info;
        p_bind_buf ubo[2][18], ssbo[2][16];
        float vp[6], point_size;
    } last, cur;
    if (!bind_stats_on()) return;
    memset(&cur, 0, sizeof cur);
    cur.prog = ex->prog, cur.pipe = ex->pipe, cur.info = info;
    const int st[2] = {NVN_STAGE_VS, NVN_STAGE_FS};
    for (int k = 0; k < 2; k++) {
        memcpy(cur.tex[k], ex->tex[st[k]], sizeof cur.tex[k]), memcpy(cur.img[k], ex->img[st[k]], sizeof cur.img[k]);
        memcpy(cur.ubo[k], ex->ubo[st[k]], sizeof cur.ubo[k]), memcpy(cur.ssbo[k], ex->ssbo[st[k]], sizeof cur.ssbo[k]);
    }
    memcpy(cur.vp, ex->vp[0], sizeof cur.vp), cur.point_size = ex->point_size;
    size_t prog_part = offsetof(typeof(cur), info);
    if (!memcmp(&cur, &last, prog_part)) {
        BS.same_prog++;
        if (!memcmp((const char *)&cur + prog_part, (const char *)&last + prog_part, sizeof cur - prog_part)) BS.same_all++;
    }
    last = cur;
}

static int push_skip_on(void) {   // OOT_PUSH_SKIP=0: every graphics draw pushes its set again
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("OOT_PUSH_SKIP");
    }
    return on;
}

// A graphics draw whose descriptor inputs equal the previous graphics draw's in this command buffer skips building its
// sets: every write would be the same, the push would be skipped and the allocated set reused without a bind (about a
// quarter of field and monsters2 draws, OOT_BIND_STATS). The inputs, compared and stored in place, are what
// bind_resources and its helpers read beyond the pipeline info's fixed bindings and Z's stand-ins:
// - the info (bindings, layouts), the program and its serial (the compiler-constant range of UBO slot 1);
// - vk_texture_slots_gen, vk_texture_epoch (a slot's image, view and once-per-submit storage check), vk_sampler_gen;
// - vk_mem_gen (buffer and offset of a span, mirror creation and per-submission sync);
// - the images in the feedback layout (the sampled layout);
// - per binding: the support inputs, c0's storage table and base instance, the bound uniform and storage ranges, the
//   texture and image handles (RsList; compared only when a bind op ran since the last draw, bind_gen);
// - storage buffer addresses read from user uniform blocks (RsList.p, ubo_ptr): compared on every draw, since the
//   block's contents change without a bind op.
// The side effects of a build repeat as no-ops in the same command buffer and submission: c0 and support buffers kept
// in B, note_written's ranges, vk_mem_gpu_wrote's write-backs, the per-submission mirror sync and dirty census note,
// the views cached per submission. The census counts are added again. A build that copied guest bytes into transient
// memory (an unaligned host range) is never skipped after: the copy may differ. B.res_valid says the sets bound hold
// the last build; vk_exec_draw_new_cb's memset clears it, and every build clears it until it ends.
// OOT_BIND_SKIP=0 builds every draw; =check builds every draw and stops when a draw with equal inputs gets other
// writes, census counts or descriptor commands.
static int bind_skip_on(void) {
    if (RS.on < 0) {
        const char *e = cfg_value("OOT_BIND_SKIP");
        RS.on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
        if (RS.on != 1) rt_log("vk_exec: descriptor build skip %s", RS.on ? "check" : "off");
    }
    return RS.on;
}

static void rs_add(RsRange *r, uint32_t *n, const VkExec *ex, const void *at, size_t len) {
    uint32_t off = (uint32_t)((const uint8_t *)at - (const uint8_t *)ex);
    if (*n && r[*n - 1].off + r[*n - 1].len == off) r[*n - 1].len += (uint32_t)len;
    else r[(*n)++] = (RsRange){off, (uint32_t)len};
}

static const RsList *rs_list(const VkExec *ex, const VkPipeInfo *info) {
    if (info == rs_last) return rs_last_l;
    if (!rs_lists.klen) nvn_hm_init(&rs_lists, sizeof info);
    RsList *l = nvn_hm_get(&rs_lists, &info);
    if (!l) {
        static RsRange tmp[MAX_DESC];
        RsPtr ptr[RS_PTRS];
        uint32_t n = 0, nd = 0, np = 0;
        int sup = 0, c0 = 0, ok = 1;
        for (uint32_t i = 0; i < info->n_bindings && ok; i++) {
            const VkPipeBinding *b = &info->bindings[i];
            const GfxShaderRes *r = b->res;
            uint32_t cnt = b->count ? b->count : 1;
            if ((nd += cnt) > MAX_DESC) { ok = 0; break; }   // the build stops on it
            int st = nvn_stage_of(b->stage);
            if (b->kind == VK_PIPE_SUPPORT) {
                sup = 1;
            } else if (b->kind == GFX_RES_UBO) {
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    if (slot == 0 && c0_on()) c0 = 1, rs_add(tmp, &n, ex, ex->ssbo[st], sizeof ex->ssbo[st]);
                    else if (slot >= 3 && slot - 3 < 18) rs_add(tmp, &n, ex, &ex->ubo[st][slot - 3], sizeof(p_bind_buf));
                }   // slot 1: the program (bind_sig); others: the zero buffer
            } else if (b->kind == GFX_RES_SSBO) {   // every descriptor of the binding is the same range
                int ubo, slot = ssbo_slot(r, st, &ubo);
                if (slot >= 0 && slot < 16) {
                    rs_add(tmp, &n, ex, ubo ? &ex->ubo[st][slot] : &ex->ssbo[st][slot], sizeof(p_bind_buf));
                } else if (ubo_ptr_res(r)) {
                    if (np == RS_PTRS) { ok = 0; break; }
                    rs_add(tmp, &n, ex, &ex->ubo[st][r->cbuf_slot - 3], sizeof(p_bind_buf));
                    ptr[np++] = (RsPtr){st, r};
                }
            } else if ((b->kind == GFX_RES_TEX || b->kind == GFX_RES_IMG) && b->type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER &&
                       b->type != VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
                int img = b->kind == GFX_RES_IMG, cap = img ? 32 : 128;
                if (r->cbuf_slot >= 3 || r->cbuf_slot == 0) ok = 0;   // handles in a uniform block or the pushed constants (cbuf_handle): not compared
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    if (slot >= 0 && slot < cap) rs_add(tmp, &n, ex, img ? &ex->img[st][slot] : &ex->tex[st][slot], 8);
                }
            }
        }
        uint64_t bytes = 0;
        for (uint32_t i = 0; i < n; i++) bytes += tmp[i].len;
        if (bytes > RS_BYTES) ok = 0;
        l = malloc(sizeof *l + sizeof *tmp * n);
        if (!l) rt_fatal("vk: out of memory");
        l->n = n, l->bytes = (uint32_t)bytes, l->sup = (uint8_t)sup, l->c0 = (uint8_t)c0, l->ok = (uint8_t)ok;
        l->np = (uint8_t)np;
        memcpy(l->p, ptr, sizeof *ptr * np);
        memcpy(l->r, tmp, sizeof *tmp * n);
        nvn_hm_put(&rs_lists, &info, l);
    }
    rs_last = info, rs_last_l = l;
    return l;
}

// The signature of a graphics draw's descriptor inputs into RS; 1 when it equals the stored one. The bound-slot bytes
// are compared only when a bind op ran since they were stored or the pipeline changed: otherwise they are the same.
// Check mode compares them anyway and stops when the bind op generation said they could not differ but they do.
static int bind_sig(const VkExec *ex, const VkPipeInfo *info) {
    const RsList *l = rs_list(ex, info);
    if (!l->ok) return RS.hn = 0, RS.info = NULL, 0;
    uint64_t *h = RS.h, diff = 0;
    uint32_t n = 0;
#define PUT(v) do { uint64_t v_ = (uint64_t)(v); diff |= h[n] ^ v_; h[n++] = v_; } while (0)
    PUT((uintptr_t)info);
    PUT((uintptr_t)ex->prog);
    PUT(ex->prog->serial);
    PUT((uint64_t)vk_texture_slots_gen() << 32 | vk_texture_epoch());
    PUT((uint64_t)vk_sampler_gen() << 32 | (uint32_t)ex->nfbl);
    PUT(vk_mem_gen() + vk_usnap_gen());   // both only grow
    if (info->pool_mask) {   // the pool set pool_bind binds
        PUT((uintptr_t)ex->tex_pool);
        PUT((uintptr_t)ex->samp_pool);
    }
    for (int i = 0; i < ex->nfbl && i < 9; i++) PUT((uintptr_t)ex->fbl_img[i]);
    if (l->sup) {
        float in[3];
        support_in(ex, in);
        uint64_t a;
        uint32_t ps;
        memcpy(&a, in, 8), memcpy(&ps, &in[2], 4);
        PUT(a);
        PUT(ps);
    }
    if (l->c0) PUT(c0_binst);
    for (int i = 0; i < l->np; i++) PUT(ubo_ptr(ex, l->p[i].st, l->p[i].res));
#undef PUT
    diff |= n != RS.hn;
    RS.hn = n;
    int fast = RS.info == info && RS.gen == bind_gen;
    if (fast && RS.on != 2) return !diff;
    const uint8_t *e = (const uint8_t *)ex;
    uint8_t *d = RS.body;
    int bd = 0;
    for (uint32_t i = 0; i < l->n; i++) {
        const RsRange *r = &l->r[i];
        if (!bd && memcmp(d, e + r->off, r->len)) bd = 1;
        if (bd) memcpy(d, e + r->off, r->len);
        d += r->len;
    }
    if (fast && bd) rt_fatal("vk_exec: descriptor build skip check: bound slots changed without a bind op");
    RS.info = info, RS.gen = bind_gen;
    return !diff && !bd;
}

// Writes allocated set s from its packed descriptors d: through the layout's update template, or with the write
// structs w[i0..i1) when it has none.
static void write_set(const VkPipeInfo *info, uint32_t s, VkDescriptorSet set, const VkDescSlot *d, VkWriteDescriptorSet *w,
                      uint32_t i0, uint32_t i1) {
    if (i1 == i0) return;
    if (info->tpl[s]) {
        vk_pipeline_update_tpl_fn()(vk_ctx()->device, set, info->tpl[s], d);
        return;
    }
    for (uint32_t k = i0; k < i1; k++) w[k].dstSet = set;
    vkUpdateDescriptorSets(vk_ctx()->device, i1 - i0, w + i0, 0, NULL);
}

// Writes the pipeline's sets from the bound state: set 0, and set 1 with independent layouts. Each is pushed into the
// command buffer, or allocated, written and bound; a graphics set whose contents equal the last one allocated for that
// set number in this command buffer is reused, and bound again only when another layout or a push came between. A
// graphics push equal to the last one at its set number, with no other layout or bind between, is not recorded.
// The descriptors go into one packed array d, a VkDescSlot per descriptor in binding order, which a set's update
// template reads directly (VkPipeInfo.tpl); write structs are filled only when a set has no template.
// A texture or image the shader takes by a handle in a user uniform block (cbuf_slot 3 and up; Ryujinx names it
// <stage>_t<type>_cb<slot>_<offset>, usually an array indexed by data, such as the cookie textures of this game's
// light list in its fog pass): element k is the 64-bit handle at byte offset * 4 + 8 k of the block bound to that slot,
// as the submission's snapshot of it holds (vk_usnap.c), else guest memory. 0 when it lies outside the binding. Before,
// such a resource got slot -1 (gfx_shader.c handle_slot) and element k the texture bound to slot k - 1: the morph
// ball's light sampled the depth buffer for its cookie, and the fog over its screen tiles turned dark.
static uint64_t cbuf_handle(const VkExec *ex, int st, const GfxShaderRes *r, uint32_t k) {
    if (r->bindless) return 0;   // a pool array the pool set does not support (gfx_shader.c add_res): the stand-in
    if (r->cbuf_slot == 0) {   // NVN2: the bound slot whose handle the bank-0 word carries
        int kind, slot;
        if (!nvn2_cs(ex, st)) return 0;
        slot = nvn2_c0_slot(ex, (uint32_t)(r->handle & 0x3fff) * 4 + (uint32_t)k * 8, &kind);
        if (slot < 0 || kind == 0) return 0;
        return kind == 2 ? (slot < 32 ? ex->img[st][slot] : 0) : (slot < 128 ? ex->tex[st][slot] : 0);
    }
    if (r->cbuf_slot < 3 || r->cbuf_slot - 3 >= 18) return 0;
    const p_bind_buf *u = &ex->ubo[st][r->cbuf_slot - 3];
    uint64_t off = (uint64_t)(r->handle & 0x3fff) * 4 + (uint64_t)k * 8, h = 0;
    if (!u->addr || off + 8 > u->size) return 0;
    const void *p = vk_usnap_peek(u->addr + off, 8);
    VkMemSpan sp;
    if (!p && vk_mem_resolve(u->addr + off, 8, &sp)) p = (const void *)(uintptr_t)(u->addr + off);
    if (p) memcpy(&h, p, 8);
    return h;
}

// The handle a texture or image declaration resolves to in the replay: its bound slot, or the word the shader reads
// it from (a user uniform block, or NVN2's bank 0). For vk_dump.c.
uint64_t vk_exec_res_handle(const VkExec *ex, int st, const GfxShaderRes *r, uint32_t k, int img) {
    int slot = r->slot + (int)k;
    if (r->cbuf_slot >= 3 || (r->cbuf_slot == 0 && slot < 0)) return cbuf_handle(ex, st, r, k);
    if (img) return slot >= 0 && slot < 32 ? ex->img[st][slot] : 0;
    return slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
}

static void bind_resources(VkExec *ex, const VkPipeInfo *info) {
    static VkWriteDescriptorSet w[MAX_BINDINGS];
    static VkDescSlot d[MAX_DESC];
    static VkBufferView bv[MAX_DESC];
    if (info->n_bindings > MAX_BINDINGS) rt_fatal("vk: %u descriptor bindings exceed the replay's %d", info->n_bindings, MAX_BINDINGS);
    bind_stats(info);
    int gfx = info->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS;
    int skip = gfx ? bind_skip_on() : 0, same = 0, was_valid = B.res_valid;
    int stored[32], nstored = 0;   // slot ids of storage images the shaders store into (marked written below)
    if (skip) {
        same = bind_sig(ex, info);
        if (same && was_valid && skip == 1) {
            for (int i = 0; i < RS.ncnt; i++) vk_mem_count(RS.cnt[i].cls, RS.cnt[i].use, RS.cnt[i].bytes);
            BS.bind_skip++;
            ex->recorded++;
            return;
        }
        B.res_valid = 0;   // until the build below ends
        if (skip == 2) memcpy(RS.pcnt, RS.cnt, sizeof RS.cnt), RS.pncnt = RS.ncnt;
        RS.rec = 1, RS.ncnt = 0, RS.cnt_over = 0, RS.volatile_ = 0, RS.cmds = 0;
    }
    uint32_t n_sets = info->n_sets ? info->n_sets : 1;
    uint32_t lo[2] = {0, 0}, hi[2] = {0, 0}, d0[2] = {0, 0};   // bindings [lo, hi) and first descriptor per set
    int need_w = 0;                                             // a set without a template: fill the write structs
    for (uint32_t s = 0, i0 = 0; s < n_sets; s++) {
        uint32_t i1 = i0;
        while (i1 < info->n_bindings && (n_sets == 1 || info->bindings[i1].set == s)) i1++;
        lo[s] = i0, hi[s] = i1, i0 = i1;
        if (i1 > lo[s] && !info->tpl[s]) need_w = 1;
    }
    uint32_t nd = 0;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        const GfxShaderRes *r = b->res;
        uint32_t cnt = b->count ? b->count : 1;
        if (nd + cnt > MAX_DESC) rt_fatal("vk: %u descriptors exceed the replay's %d", nd + cnt, MAX_DESC);
        if (n_sets == 2 && i == lo[1]) d0[1] = nd;
        VkDescSlot *o = &d[nd];
        int st = nvn_stage_of(b->stage), kind = 0;   // kind: 0 buffer, 1 image, 2 texel buffer view
        if (b->kind == VK_PIPE_SUPPORT) {
            float in[3];
            support_in(ex, in);
            if (!B.sup_valid || memcmp(in, B.sup_in, sizeof in)) {   // support_fill reads only these
                uint8_t tmp[SUPPORT_BYTES];
                support_fill(tmp, in[0], in[1], in[2]);
                memcpy(B.sup_in, in, sizeof in);
                if (!B.sup_valid || memcmp(tmp, B.support, SUPPORT_BYTES)) {   // a new buffer only when the contents changed
                    VkBuffer sb;
                    VkDeviceSize so;
                    memcpy(scratch(ex, SUPPORT_BYTES, &sb, &so), tmp, SUPPORT_BYTES);
                    memcpy(B.support, tmp, SUPPORT_BYTES);
                    B.sup_buf = sb, B.sup_off = so, B.sup_valid = 1;
                }
            }
            o->buffer = (VkDescriptorBufferInfo){B.sup_buf, B.sup_off, SUPPORT_BYTES};
        } else if (b->kind == GFX_RES_UBO) {
            for (uint32_t k = 0; k < cnt; k++) o[k].buffer = ubo_info(ex, st, r->slot + (int)k);
        } else if (b->kind == GFX_RES_SSBO) {
            for (uint32_t k = 0; k < cnt; k++) o[k].buffer = ssbo_info(ex, st, r, r->flags & 1);
        } else if (b->kind == GFX_RES_TEX || b->kind == GFX_RES_IMG) {
            int img = b->kind == GFX_RES_IMG;
            if (b->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || b->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
                kind = 2;
                for (uint32_t k = 0; k < cnt; k++) {   // buffer textures are not replayed yet
                    o[k].w[0] = 0, o[k].w[1] = 0, o[k].w[2] = 0;
                    o[k].view = Z.zview;
                    if (need_w) bv[nd + k] = Z.zview;
                }
            } else {
                kind = 1;
                vk_texture_lock();   // tex_info looks up with the texture lock held
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    uint64_t h = 0;
                    if (r->cbuf_slot >= 3 || (r->cbuf_slot == 0 && slot < 0)) h = cbuf_handle(ex, st, r, k);
                    else if (img) h = slot >= 0 && slot < 32 ? ex->img[st][slot] : 0;
                    else h = slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
                    {   // OOT_CS_DUMMY_TEX=1 (diagnostic, 2026-11-06): compute shaders read the dummy texture instead of
                        // the bound ones, to tell a texture read from the rest when a kernel loses the device
                        static int dummy_cs = -1;
                        if (dummy_cs < 0) dummy_cs = getenv("OOT_CS_DUMMY_TEX") != NULL;
                        if (dummy_cs && st == NVN_STAGE_CS && !img) h = 0;
                    }
                    tex_info(ex, r, h, img, r->separate, &o[k].image);
                    o[k].w[2] = (uint32_t)o[k].image.imageLayout;   // zero the tail padding
                    // an image the shader stores into: marked written after the lock is released (below)
                    if (img && (r->flags & 4) && h && nstored < (int)(sizeof stored / sizeof stored[0]))
                        stored[nstored++] = (int)(h & 0xfffff);
                }
                vk_texture_unlock();
            }
        } else {
            rt_fatal("vk: binding %u has an unknown kind %d", (unsigned)b->binding, (int)b->kind);
        }
        if (need_w) {
            w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstBinding = b->binding, .descriptorCount = cnt, .descriptorType = b->type};
            if (kind == 0) w[i].pBufferInfo = &o->buffer;   // a VkDescSlot is as large as either info struct
            else if (kind == 1) w[i].pImageInfo = &o->image;
            else w[i].pTexelBufferView = &bv[nd];
        }
        nd += cnt;
    }
    // Storage images the shaders store into: a render into them for the texture bookkeeping (vk_texture_gpu_written_id),
    // so a mirrored one is not uploaded from its storage over the stores and its storage gets the result back.
    for (int k = 0; k < nstored; k++) vk_texture_gpu_written_id(stored[k]);
    // Under the validation layer, a new graphics layout binds set 1 before set 0 is pushed or bound. The layer (SDK
    // 1.4.363, PipelineLayoutCompatDef::operator==) takes any two INDEPENDENT_SETS layouts as compatible, so
    // vkCmdPushDescriptorSetKHR keeps the push set it made for an earlier layout and drops writes to bindings that
    // layout lacks (false VUID-vkCmdDrawIndexed-None-08114, then an access violation inside the layer). A bind at set 1
    // disturbs set 0 by its canonical id, which drops the layer's push set: the push after it gets a new one.
    int set1_first = 0;
    if (gfx && info->layout != B.ds_layout) {   // another layout: sets bound before may be disturbed
        B.ds_layout = info->layout;
        B.ds_bound[0] = B.ds_bound[1] = VK_NULL_HANDLE;
        B.ds_pushed[0] = B.ds_pushed[1] = 0;
        set1_first = n_sets == 2 && vk_ctx()->validation;
    }
    for (uint32_t k = 0; k < n_sets; k++) {
        uint32_t s = set1_first ? 1 - k : k, i0 = lo[s], i1 = hi[s];
        const VkDescSlot *ds = &d[d0[s]];
        uint32_t dbytes = (uint32_t)(((s + 1 < n_sets ? d0[s + 1] : nd) - d0[s]) * sizeof *d);
        if (s && i1 == i0) {               // no fragment stage: set 1 is empty and unused
            if (set1_first) {              // bound only to drop the layer's push set (comment above)
                VkDescriptorSet set = alloc_set(ex, info->set_layout1);
                vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, 1, 1, &set, 0, NULL);
                B.ds_bound[1] = set, B.ds_pushed[1] = 0;
                RS.cmds++;
            }
            continue;
        }
        // A stage without resources (this game's transform feedback vertex shaders read attributes only) has nothing
        // to push: a push of no bytes hands the driver a NULL pData, which it faults on (nvoglv64.dll, found with the
        // validation layer's VUID-vkCmdPushDescriptorSetWithTemplate-pData-01686).
        if (i1 == i0 || !dbytes) continue;
        if (s ? info->push1 : info->push) {
            uint32_t n = gfx && push_skip_on() ? dbytes : 0;
            if (n && B.ds_pushed[s] && DS.pn[s] == n && !memcmp(DS.psig[s], ds, n)) {
                BS.push_skip++;
                continue;
            }
            if (info->tpl[s]) rec_push_tpl(vk_pipeline_push_tpl_fn(), ex->cb, info->tpl[s], info->layout, s, ds, dbytes);
            else rec_push(vk_pipeline_push_fn(), ex->cb, info->bind_point, info->layout, s, i1 - i0, w + i0);
            BS.push++;
            RS.cmds++;
            if (gfx) {
                B.ds_bound[s] = VK_NULL_HANDLE, B.ds_pushed[s] = n != 0;
                if (n) DS.pn[s] = n, memcpy(DS.psig[s], ds, n);
            }
        } else if (!gfx || s > 1) {
            VkDescriptorSet set = alloc_set(ex, s ? info->set_layout1 : info->set_layout);
            write_set(info, s, set, ds, w, i0, i1);
            vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, s, 1, &set, 0, NULL);
        } else {
            VkDescriptorSetLayout sl = s ? info->set_layout1 : info->set_layout;
            VkDescriptorSet set = B.ds_last[s];
            if (!set || B.ds_last_layout[s] != sl || DS.n[s] != dbytes || memcmp(DS.sig[s], ds, dbytes)) {
                set = alloc_set(ex, sl);
                write_set(info, s, set, ds, w, i0, i1);
                B.ds_last[s] = set, B.ds_last_layout[s] = sl, DS.n[s] = dbytes;
                memcpy(DS.sig[s], ds, dbytes);
                BS.set_new++;
                RS.cmds++;
            } else {
                BS.set_reuse++;
            }
            if (B.ds_bound[s] != set) {
                vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, s, 1, &set, 0, NULL);
                B.ds_bound[s] = set, B.ds_pushed[s] = 0;
                RS.cmds++;
            }
        }
    }
    if (info->pool_mask) pool_bind(ex, info);
    else if (gfx && B.pool_layout != info->layout) B.pool_layout = VK_NULL_HANDLE;   // sets bound with another layout may
                                                                                     // have disturbed set GFX_POOL_SET
    if (skip) {
        RS.rec = 0;
        // An equal next draw records nothing when every push set holds its push (OOT_PUSH_SKIP on) and nothing made
        // the build unrepeatable; allocated sets are always bound to the one written last.
        int valid = !RS.volatile_ && !RS.cnt_over && RS.hn;
        for (uint32_t s = 0; s < n_sets; s++)
            if (hi[s] > lo[s] && (s ? info->push1 : info->push) && !B.ds_pushed[s]) valid = 0;
        if (skip == 2) {   // with the pipeline info equal (bind_sig), the packed descriptors determine every write
            uint32_t n = (uint32_t)(nd * sizeof *d);
            int wr = n != RS.ckn || memcmp(RS.ck, d, n);
            int cn = RS.ncnt != RS.pncnt || memcmp(RS.cnt, RS.pcnt, sizeof RS.cnt[0] * (size_t)RS.ncnt);
            if (same && was_valid && (wr || cn || RS.cmds))
                rt_fatal("vk_exec: descriptor build skip check: a draw of program %llu with the inputs of the previous one "
                         "gets %s", (unsigned long long)ex->prog->serial,
                         RS.cmds ? "descriptor commands" : wr ? "other writes" : "other census counts");
            if (same && was_valid) BS.bind_skip++;   // the draws a skip would have spared, compared instead
            RS.ckn = n;
            memcpy(RS.ck, d, n);
        }
        B.res_valid = valid;
    }
    ex->recorded++;
}

// ---- draws

static void bind_pipeline(VkExec *ex, VkPipelineBindPoint bp, VkPipeline p) {
    int k = bp == VK_PIPELINE_BIND_POINT_COMPUTE;
    if (B.pipe[k] == p) return;
    vkCmdBindPipeline(ex->cb, bp, p);
    ex->recorded++;
    B.pipe[k] = p;
    if (!k) B.st_valid = 0;
}

static void record_fixed(VkExec *ex) {
    VkCommandBuffer cb = ex->cb;
    if (ex->dirty & VK_EXEC_DIRTY_VIEWPORT) {
        float x = 0, y = 0, w = (float)ex->pass_w, h = (float)ex->pass_h;
        if (ex->has_vp) x = ex->vp[0][0], y = ex->vp[0][1], w = ex->vp[0][2], h = ex->vp[0][3];
        if (w < 1) w = 1;
        if (h < 1) h = 1;
        // Window y is the memory row in both NVN origin modes; they differ in where NDC +y lands. Upper-left (TotK):
        // at window y 0, the first row, so the viewport gets a negative height, which is how Vulkan puts NDC +y at the
        // first row. Lower-left (NVN's default, this game): at the last row, as GL renders into a texture, so the
        // viewport keeps Vulkan's own orientation; the picture then sits bottom-up in memory, which the game's
        // texture reads expect and the present blit turns over (vk_present_blit).
        VkViewport v = nvn_origin_lower_left() ? (VkViewport){x, y, w, h, ex->depth_range[0], ex->depth_range[1]}
                                               : (VkViewport){x, y + h, w, -h, ex->depth_range[0], ex->depth_range[1]};
        vkCmdSetViewportWithCount(cb, 1, &v);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_SCISSOR) {
        VkRect2D r = vk_exec_scissor(ex);
        vkCmdSetScissorWithCount(cb, 1, &r);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_DEPTH_BIAS) {
        vkCmdSetDepthBias(cb, ex->poly_offset.units, ex->poly_offset.clamp, ex->poly_offset.factor);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_BLEND_CONST) {
        vkCmdSetBlendConstants(cb, ex->blend_color);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_STENCIL) {
        for (int f = 0; f < 2; f++) {
            VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
            vkCmdSetStencilCompareMask(cb, face, ex->st_vmask[f]);
            vkCmdSetStencilWriteMask(cb, face, ex->st_wmask[f]);
            vkCmdSetStencilReference(cb, face, ex->st_ref[f]);
            ex->recorded += 3;
        }
    }
    ex->dirty = 0;
}

static void bind_vertex_buffers(VkExec *ex) {
    VkBuffer vb[17];
    VkDeviceSize vo[17], vsz[17], vstr[17];
    int vcls[17] = {0};
    void *vpool[17] = {0};
    int dyn = vk_pipeline_vertex_input_dynamic();
    // A stream whose range is the one its bound buffer was resolved from, with no pool or mirror change since
    // (vk_mem_gen, read before any resolve), resolves to that buffer again: most draws change one stream or none.
    uint64_t gen = vk_mem_gen();
    int same_gen = B.vb_valid && B.vgen == gen;
    for (int b = 0; b < 17; b++) {
        vb[b] = Z.zbuf, vo[b] = 0, vsz[b] = ZERO_BYTES, vstr[b] = b < 16 ? ex->pipe.vstream[b].stride : 0;
        if (b == VK_PIPE_ZERO_BINDING) continue;
        const NVNbufferRange *r = &ex->vbuf[b];
        if (same_gen && B.vin[b].address == r->address && B.vin[b].size == r->size) {
            vb[b] = B.vb[b], vo[b] = B.vo[b], vsz[b] = B.vsz[b], vcls[b] = B.vcls[b], vpool[b] = B.vpool[b];
            continue;
        }
        VkMemSpan sp;
        if (!r->address || !vk_mem_resolve(r->address, 1, &sp)) continue;
        VK_MEM_USE(sp);
        uint64_t n = r->size && r->size < sp.avail ? r->size : sp.avail;
        vb[b] = sp.buffer, vo[b] = sp.offset, vsz[b] = n, vcls[b] = sp.cls, vpool[b] = sp.pool;
    }
    memcpy(B.vcls, vcls, sizeof vcls), memcpy(B.vpool, vpool, sizeof vpool);
    memcpy(B.vin, ex->vbuf, sizeof B.vin);
    B.vgen = gen;
    if (B.vb_valid && !memcmp(B.vb, vb, sizeof vb) && !memcmp(B.vo, vo, sizeof vo) && !memcmp(B.vsz, vsz, sizeof vsz) &&
        (dyn || !memcmp(B.vstr, vstr, sizeof vstr)))
        return;
    vkCmdBindVertexBuffers2(ex->cb, 0, 17, vb, vo, vsz, dyn ? NULL : vstr);
    ex->recorded++;
    for (int b = 0; b < 17; b++)
        if (vcls[b]) vk_mem_count(vcls[b], VK_MUSE_VB, vsz[b]);
    if (rt_trace & RT_TRACE_PERF)
        for (int b = 0; b < 17; b++)
            if (vpool[b]) vk_mem_bound_note(vpool[b]);
    memcpy(B.vb, vb, sizeof vb), memcpy(B.vo, vo, sizeof vo), memcpy(B.vsz, vsz, sizeof vsz), memcpy(B.vstr, vstr, sizeof vstr);
    B.vb_valid = 1;
}

// OOT_PIPE_SKIP: which draws are skipped, rather than waited for, while their shaders or pipeline build.
// strict (0): only draws whose pass targets were all rendered in the previous presented frame; a pass with a target
//   that was not counts as one-shot and waits.
// any (1): also draws in a pass with at least one target rendered in the previous frame (a per-frame depth
//   buffer or g-buffer with a colour target the frame before did not use: half-resolution effect buffers in fights).
//   Passes with none (volumes, LUTs, shadow layers seen for the first time) still wait.
// fullscreen (2, default): as any, and full-screen draws into such passes are skipped instead of waiting from their
//   budget. A full-screen pass whose stage libraries are cold in the driver cache took 86 ms on average (up to 194 ms)
//   in a fight on an RX 9070: three frames in a row each spent the 50 ms budget, and some of the draws were skipped
//   after all. Skipping costs the pass for the frames until the pipeline is ready.
static int pipe_skip_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = cfg_value("OOT_PIPE_SKIP");
        mode = !strcmp(e, "strict") || !strcmp(e, "0") ? 0 : !strcmp(e, "fullscreen") || !strcmp(e, "2") ? 2 : 1;
        static const char *const what[3] = {"only draws into per-frame passes (OOT_PIPE_SKIP=strict)",
                                            "draws into passes with a target the previous frame rendered",
                                            "those and full-screen draws (OOT_PIPE_SKIP=fullscreen)"};
        rt_log("vk_exec: draws skipped while their pipeline builds: %s", what[mode]);
    }
    return mode;
}

// The previous draw's shaders, kept while its program, bound pipeline state, the vertex and fragment stages' first 32
// texture and image handles and every slot's resolution (vk_texture_slots_gen, read unlocked: a registration racing
// with it is ordered after this draw, as answers_for's own check would order it) stay the same: answers_for reads
// nothing else, so shader_for would return the same shaders. About 65% of field draws repeat them
// (OOT_BIND_STATS). OOT_SHADER_MEMO=0 turns it off; =check looks the shaders up on every hit too, and every slot's
// texture type in answers_for, and stops on a difference.
// tex_serial (bumped by every texture and image bind) equal to the key's means the handles were not rebound since
// they were compared, so the 1 KiB compare is skipped; a rebind to the same handles still matches by the compare.
typedef struct {
    const prog_t *prog;
    uint64_t serial;
    uint32_t gen, tex_serial;
    uint64_t tex[2][32], img[2][32];
    VkPipeState pipe;
} MemoKey;
static struct { int on, valid, has_fs; MemoKey k; const GfxShader *vs, *fs; } SM = {-1};

static void memo_key(const VkExec *ex, uint32_t gen, MemoKey *k) {
    k->prog = ex->prog, k->serial = ex->prog->serial, k->gen = gen, k->tex_serial = ex->tex_serial;
    memcpy(k->tex[0], ex->tex[NVN_STAGE_VS], sizeof k->tex[0]), memcpy(k->tex[1], ex->tex[NVN_STAGE_FS], sizeof k->tex[1]);
    memcpy(k->img[0], ex->img[NVN_STAGE_VS], sizeof k->img[0]), memcpy(k->img[1], ex->img[NVN_STAGE_FS], sizeof k->img[1]);
    memcpy(&k->pipe, &ex->pipe, sizeof k->pipe);
}

static int memo_match(const VkExec *ex, uint32_t gen, MemoKey *k) {
    if (k->prog != ex->prog || k->serial != ex->prog->serial || k->gen != gen || memcmp(&k->pipe, &ex->pipe, sizeof k->pipe))
        return 0;
    if (k->tex_serial == ex->tex_serial) return 1;
    if (memcmp(k->tex[0], ex->tex[NVN_STAGE_VS], sizeof k->tex[0]) || memcmp(k->tex[1], ex->tex[NVN_STAGE_FS], sizeof k->tex[1]) ||
        memcmp(k->img[0], ex->img[NVN_STAGE_VS], sizeof k->img[0]) || memcmp(k->img[1], ex->img[NVN_STAGE_FS], sizeof k->img[1]))
        return 0;
    k->tex_serial = ex->tex_serial;
    return 1;
}

static int shader_memo_on(void) {
    if (SM.on < 0) {
        const char *e = cfg_value("OOT_SHADER_MEMO");
        SM.on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
    }
    return SM.on;
}

static struct { int on, valid; VkPipeState s; const VkPipeInfo *info; } PM = {-1};

static int pipe_memo_on(void) {
    if (PM.on < 0) PM.on = cfg_bool("OOT_PIPE_MEMO");
    return PM.on;
}

// Pass, shaders, pipeline, fixed state and resources for a draw. 0: the draw is skipped.
static int prepare_draw(VkExec *ex, int prim, int32_t count, int restart) {
    if (ex->render_disabled) return 0;
    cb_res(ex);
    dummies_init();
    if (ex->pass_noatt) {   // no attachment: the pass covers this draw's viewport or scissor, reopened when it needs more
        uint32_t w, h;
        vk_exec_noatt_dims(ex, &w, &h);
        if (ex->in_pass && (w > ex->pass_w || h > ex->pass_h)) vk_exec_pass_end(ex);
        if (!ex->in_pass) ex->rt_resolved = 0;
    }
    if (!ex->in_pass) ex->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
    vk_exec_pass_begin(ex);
    VkRect2D sc = vk_exec_scissor(ex);
    if (!sc.extent.width || !sc.extent.height) return 0;
    if (!ex->prog) return warn_once(W_PROGRAM, "a draw without a bound program is skipped"), 0;
    int has_vs, has_fs, vs_wait = 0, fs_wait = 0, memo = shader_memo_on();
    const GfxShader *vs, *fs;
    uint32_t gen = memo ? vk_texture_slots_gen() : 0;   // before shader_for, whose lookups may bump it
    if (memo && SM.valid && memo_match(ex, gen, &SM.k)) {
        vs = SM.vs, fs = SM.fs, has_vs = 1, has_fs = SM.has_fs;
        if (memo == 2) {
            int hv, hf, wv, wf;
            const GfxShader *v2 = shader_for(ex, NVN_STAGE_VS, &hv, &wv, 0), *f2 = shader_for(ex, NVN_STAGE_FS, &hf, &wf, 0);
            if (v2 != vs || f2 != fs || hf != has_fs || wv || wf)
                rt_fatal("vk_exec: shader memo of program %llu differs from the lookup", (unsigned long long)ex->prog->serial);
        }
    } else {
        SM.valid = 0;
        vs = shader_for(ex, NVN_STAGE_VS, &has_vs, &vs_wait, 0);
        fs = shader_for(ex, NVN_STAGE_FS, &has_fs, &fs_wait, 0);
        if (memo && vs && !vs_wait && !fs_wait && (!has_fs || fs))
            memo_key(ex, gen, &SM.k), SM.vs = vs, SM.fs = fs, SM.has_fs = has_fs, SM.valid = 1;
    }
    if (!has_vs) return warn_once(W_NO_VS, "a draw with a program that has no vertex stage is skipped"), 0;
    // A draw into a target the previous frame did not render (or into none) waits for a pending shader or pipeline:
    // one-shot draws (volumes and LUTs generated once, later only sampled) would leave their target empty for the
    // session. A draw into per-frame targets is skipped until they are ready and comes back next frame.
    int oneshot = -1, skip = pipe_skip_mode();   // -1: not looked up
    if (vs_wait || fs_wait) {
        if (!(oneshot = !vk_exec_targets_recent(ex, skip >= 1))) return perf_count(PERF_CNT_SHADER_SKIP), 0;   // skipped until ready
        perf_count(PERF_CNT_ONESHOT_WAIT);
        if (vs_wait) vs = shader_for(ex, NVN_STAGE_VS, &has_vs, NULL, 1);
        if (fs_wait) fs = shader_for(ex, NVN_STAGE_FS, &has_fs, NULL, 1);
    }
    if (!vs || (has_fs && !fs)) {
        if (vk_dump_armed) {   // which program the draw needed, by the code hash tools/kernel.py prints
            char hv[2][17] = {"-", "-"};
            for (int k = 0; k < 2; k++) {
                const uint8_t *ctl, *code;
                uint32_t len;
                if (!ex->prog || !vk_exec_find_stage(ex->prog, k ? NVN_STAGE_FS : NVN_STAGE_VS, &ctl, &code, &len)) continue;
                uint64_t h = 0xcbf29ce484222325ull;
                for (uint32_t i = 0; i < len; i++) h = (h ^ code[i]) * 0x100000001b3ull;
                snprintf(hv[k], sizeof hv[k], "%016llx", (unsigned long long)h);
            }
            vk_dump_note("draw of %u vertices skipped: no %s shader (vs code %s%s, fs code %s%s), pass %ux%u", count,
                         !vs ? "vertex" : "fragment", hv[0], vs ? " ok" : " missing", hv[1], fs ? " ok" : has_fs ? " missing" : "",
                         ex->pass_w, ex->pass_h);
        }
        return 0;
    }
    uint32_t fbl = feedback_prepare(ex, fs);
    VkPipeState s = ex->pipe;
    s.vs = vs, s.fs = fs;
    s.feedback = (uint8_t)((fbl & 0xff ? 1 : 0) | (fbl & 0x100 ? 2 : 0));
    s.prim = (uint8_t)prim;
    s.depth_minus_w = (uint8_t)(nvn_depth_minus_w() != 0);
    s.y_flip = 1;
    s.prim_restart = (uint8_t)(restart != 0);
    s.xfb = (uint8_t)(ex->xfb_active != 0);   // a capture draw gets the vertex shader with the transform feedback layout
    const VkPipeInfo *info = NULL;
    int pending;
    // A draw of at most 6 vertices over half the render area or more (a full-screen triangle or quad, a tile) waits for
    // its pipeline: skipping it can leave a black tile. The waits share 50 ms per presented frame
    // (OOT_PIPE_SKIP=fullscreen skips them instead). A one-shot draw waits without that budget. Everything else is
    // skipped while its pipeline is built.
    static uint64_t wait_frame;
    static uint64_t wait_left_ns;
    uint32_t wait_ms = 0;
    uint64_t t0 = 0;
    if (oneshot > 0 || ex->xfb_active) {   // a capture draw waits too: skipped, the CPU would read back nothing
        wait_ms = ONESHOT_WAIT_MS;
    } else if (skip < 2 && count <= 6 && (uint64_t)sc.extent.width * sc.extent.height * 2 >= (uint64_t)ex->pass_w * ex->pass_h) {
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (f != wait_frame) wait_frame = f, wait_left_ns = 50000000;
        wait_ms = (uint32_t)(wait_left_ns / 1000000);
        if (wait_ms) t0 = kern_now_ns();
    }
    // The previous draw's pipeline while its state repeats (PM, OOT_PIPE_MEMO=0 turns it off): the cache key is made
    // from the state alone, and a pipeline once found only changes from its fast link to its optimised build, read
    // from the info the cache entry keeps.
    VkPipeline p;
    const char *why = NULL;
    uint64_t w0 = 0;
    if (pipe_memo_on() && PM.valid && !memcmp(&PM.s, &s, sizeof s)) {
        info = PM.info, pending = 0;
        p = __atomic_load_n(&info->pipeline, __ATOMIC_ACQUIRE);
    } else {
        w0 = t0 ? t0 : wait_ms ? kern_now_ns() : 0;   // the clock is read only for draws that may wait
        p = vk_pipeline_graphics_async(&s, &info, &pending, wait_ms);
        if (t0) {
            uint64_t spent = kern_now_ns() - t0;
            wait_left_ns = spent < wait_left_ns ? wait_left_ns - spent : 0;
        }
        why = oneshot > 0 ? "one-shot, shader" : wait_ms ? "full-screen" : NULL;
        if (pending && oneshot < 0 && !vk_exec_targets_recent(ex, skip >= 1)) {
            perf_count(PERF_CNT_ONESHOT_WAIT);
            why = "one-shot";
            if (!w0) w0 = kern_now_ns();
            p = vk_pipeline_graphics_async(&s, &info, &pending, ONESHOT_WAIT_MS);
        }
        PM.valid = !pending && p && info;
        if (PM.valid) PM.s = s, PM.info = info;
    }
    uint64_t waited = w0 ? kern_now_ns() - w0 : 0;
    static int wait_logs;
    if (why && waited >= 2000000 && wait_logs < 200) {
        char tg[512];
        wait_logs++;
        vk_exec_targets_describe(ex, tg, sizeof tg);
        rt_log("vk_exec: frame %llu: a draw waited %.1f ms for its pipeline (%s%s): %u vertices, pass %ux%u, %s",
               (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1, waited / 1e6, why,
               pending ? ", skipped after all" : "", count, ex->pass_w, ex->pass_h, tg);
    }
    if (pending) return 0;   // a background worker is building it: the draw is skipped until it is ready
    if (!p || !info) return warn_once(W_PIPELINE, "a graphics pipeline could not be created: draws with it are skipped"), 0;
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    if (!B.st_valid || memcmp(&B.st, &s, sizeof s)) {
        vk_pipeline_set_dynamic(ex->cb, &s);
        ex->recorded++;
        B.st = s, B.st_valid = 1;
    }
    record_fixed(ex);
    repeat_stats(ex, info);
    bind_resources(ex, info);
    if (fbl & ex->fb_dirty) feedback_barrier(ex);
    feedback_written(ex, &s);
    bind_vertex_buffers(ex);
    if (vk_dump_armed) vk_dump_draw(ex, &s, p, info);
    vk_exec_oq_draw(ex);
    return 1;
}

static void bind_index(VkExec *ex, const void *data, uint64_t bytes, VkIndexType type) {
    VkBuffer b;
    VkDeviceSize o;
    memcpy(scratch(ex, bytes, &b, &o), data, bytes);
    vkCmdBindIndexBuffer(ex->cb, b, o, type);
    ex->recorded++;
}

// OOT_DEPTH_EQ_LE=3 (diagnostic, vk_pipeline.c): a draw that tests depth EQUAL is recorded twice, the left half of
// its scissor with LESS_OR_EQUAL, the right half with EQUAL, so one frame compares both. Returns 1 when it drew.
#define EQ_SPLIT(ex, DRAW)                                                                                          \
    do {                                                                                                            \
        if (vk_depth_eq_split) {                                                                                    \
            VkRect2D r0 = vk_exec_scissor(ex), h = r0;                                                              \
            uint32_t mid = (uint32_t)(ex->has_vp ? ex->vp[0][0] + ex->vp[0][2] / 2 : (float)(r0.offset.x + r0.extent.width / 2)); \
            int32_t m = (int32_t)mid < r0.offset.x ? r0.offset.x : (int32_t)mid > r0.offset.x + (int32_t)r0.extent.width ? r0.offset.x + (int32_t)r0.extent.width : (int32_t)mid; \
            h.extent.width = (uint32_t)(m - r0.offset.x);                                                           \
            vkCmdSetScissorWithCount(ex->cb, 1, &h);                                                                \
            vkCmdSetDepthCompareOp(ex->cb, VK_COMPARE_OP_LESS_OR_EQUAL);                                            \
            if (h.extent.width) DRAW;                                                                               \
            h.offset.x = m, h.extent.width = r0.extent.width - (uint32_t)(m - r0.offset.x);                         \
            vkCmdSetScissorWithCount(ex->cb, 1, &h);                                                                \
            vkCmdSetDepthCompareOp(ex->cb, VK_COMPARE_OP_EQUAL);                                                    \
            if (h.extent.width) DRAW;                                                                               \
            vkCmdSetScissorWithCount(ex->cb, 1, &r0);                                                               \
            ex->recorded += 6;                                                                                      \
        } else {                                                                                                    \
            DRAW;                                                                                                   \
        }                                                                                                           \
    } while (0)

static void op_draw(VkExec *ex, const p_draw *d) {
    c0_binst = (uint32_t)d->base_instance;
    if (d->count <= 0 || !prepare_draw(ex, d->prim, d->count, 0)) return;
    xfb_before_draw(ex);
    if (vk_dump_armed) vk_dump_draw_params(ex, 0, d->prim, d->first, d->count, d->instances, d->base_instance, 0);
    vk_exec_frame.draws++;
    uint32_t inst = d->instances > 0 ? (uint32_t)d->instances : 1, first = (uint32_t)d->first, n = (uint32_t)d->count;
    if (d->prim == NVN_DRAW_PRIMITIVE_QUADS && n >= 4) {   // quads become two triangles each
        uint32_t q = n / 4;
        uint32_t *ix = malloc(sizeof *ix * 6 * q);
        if (!ix) rt_fatal("vk: out of memory");
        for (uint32_t i = 0; i < q; i++) {
            uint32_t v = first + 4 * i, k[6] = {0, 1, 2, 0, 2, 3};
            for (int j = 0; j < 6; j++) ix[6 * i + j] = v + k[j];
        }
        bind_index(ex, ix, sizeof *ix * 6 * q, VK_INDEX_TYPE_UINT32);
        free(ix);
        vkCmdDrawIndexed(ex->cb, 6 * q, inst, 0, 0, (uint32_t)d->base_instance);
    } else if (d->prim == NVN_DRAW_PRIMITIVE_LINE_LOOP && n >= 2) {   // a strip that returns to its first vertex
        uint32_t *ix = malloc(sizeof *ix * (n + 1));
        if (!ix) rt_fatal("vk: out of memory");
        for (uint32_t i = 0; i < n; i++) ix[i] = first + i;
        ix[n] = first;
        bind_index(ex, ix, sizeof *ix * (n + 1), VK_INDEX_TYPE_UINT32);
        free(ix);
        vkCmdDrawIndexed(ex->cb, n + 1, inst, 0, 0, (uint32_t)d->base_instance);
    } else {
        if (M.drawing && M.mi) EQ_SPLIT(ex, vkCmdDrawIndirect(ex->cb, MI.buf, M.mi_off + 16, 1, sizeof(VkDrawIndirectCommand)));
        else EQ_SPLIT(ex, vkCmdDraw(ex->cb, n, inst, first, (uint32_t)d->base_instance));
    }
    ex->recorded++;
    if (vk_dump_armed) vk_dump_after_draw(ex);
}

// Primitive restart of an indexed draw. NVN restarts strips wherever an index equals the SetPrimitiveRestart value;
// Vulkan restarts only at the all-ones value of the bound index type, and only strips and fans without
// primitiveTopologyListRestart. Byte indices are widened to 16 bits, so their restart value becomes 0xffff
// (op_draw_idx). Without restart the 0xffff entries of TOTK's terrain and cliff strips fetch vertex 65535 + base and
// the triangles around them drop out.
static int restart_for(const VkExec *ex, int prim, int itype) {
    if (!ex->prim_restart.enable) return 0;
    switch (vk_pipeline_topology(prim)) {
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY: break;
    default: return 0;   // lists: a restart index only drops the unfinished primitive
    }
    static const uint32_t all_ones[3] = {0xffu, 0xffffu, 0xffffffffu};
    if (ex->prim_restart.index == all_ones[itype]) return 1;
    warn_once(W_RESTART, "primitive restart index %#x with %d-byte indices is not replayed: strips drawn without restart",
              ex->prim_restart.index, 1 << itype);
    return 0;
}

static void op_draw_idx(VkExec *ex, const p_draw_idx *d) {
    c0_binst = (uint32_t)d->base_instance;
    static const uint32_t isize[3] = {1, 2, 4};
    if (d->itype < 0 || d->itype > 2) return warn_once(W_INDEX, "index type %d is unknown: draw skipped", d->itype);
    int restart = restart_for(ex, d->prim, d->itype);
    int probe = vk_dump_armed && !ex->xfb_active && d->count <= 64 ? vk_dump_probe_draw() : -1;   // sprites only
    if (probe < 0 && vk_dump_armed && !ex->xfb_active) probe = vk_dump_probe_count(d->count);
    if (probe >= 0) ex->xfb_active = 1;   // prepare_draw picks the pipeline with the capture layout
    int ok = d->count > 0 && prepare_draw(ex, d->prim, d->count, restart);
    if (probe >= 0) ex->xfb_active = 0;
    if (!ok) return;
    xfb_before_draw(ex);
    vk_exec_frame.draws++;
    uint64_t bytes = (uint64_t)d->count * isize[d->itype];
    VkMemSpan sp;
    if (!vk_mem_resolve(d->ibuf, bytes, &sp)) return warn_once(W_INDEX, "index buffer %#llx is not backed: draws with it are skipped", (unsigned long long)d->ibuf);
    if (d->itype == NVN_INDEX_TYPE_UNSIGNED_BYTE) {   // no uint8 index type here: widen (host-readable backing only)
        if (sp.alloc) return warn_once(W_INDEX, "byte indices in device-local memory: draw skipped");
        vk_mem_count(sp.cls, VK_MUSE_CPU_READ, bytes);
        vk_mem_count(VK_MCLS_SCRATCH, VK_MUSE_IB, 2 * bytes);
        uint16_t *ix = malloc(sizeof *ix * (size_t)d->count);
        if (!ix) rt_fatal("vk: out of memory");
        for (int i = 0; i < d->count; i++) {
            uint8_t v = ((const uint8_t *)(uintptr_t)d->ibuf)[i];
            ix[i] = restart && v == 0xff ? 0xffff : v;
        }
        bind_index(ex, ix, sizeof *ix * (size_t)d->count, VK_INDEX_TYPE_UINT16);
        free(ix);
    } else if (sp.offset % isize[d->itype]) {
        return warn_once(W_INDEX, "index buffer %#llx is not aligned to its index size: draw skipped", (unsigned long long)d->ibuf);
    } else {
        VK_MEM_USE(sp);
        vkCmdBindIndexBuffer(ex->cb, sp.buffer, sp.offset, d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        ex->recorded++;
        vk_mem_count(sp.cls, VK_MUSE_IB, bytes);
        VK_MEM_BOUND(sp);
    }
    if (vk_dump_armed) vk_dump_draw_params(ex, 1 + d->itype, d->prim, 0, d->count, d->instances, d->base_instance, d->base_vertex), vk_dump_indices(d->ibuf, d->itype, d->count);
    ProbeJob *pj = probe >= 0 ? probe_begin(ex, probe) : NULL;
    EQ_SPLIT(ex, vkCmdDrawIndexed(ex->cb, (uint32_t)d->count, d->instances > 0 ? (uint32_t)d->instances : 1, 0,
                                  d->base_vertex, (uint32_t)d->base_instance));
    ex->recorded++;
    probe_end(ex, pj);
    if (vk_dump_armed) vk_dump_after_draw(ex);
}

// DrawArraysIndirect: as DrawElementsIndirect below, without the index buffer (vkCmdDrawIndirect from the pool buffer,
// through the recorder like every other command: vk_rec.h wraps it, or the call would go to the driver on the game's
// thread while the recorder records the same command buffer, which lost the device in FE3H's first battle).
static void op_draw_ind(VkExec *ex, const p_draw_ind *d) {
    c0_binst = 0;
    VkMemSpan ip;
    if (!vk_mem_resolve(d->ind, sizeof(VkDrawIndirectCommand), &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect draw arguments at %#llx are not backed or not aligned: draws skipped", (unsigned long long)d->ind);
    if (!prepare_draw(ex, d->prim, 0x7fffffff, 0)) return;
    xfb_before_draw(ex);
    vk_exec_frame.draws++;
    VK_MEM_USE(ip);
    vkCmdDrawIndirect(ex->cb, ip.buffer, ip.offset, 1, sizeof(VkDrawIndirectCommand));
    ex->recorded += 1;
    vk_mem_count(ip.cls, VK_MUSE_IND, sizeof(VkDrawIndirectCommand));
    VK_MEM_BOUND(ip);
}

// DrawElementsIndirect: the arguments are read by the GPU from the pool buffer (they are often written by an earlier
// dispatch), as nxnative nvn_mtl.m does. The base instance stays out of c0 (0x644): c0 cannot be patched by a transfer
// inside a pass; gl_BaseInstance of the Vulkan draw is the argument's. Byte indices need the count to be widened: skipped.
static void op_draw_idx_ind(VkExec *ex, const p_draw_idx_ind *d) {
    c0_binst = 0;
    if (d->itype != NVN_INDEX_TYPE_UNSIGNED_SHORT && d->itype != NVN_INDEX_TYPE_UNSIGNED_INT)
        return warn_once(W_INDEX, "indirect draw with index type %d: skipped", d->itype);
    VkMemSpan ib, ip;
    if (!vk_mem_resolve(d->ind, sizeof(VkDrawIndexedIndirectCommand), &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect draw arguments at %#llx are not backed or not aligned: draws skipped", (unsigned long long)d->ind);
    if (!vk_mem_resolve(d->ibuf, 1, &ib) || ib.offset % (d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 4))
        return warn_once(W_INDEX, "index buffer %#llx is not backed or not aligned: indirect draws skipped", (unsigned long long)d->ibuf);
    if (!prepare_draw(ex, d->prim, 0x7fffffff, restart_for(ex, d->prim, d->itype))) return;
    xfb_before_draw(ex);
    vk_exec_frame.draws++;
    VK_MEM_USE(ip);
    VK_MEM_USE(ib);
    vkCmdBindIndexBuffer(ex->cb, ib.buffer, ib.offset, d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirect(ex->cb, ip.buffer, ip.offset, 1, sizeof(VkDrawIndexedIndirectCommand));
    ex->recorded += 2;
    vk_mem_count(ip.cls, VK_MUSE_IND, sizeof(VkDrawIndexedIndirectCommand));
    VK_MEM_BOUND(ip);
    VK_MEM_BOUND(ib);
}

// ---- dispatch and uniform buffer updates

// OOT_CS_TIME=1 (diagnostic): the GPU time of every compute dispatch, from timestamps around it, summed per shader
// (its input id) and logged every CST_REPORT timed dispatches, the costliest first. For finding which kernel of a
// frame the GPU spends its time in (the title cinematic's upscaler: 49 ms of GPU time per frame, 2026-11-06).
enum { CST_PAIRS = 2048, CST_SHADERS = 1024, CST_REPORT = 6000, CST_LINES = 28 };   // 64 shaders hid the mesh programs, the first to overflow it
typedef struct { uint8_t iid[8]; uint32_t groups[3]; uint64_t n; double ns; } CstShader;
typedef struct { uint32_t pair; uint8_t iid[8]; uint32_t groups[3]; } CstRead;
static VkQueryPool cst_pool;
static uint32_t cst_next;
static volatile LONG cst_busy[CST_PAIRS];
static CstShader cst_sh[CST_SHADERS];
static int cst_nsh, cst_reads;
static SRWLOCK cst_lock = SRWLOCK_INIT;

static int cst_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("OOT_CS_TIME");
        on = e && atoi(e) ? 1 : 0;
        if (on) {
            VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                        .queryCount = 2 * CST_PAIRS};
            if (vkCreateQueryPool(vk_ctx()->device, &qi, NULL, &cst_pool) != VK_SUCCESS) on = 0;
            rt_log("vk_exec: OOT_CS_TIME: %s", on ? "compute dispatches timed on the GPU" : "no query pool, not timed");
        }
    }
    return on;
}

static int cst_cmp(const void *a, const void *b) {
    double x = ((const CstShader *)a)->ns, y = ((const CstShader *)b)->ns;
    return x < y ? 1 : x > y ? -1 : 0;
}

static void cst_read(void *arg) {   // completion thread
    CstRead *r = arg;
    uint64_t v[2];
    VkPhysicalDeviceProperties pr;
    static double period;
    if (!period) vkGetPhysicalDeviceProperties(vk_ctx()->phys, &pr), period = pr.limits.timestampPeriod;
    int ok = vkGetQueryPoolResults(vk_ctx()->device, cst_pool, 2 * r->pair, 2, sizeof v, v, sizeof v[0],
                                   VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && v[1] >= v[0];
    InterlockedExchange(&cst_busy[r->pair], 0);
    if (!ok) { free(r); return; }
    AcquireSRWLockExclusive(&cst_lock);
    int i = 0;
    while (i < cst_nsh && memcmp(cst_sh[i].iid, r->iid, 8)) i++;
    if (i == cst_nsh && cst_nsh < CST_SHADERS) memcpy(cst_sh[cst_nsh++].iid, r->iid, 8);
    if (i < cst_nsh) {
        cst_sh[i].n++, cst_sh[i].ns += (double)(v[1] - v[0]) * period;
        memcpy(cst_sh[i].groups, r->groups, sizeof r->groups);
    }
    if (++cst_reads >= CST_REPORT) {
        qsort(cst_sh, (size_t)cst_nsh, sizeof *cst_sh, cst_cmp);
        double all = 0;
        for (int k = 0; k < cst_nsh; k++) all += cst_sh[k].ns;
        rt_log("vk_exec: OOT_CS_TIME: %d dispatches, %.1f ms of GPU time in %d shaders:", cst_reads, all / 1e6, cst_nsh);
        for (int k = 0; k < cst_nsh && k < CST_LINES; k++) {
            const CstShader *c = &cst_sh[k];
            rt_log("vk_exec:   %02x%02x%02x%02x%02x%02x%02x%02x x%llu: %.3f ms each, %.1f%%, groups %ux%ux%u", c->iid[0],
                   c->iid[1], c->iid[2], c->iid[3], c->iid[4], c->iid[5], c->iid[6], c->iid[7], (unsigned long long)c->n,
                   c->ns / (double)c->n / 1e6, 100.0 * c->ns / all, c->groups[0], c->groups[1], c->groups[2]);
        }
        memset(cst_sh, 0, sizeof cst_sh);
        cst_nsh = 0, cst_reads = 0;
    }
    ReleaseSRWLockExclusive(&cst_lock);
    free(r);
}

// Around one dispatch: begin returns the pair (or -1), end writes the second timestamp and queues the read.
static int cst_begin(VkExec *ex) {
    if (!cst_on()) return -1;
    uint32_t p = cst_next++ % CST_PAIRS;
    if (InterlockedCompareExchange(&cst_busy[p], 1, 0)) return -1;
    vkCmdResetQueryPool(ex->cb, cst_pool, 2 * p, 2);
    vkCmdWriteTimestamp2(ex->cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, cst_pool, 2 * p);
    return (int)p;
}

static void cst_end(VkExec *ex, int pair, const GfxShader *cs, const uint32_t groups[3]) {
    if (pair < 0) return;
    vkCmdWriteTimestamp2(ex->cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, cst_pool, 2 * (uint32_t)pair + 1);
    CstRead *r = malloc(sizeof *r);
    if (!r) rt_fatal("vk_exec: out of memory");
    r->pair = (uint32_t)pair;
    memcpy(r->iid, gfx_shader_input_id(cs), 8);
    memcpy(r->groups, groups, sizeof r->groups);
    vk_completion_add(ex->fence, cst_read, r);
}

// OOT_SKIP_CS=all|<hex>[,<hex>...]|!<hex>[,...] (diagnostic): compute dispatches of every shader, of those whose input
// id starts with one of the prefixes, or (!) of all but those, are not recorded; for finding the kernel behind a GPU timeout (2026-11-06: the
// Kokiri Forest level loses the device on its first frames). Logs each shader it skips once.
static int skip_cs(const GfxShader *cs) {
    const char *e = ENV_ONCE("OOT_SKIP_CS");
    if (!e || !*e) return 0;
    char hex[17];
    const uint8_t *id = gfx_shader_input_id(cs);
    for (int i = 0; i < 8; i++) snprintf(hex + 2 * i, 3, "%02x", id[i]);
    int invert = *e == '!', listed = 0;   // !<list>: every shader except those listed
    for (const char *p = e + invert; *p && !listed;) {
        size_t n = strcspn(p, ",");
        if (n && n <= 16 && !strncmp(hex, p, n)) listed = 1;
        p += n + (p[n] == ',');
    }
    int skip = !strcmp(e, "all") || (invert ? !listed : listed);
    if (skip) {
        static char seen[512][17];
        static int nseen;
        int known = 0;
        for (int i = 0; i < nseen && !known; i++) known = !strcmp(seen[i], hex);
        if (!known && nseen < 512) strcpy(seen[nseen++], hex), rt_log("vk_exec: OOT_SKIP_CS: dispatches of %s skipped", hex);
    }
    return skip;
}

static int push_fresh;   // a push came since the last dispatch (op_push_cc sets it, push_after_dispatch clears it)
static uint32_t push_hidden;   // OOT_PUSH_NOINHERIT: the push length + 1 while one dispatch sees none
static void push_hide(VkExec *ex, const GfxShader *cs, int indirect);
// OOT_PUSH_NOINHERIT (diagnostic): dispatches that see an empty push unless the game pushed right before them:
// "direct", "indirect", "all", or a comma list of shader hash prefixes (the frame dump's "cs" hashes). To find which
// programs go wrong with a push they inherit (2026-11-10, the game crash at main+0xc938d0).
static int push_inherit_blocked(const GfxShader *cs, int indirect) {
    static char spec[512];
    static int init;
    if (!init) { const char *e = getenv("OOT_PUSH_NOINHERIT"); snprintf(spec, sizeof spec, "%s", e ? e : ""); init = 1; }
    if (!spec[0] || push_fresh) return 0;
    if (!strcmp(spec, "all")) return 1;
    if (!strcmp(spec, "direct")) return !indirect;
    if (!strcmp(spec, "indirect")) return indirect;
    uint64_t words = 0, h = 0xcbf29ce484222325ull;   // vk_dump.c shader_hash
    const uint8_t *b = (const uint8_t *)gfx_shader_spirv(cs, &words);
    for (uint64_t i = 0; b && i < words * 4; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    char hex[17];
    snprintf(hex, sizeof hex, "%016llx", (unsigned long long)h);
    for (const char *p = spec; *p;) {
        size_t n = strcspn(p, ",");
        if (n && n <= 16 && !strncmp(hex, p, n)) return 1;
        p += n;
        if (*p) p++;
    }
    return 0;
}
static void push_hide(VkExec *ex, const GfxShader *cs, int indirect) {
    if (!push_hidden && ex->push_cc_len && push_inherit_blocked(cs, indirect))
        push_hidden = ex->push_cc_len + 1, ex->push_cc_len = 0;
}

// Frame dump: the constants the dispatch got pushed (bank 0 from 0x10), up to 0x80 bytes as words, for passes.txt.
static void dump_push(const VkExec *ex) {
    char line[0x80 / 4 * 9 + 1], *o = line;
    uint32_t n = ex->push_cc_len < 0x80 ? ex->push_cc_len : 0x80;
    for (uint32_t i = 0; i + 4 <= n; i += 4) {
        uint32_t w;
        memcpy(&w, ex->push_cc + i, 4);
        o += sprintf(o, " %08x", w);
    }
    *o = 0;
    vk_dump_note("    pushed constants: %u bytes:%s", ex->push_cc_len, line);
}

static void op_dispatch(VkExec *ex, const p_dispatch *d) {
    if (d->x <= 0 || d->y <= 0 || d->z <= 0) return;
    if (!ex->prog) return warn_once(W_PROGRAM, "a dispatch without a bound program is skipped");
    {   // OOT_SKIP_DISPATCH_FROM=<n> (diagnostic): the n-th and later compute dispatches since boot are not recorded, to
        // bisect a device loss down to one of the cinematic's upscaler kernels (2026-11-06)
        static long long skip_from = -2, count;
        if (skip_from == -2) { const char *e = getenv("OOT_SKIP_DISPATCH_FROM"); skip_from = e ? atoll(e) : -1; }
        if (skip_from >= 0 && count++ >= skip_from) return;
    }
    cb_res(ex);
    dummies_init();
    cnt_log();
    int present;
    const GfxShader *cs = shader_for(ex, NVN_STAGE_CS, &present, NULL, 0);   // synchronous: a skipped dispatch changes results
    if (!cs) {
        const uint8_t *ctl, *code;
        uint32_t len;
        if (vk_dump_armed && ex->prog && vk_exec_find_stage(ex->prog, NVN_STAGE_CS, &ctl, &code, &len)) {
            uint64_t h = 0xcbf29ce484222325ull;
            for (uint32_t i = 0; i < len; i++) h = (h ^ code[i]) * 0x100000001b3ull;
            vk_dump_note("dispatch %dx%dx%d skipped: no compute shader (code %016llx)", d->x, d->y, d->z, (unsigned long long)h);
        }
        return;
    }
    if (skip_cs(cs)) return;
    push_hide(ex, cs, 0);
    if (mwatch_on()) {
        char dims[48];
        snprintf(dims, sizeof dims, "%dx%dx%d", d->x, d->y, d->z);
        mwatch_seq(ex, cs, dims);
    }
    const VkPipeInfo *info = NULL;
    VkPipeline p = vk_pipeline_compute(cs, &info);
    if (!p || !info) return warn_once(W_PIPELINE, "a compute pipeline could not be created: dispatches with it are skipped");
    vk_exec_barrier(ex);   // ends the pass; earlier writes are visible to the shader
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    c0_dim[0] = (uint32_t)d->x, c0_dim[1] = (uint32_t)d->y, c0_dim[2] = (uint32_t)d->z;
    gw_select(ex, info);
    twin_dispatch = 1;
    bind_resources(ex, info);
    twin_dispatch = 0;
    gw_n = 0;
    int tp = cst_begin(ex);
    vkCmdDispatch(ex->cb, (uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z);
    cst_end(ex, tp, cs, (const uint32_t[3]){(uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z});
    ex->recorded++;
    if (vk_dump_armed) vk_dump_dispatch(ex, cs, p, info, d), dump_push(ex);
}

// OOT_IND_LOG=1 (diagnostic): the group counts each indirect dispatch reads, copied on the GPU into a host-visible
// ring before the dispatch and logged with the shader once the submission completes (the first 200), and the device's
// limit; for the Kokiri Forest GPU timeout (2026-11-06: kernels eb67c232 and c82d3cba together, one feeding the other).
enum { IND_SLOTS = 256 };
static VkBuffer ind_buf;
static VmaAllocation ind_mem;
static uint32_t *ind_map, ind_next;
typedef struct { uint32_t slot; char hex[17]; uint64_t addr, frame; uint32_t cpu[3]; } IndRead;
static void ind_read(void *arg) {
    IndRead *r = arg;
    static volatile LONG n;
    if (InterlockedIncrement(&n) <= 200) {
        const uint32_t *v = ind_map + 4 * r->slot;
        const uint32_t *lim = vk_ctx()->props.limits.maxComputeWorkGroupCount;
        rt_log("vk_exec: OOT_IND_LOG: f%llu indirect dispatch of %s from %#llx (%s): %u x %u x %u groups (next word %u), CPU %u %u %u%s",
               (unsigned long long)r->frame, r->hex, (unsigned long long)r->addr, nvn_gpu_only(r->addr) ? "GPU-only pool" : "CPU-visible",
               v[0], v[1], v[2], v[3], r->cpu[0], r->cpu[1], r->cpu[2],
               v[0] > lim[0] || v[1] > lim[1] || v[2] > lim[2] ? " (over the device's limit)" : "");
    }
    free(r);
}
static void ind_log(VkExec *ex, const VkMemSpan *ip, const GfxShader *cs, uint64_t addr) {
    static int on = -1;
    static long long from;   // OOT_IND_LOG=<frame> (above 1): from that frame on
    if (on < 0) {
        const char *e = getenv("OOT_IND_LOG");
        on = e != NULL, from = e ? atoll(e) : 0;
        if (on) {
            VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = IND_SLOTS * 16,
                                     .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
            VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                          .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                          .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
            VmaAllocationInfo info;
            if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &ind_buf, &ind_mem, &info) == VK_SUCCESS) ind_map = info.pMappedData;
            else on = 0;
        }
    }
    if (!on || (from > 1 && (long long)gfx_stats().frames < from)) return;
    uint32_t slot = ind_next++ % IND_SLOTS;
    VkMemSpan ip16;   // the word after the group counts too (the culling's counter, 2026-11-07)
    int w = vk_mem_resolve(addr, 16, &ip16);
    VkBufferCopy cp = {ip->offset, (VkDeviceSize)slot * 16, w ? 16 : 12};
    vkCmdCopyBuffer(ex->cb, ip->buffer, ind_buf, 1, &cp);
    ex->recorded++;
    vk_exec_barrier(ex);
    IndRead *r = malloc(sizeof *r);
    if (!r) return;
    r->slot = slot, r->addr = addr, r->frame = gfx_stats().frames;
    memset(r->cpu, 0, sizeof r->cpu);
    if (!nvn_gpu_only(addr)) memcpy(r->cpu, (const void *)(uintptr_t)addr, 12);   // guest memory as the CPU sees it now
    const uint8_t *id = gfx_shader_input_id(cs);
    for (int i = 0; i < 8; i++) snprintf(r->hex + 2 * i, 3, "%02x", id[i]);
    vk_completion_add(ex->fence, ind_read, r);
}

// DispatchComputeIndirect: vkCmdDispatchIndirect from the pool buffer. The shader reads the group counts from c0 0x200,
// so they are copied there on the GPU, into a c0 range no other dispatch uses, before the dispatch.
static void op_dispatch_ind(VkExec *ex, const p_dispatch_ind *d) {
    if (!ex->prog) return warn_once(W_PROGRAM, "a dispatch without a bound program is skipped");
    VkMemSpan ip;
    if (!vk_mem_resolve(d->ind, sizeof(VkDispatchIndirectCommand), &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect dispatch arguments at %#llx are not backed or not aligned: skipped", (unsigned long long)d->ind);
    cb_res(ex);
    dummies_init();
    int present;
    const GfxShader *cs = shader_for(ex, NVN_STAGE_CS, &present, NULL, 0);
    if (!cs) return;
    if (skip_cs(cs)) return;
    push_hide(ex, cs, 1);
    if (mwatch_on()) {
        char dims[48];
        snprintf(dims, sizeof dims, "indirect@%#llx", (unsigned long long)d->ind);
        mwatch_seq(ex, cs, dims);
    }
    const VkPipeInfo *info = NULL;
    VkPipeline p = vk_pipeline_compute(cs, &info);
    if (!p || !info) return warn_once(W_PIPELINE, "a compute pipeline could not be created: dispatches with it are skipped");
    vk_exec_barrier(ex);
    VK_MEM_USE(ip);
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    c0_dim[0] = c0_dim[1] = c0_dim[2] = 0;
    B.c0_valid[2] = 0;   // a fresh c0 range: the copy below must not change one an earlier dispatch reads
    gwr_note(d->ind, 16);
    gw_select(ex, info);
    twin_dispatch = 1;
    bind_resources(ex, info);
    twin_dispatch = 0;
    gw_n = 0;
    if (B.c0_valid[2]) {   // the shader binds c0
        VkBufferCopy cp = {ip.offset, B.c0_off[2] + 0x200, sizeof(VkDispatchIndirectCommand)};
        vkCmdCopyBuffer(ex->cb, ip.buffer, B.c0_buf[2], 1, &cp);
        ex->recorded++;
        vk_exec_barrier(ex);
        B.c0_valid[2] = 0;   // the range no longer holds B.c0[2]
    }
    ind_log(ex, &ip, cs, d->ind);
    mwatch_add(d->ind);   // OOT_MESH_WATCH: who writes the indirect dispatch's group counts too
    vkCmdDispatchIndirect(ex->cb, ip.buffer, ip.offset);
    ex->recorded++;
    vk_mem_count(ip.cls, VK_MUSE_IND, sizeof(VkDispatchIndirectCommand));
    VK_MEM_BOUND(ip);
    if (vk_dump_armed) vk_dump_dispatch(ex, cs, p, info, &(p_dispatch){0, 0, 0, d->caller}), dump_push(ex);
}

// ---- OP_DRAW_MESH / OP_DRAW_MESH_IND (nvn_api_cmd.c DrawMeshTasks*): step 2 of docs/phase3.md "the mesh stages":
// the task program (if the bound program has one) and the mesh program run as compute programs on the records (M);
// nothing is drawn from them yet. A task draw of n tasks dispatches n task workgroups, then n << MESH_SHIFT mesh
// workgroups (each beyond its task's count exits at once: sm86_lower.c mesh_prologue); a mesh-only draw of n
// dispatches n mesh workgroups. Indirect draws are not run yet: their argument layout is not known, and a wrong
// guess could launch a huge grid; OOT_MESH_LOG=1 reads their first 16 bytes back instead, and the record heads of
// the direct draws, the first 40 of each.
// 5: the game's task programs launch up to exactly 32 mesh workgroups each (OOT_MESH_TASKS, forest, 2026-11-08); at 4
// the meshes past the 16th of every large object were cut off. The records and the pass-through draw grow with it.
#define MESH_SHIFT 5u
#define MESH_MAX_TASKS 4096u
static const NVNshaderData *stage_entry(const prog_t *p, int st);

static struct { VkBuffer buf; VmaAllocation mem; uint32_t *map; uint32_t next; } MRB;   // OOT_MESH_LOG read-back
typedef struct { uint32_t slot, kind, a, b; uint64_t ind, frame; uint32_t cpu[4]; } MeshRead;
static void mesh_read(void *arg) {
    MeshRead *r = arg;
    const uint32_t *v = MRB.map + 16 * r->slot;
    if (r->kind)
        rt_log("meshlog: f%llu indirect mesh draw (%u, task %u) at %#llx: GPU %#x %#x %#x %#x, CPU at record %#x %#x %#x %#x",
               (unsigned long long)r->frame, r->a & 0xff, r->a >> 8 & 1, (unsigned long long)r->ind, v[0], v[1], v[2], v[3],
               r->cpu[0], r->cpu[1], r->cpu[2], r->cpu[3]);
    else
        rt_log("meshlog: mesh draw of %u tasks, %u mesh groups: task 0 words %#x %#x %#x %#x; primitive counts of groups "
               "0-7: %u %u %u %u %u %u %u %u; group 0 vertex 0: %g %g %g %g", r->a, r->b, v[0], v[1], v[2], v[3], v[4],
               v[5], v[6], v[7], v[8], v[9], v[10], v[11], *(const float *)&v[12], *(const float *)&v[13],
               *(const float *)&v[14], *(const float *)&v[15]);
    free(r);
}
// A read-back slot of 16 words for the next OOT_MESH_LOG line, or -1.
static int mesh_rb_slot(void) {
    static int on = -1;
    static volatile LONG n;
    if (on < 0) {
        on = getenv("OOT_MESH_LOG") != NULL;
        if (on) {
            VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 64 * 64,
                                     .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
            VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                          .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                          .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
            VmaAllocationInfo info;
            if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &MRB.buf, &MRB.mem, &info) == VK_SUCCESS) MRB.map = info.pMappedData;
            else on = 0;
        }
    }
    static long long from = -1;   // OOT_MESH_LOG_FROM=<frame>: read back from that frame on (the level, not the title)
    if (from < 0) { const char *e = getenv("OOT_MESH_LOG_FROM"); from = e ? atoll(e) : 0; }
    if (!on || (long long)gfx_stats().frames < from || InterlockedIncrement(&n) > 80) return -1;
    return (int)(MRB.next++ % 64);
}
// OOT_MESH_SYNC (--probe-cs): the entries wait for vk_exec_mesh_flush, called after the harness's fence wait (its
// completion callbacks run before the GPU has run the submission).
static MeshRead *mesh_held[64];
static int mesh_nheld;
void vk_exec_mesh_flush(void) {
    for (int i = 0; i < mesh_nheld; i++) mesh_read(mesh_held[i]);
    mesh_nheld = 0;
}
static void mesh_rb_done(VkExec *ex, int slot, uint32_t kind, uint32_t a, uint32_t b, uint64_t ind) {
    MeshRead *r = malloc(sizeof *r);
    if (!r) return;
    *r = (MeshRead){(uint32_t)slot, kind, a, b, ind, gfx_stats().frames, {0}};
    VkMemSpan sp;
    if (ind && vk_mem_resolve(ind, 16, &sp)) memcpy(r->cpu, (const void *)(uintptr_t)ind, 16);
    (void)ex;
    if (ENV_ONCE("OOT_MESH_SYNC")) {
        if (mesh_nheld < 64) mesh_held[mesh_nheld++] = r;
        else free(r);
        return;
    }
    vk_completion_add(vk_sync_issued() + 1, mesh_read, r);   // the submission being recorded (ex->fence is set at its submit)
}

// OOT_MESH_TASKS=<frame> (diagnostic): for every mesh draw of that frame, its indirect arguments and the mesh workgroup
// count each of its tasks asked for, read back on the GPU. A slot holds 64 task counts and the 4 argument words.
#define MTB_SLOT_WORDS 136u
#define MTB_SLOTS 256u
static struct { VkBuffer buf; VmaAllocation mem; uint32_t *map; uint32_t next; } MTB;
typedef struct { uint32_t slot, tasks, shift; uint64_t ht, hm, ind, s[9]; uint32_t sz[9]; } MeshTasksRead;
static void mesh_tasks_read(void *arg) {
    MeshTasksRead *r = arg;
    const uint32_t *v = MTB.map + MTB_SLOT_WORDS * r->slot;
    uint32_t n = r->tasks < 64 ? r->tasks : 64, sum = 0, max = 0, over = 0;
    char s[640];
    size_t o = 0;
    for (uint32_t i = 0; i < n; i++) {
        sum += v[i], max = v[i] > max ? v[i] : max, over += r->shift < 31 && v[i] > 1u << r->shift;
        if (v[i] && o < sizeof s - 14) o += (size_t)snprintf(s + o, sizeof s - o, " %u", v[i]);
    }
    s[o] = 0;
    rt_log("meshtasks: task %016llx mesh %016llx args at %#llx: %u %u %u %u; %u tasks ran: sum %u max %u, %u over the limit of %u:%s",
           (unsigned long long)r->ht, (unsigned long long)r->hm, (unsigned long long)r->ind, v[64], v[65], v[66], v[67], n, sum,
           max, over, r->shift < 31 ? 1u << r->shift : 0u, s);
    char b[400];
    size_t bo = 0;
    for (int i = 0; i < 9; i++) if (r->s[i]) bo += (size_t)snprintf(b + bo, sizeof b - bo, " %d=%#llx+%#x", i, (unsigned long long)r->s[i], r->sz[i]);
    b[bo] = 0;
    uint32_t psum = 0, pn = 0, pmax = 0, pfirst = 0xffffffffu;
    for (uint32_t i = 0; i < 64; i++) if (v[72 + i]) psum += v[72 + i], pn++, pmax = v[72 + i] > pmax ? v[72 + i] : pmax, pfirst = pfirst == 0xffffffffu ? i : pfirst;
    rt_log("meshtasks: mesh storage:%s; slot 2 head on the GPU %#x %#x %#x %#x; primitive counts of 64 groups: sum %u in %u groups, max %u, first %d", b, v[68], v[69], v[70], v[71], psum, pn, pmax, (int)pfirst);
    free(r);
}
static void mesh_tasks_note(VkExec *ex, const VkMemSpan *ip, uint64_t ind, uint32_t tasks, uint32_t shift) {
    static long long at = -2;
    if (at == -2) { const char *e = getenv("OOT_MESH_TASKS"); at = e ? atoll(e) : -1; }
    if (at < 0 || (long long)gfx_stats().frames != at || MTB.next >= MTB_SLOTS) return;
    if (!MTB.buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = MTB_SLOTS * MTB_SLOT_WORDS * 4,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                      .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &MTB.buf, &MTB.mem, &info) != VK_SUCCESS) { at = -1; return; }
        MTB.map = info.pMappedData;
    }
    uint32_t slot = MTB.next++, n = tasks < 64 ? tasks : 64;
    VkDeviceSize base = (VkDeviceSize)slot * MTB_SLOT_WORDS * 4;
    for (int b = 0; b < 8; b++) {   // the mesh stage's uniform banks as the draw left them, with the words the game's grass programs scale by
        const p_bind_buf *u = &ex->mubo[1][b];
        VkMemSpan us;
        float f[3] = {0, 0, 0}, w[4] = {0, 0, 0, 0};
        if (u->addr && u->size >= 0x1e0 && vk_mem_resolve(u->addr, 0x1e0, &us)) {
            memcpy(&f[0], (const uint8_t *)(uintptr_t)u->addr + 0x98, 4);
            memcpy(&f[1], (const uint8_t *)(uintptr_t)u->addr + 0x178, 4);
            memcpy(&f[2], (const uint8_t *)(uintptr_t)u->addr + 0x1d0, 4);
            memcpy(w, (const uint8_t *)(uintptr_t)u->addr + 0x60, 16);
        }
        if (u->addr) rt_log("meshtasks: slot %u mesh ubo %d: %#llx+%#llx [0x98] %g [0x178] %g [0x1d0] %g row[0x60] %g %g %g %g", slot, b, (unsigned long long)u->addr, (unsigned long long)u->size, f[0], f[1], f[2], w[0], w[1], w[2], w[3]);
    }
    VkBufferCopy cp[64];
    for (uint32_t i = 0; i < n; i++)
        cp[i] = (VkBufferCopy){(VkDeviceSize)i * SM86_TASK_STRIDE + 4, base + 4 * i, 4};
    vk_exec_barrier(ex);
    vkCmdCopyBuffer(ex->cb, M.buf, MTB.buf, n, cp);
    ex->recorded++;
    if (ip && ip->buffer) {
        VkBufferCopy a = {ip->offset, base + 64 * 4, 16};
        vkCmdCopyBuffer(ex->cb, ip->buffer, MTB.buf, 1, &a);
        ex->recorded++;
    }
    VkBufferCopy pc[64];
    uint32_t npc = 0;
    for (uint32_t g = 0; g < 64 && (VkDeviceSize)(g + 1) * SM86_MESH_STRIDE <= M.mesh_size; g++, npc++)
        pc[g] = (VkBufferCopy){M.task_size + (VkDeviceSize)g * SM86_MESH_STRIDE + SM86_MESH_MAP, base + (72 + g) * 4, 4};
    if (npc) vkCmdCopyBuffer(ex->cb, M.buf, MTB.buf, npc, pc), ex->recorded++;
    MeshTasksRead *r = malloc(sizeof *r);
    if (!r) return;
    *r = (MeshTasksRead){slot, tasks, shift, 0, 0, ind, {0}, {0}};
    for (int i = 0; i < 9; i++) r->s[i] = ex->mssbo[1][i].addr, r->sz[i] = (uint32_t)ex->mssbo[1][i].size;
    VkMemSpan s2;
    if (ex->mssbo[1][2].addr && vk_mem_resolve(ex->mssbo[1][2].addr, 16, &s2)) {
        VkBufferCopy c2 = {s2.offset, base + 68 * 4, 16};
        vkCmdCopyBuffer(ex->cb, s2.buffer, MTB.buf, 1, &c2);
        ex->recorded++;
    }
    for (int m = 0; m < 2; m++) {
        const NVNshaderData *e = stage_entry(ex->mesh_prog[m], 6 + m);
        if (!e) continue;
        uint32_t len;
        memcpy(&len, (const uint8_t *)e->control + 0x44, 4);
        uint64_t h = 0xcbf29ce484222325ull;
        for (uint32_t i = 0; i < len; i++) h = (h ^ ((const uint8_t *)(uintptr_t)e->data)[i]) * 0x100000001b3ull;
        if (m) r->hm = h; else r->ht = h;
    }
    rt_log("meshtasks: slot %u is task %016llx mesh %016llx", slot, (unsigned long long)r->ht, (unsigned long long)r->hm);
    vk_completion_add(vk_sync_issued() + 1, mesh_tasks_read, r);
}

// OOT_MESH_SERIES=<first frame>:<frames> (diagnostic): the indirect mesh draws' arguments, read back on the GPU, per argument address and frame,
// printed once as one row per address: groups of each frame ('.' when that address was not drawn), then the sum of the items.
enum { MS_ROWS = 96, MS_FRAMES = 240, MS_SLOTS = 8192 };
static struct { VkBuffer buf; VmaAllocation mem; uint32_t *map; uint32_t next; long long from, n; int on, printed;
                struct { uint64_t ind; int32_t g[MS_FRAMES]; uint64_t items; } row[MS_ROWS]; int nrow; } MS;
typedef struct { uint32_t slot; uint64_t ind; long long frame; } MsRead;
static void ms_read(void *arg) {
    MsRead *r = arg;
    const uint32_t *v = MS.map + 4 * r->slot;
    int k = 0;
    while (k < MS.nrow && MS.row[k].ind != r->ind) k++;
    if (k == MS.nrow && MS.nrow < MS_ROWS) { MS.row[k].ind = r->ind; for (int i = 0; i < MS_FRAMES; i++) MS.row[k].g[i] = -1; MS.nrow++; }
    long long f = r->frame - MS.from;
    if (k < MS.nrow && f >= 0 && f < MS.n) { if (MS.row[k].g[f] < 0) MS.row[k].g[f] = 0; MS.row[k].g[f] += (int32_t)v[0]; MS.row[k].items += v[3]; }
    free(r);
}
static void ms_print(void) {
    MS.printed = 1;
    rt_log("meshseries: %d argument addresses, frames %lld..%lld", MS.nrow, MS.from, MS.from + MS.n - 1);
    for (int k = 0; k < MS.nrow; k++) {
        char b[1600];
        size_t o = 0;
        for (long long i = 0; i < MS.n && o < sizeof b - 8; i++)
            o += MS.row[k].g[i] < 0 ? (size_t)snprintf(b + o, sizeof b - o, " .") : (size_t)snprintf(b + o, sizeof b - o, " %d", MS.row[k].g[i]);
        b[o] = 0;
        rt_log("meshseries: %#llx items %llu:%s", (unsigned long long)MS.row[k].ind, (unsigned long long)MS.row[k].items, b);
    }
}
static void ms_note(VkExec *ex, const VkMemSpan *ip, uint64_t ind) {
    static int init;
    if (!init) {
        init = 1;
        const char *e = getenv("OOT_MESH_SERIES");
        long long a = 0, n = 0;
        if (e && sscanf(e, "%lld:%lld", &a, &n) == 2 && n > 0) MS.from = a, MS.n = n > MS_FRAMES ? MS_FRAMES : n, MS.on = 1;
        if (MS.on) {
            VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = MS_SLOTS * 16, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
            VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST, .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                          .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
            VmaAllocationInfo info;
            if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &MS.buf, &MS.mem, &info) == VK_SUCCESS) MS.map = info.pMappedData; else MS.on = 0;
        }
    }
    if (!MS.on) return;
    long long f = (long long)gfx_stats().frames;
    if (f >= MS.from + MS.n + 8 && !MS.printed) ms_print();
    if (f < MS.from || f >= MS.from + MS.n || !ip || !ip->buffer || MS.next >= MS_SLOTS) return;
    uint32_t slot = MS.next++;
    vk_exec_barrier(ex);
    VkBufferCopy a = {ip->offset, (VkDeviceSize)slot * 16, 16};
    vkCmdCopyBuffer(ex->cb, ip->buffer, MS.buf, 1, &a);
    ex->recorded++;
    MsRead *r = malloc(sizeof *r);
    if (!r) return;
    *r = (MsRead){slot, ind, f};
    vk_completion_add(vk_sync_issued() + 1, ms_read, r);
}

// OOT_MESH_REC=<frame>:<mesh hash prefix> (diagnostic): the mesh records of up to 8 such draws of that frame are written
// to build/logs/meshrec/<hash>_<n>.bin (a u32 header: groups, stride, map offset, mesh count of bytes; then the records).
#define MREC_BYTES (4u << 20)
static struct { VkBuffer buf; VmaAllocation mem; uint8_t *map; uint32_t next; } MREC;
typedef struct { uint32_t slot, groups, bytes; uint64_t hm; } MeshRecRead;
static void mesh_rec_read(void *arg) {
    MeshRecRead *r = arg;
    char name[128];
    CreateDirectoryA("build/logs/meshrec", NULL);
    snprintf(name, sizeof name, "build/logs/meshrec/%016llx_%u.bin", (unsigned long long)r->hm, r->slot);
    FILE *f = fopen(name, "wb");
    if (f) {
        uint32_t h[4] = {r->groups, SM86_MESH_STRIDE, SM86_MESH_MAP, r->bytes};
        fwrite(h, 4, 4, f), fwrite(MREC.map + (size_t)r->slot * MREC_BYTES, 1, r->bytes, f), fclose(f);
    }
    free(r);
}
static void mesh_rec_note(VkExec *ex, uint32_t groups) {
    static long long at = -2;
    static char prefix[128];   // one or more hash prefixes, comma-separated
    if (at == -2) {
        const char *e = getenv("OOT_MESH_REC");
        at = e ? atoll(e) : -1;
        if (e && strchr(e, ':')) snprintf(prefix, sizeof prefix, "%s", strchr(e, ':') + 1);
    }
    if (at < 0 || (long long)gfx_stats().frames != at || MREC.next >= 8) return;
    const NVNshaderData *e = stage_entry(ex->mesh_prog[1], 7);
    if (!e) return;
    uint32_t len;
    memcpy(&len, (const uint8_t *)e->control + 0x44, 4);
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint32_t i = 0; i < len; i++) h = (h ^ ((const uint8_t *)(uintptr_t)e->data)[i]) * 0x100000001b3ull;
    char hx[17];
    snprintf(hx, 17, "%016llx", (unsigned long long)h);
    int rec_hit = !*prefix;
    for (const char *p = prefix; *p && !rec_hit;) {
        size_t k = strcspn(p, ",");
        rec_hit = k && k <= 16 && !strncmp(hx, p, k);
        p += k + (p[k] == ',');
    }
    if (!rec_hit) return;
    if (!MREC.buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 8ull * MREC_BYTES, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                      .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &MREC.buf, &MREC.mem, &info) != VK_SUCCESS) { at = -1; return; }
        MREC.map = info.pMappedData;
    }
    uint32_t slot = MREC.next++;
    uint32_t bytes = M.mesh_size < MREC_BYTES ? (uint32_t)M.mesh_size : MREC_BYTES;
    vk_exec_barrier(ex);
    VkBufferCopy cp = {M.task_size, (VkDeviceSize)slot * MREC_BYTES, bytes};
    vkCmdCopyBuffer(ex->cb, M.buf, MREC.buf, 1, &cp);
    ex->recorded++;
    MeshRecRead *r = malloc(sizeof *r);
    if (!r) return;
    *r = (MeshRecRead){slot, groups, bytes, h};
    vk_completion_add(vk_sync_issued() + 1, mesh_rec_read, r);
}

static struct { VkPipeline pipe; VkPipelineLayout layout; VkDescriptorSetLayout dsl; PFN_vkCmdPushDescriptorSetKHR push; int state; } MC;   // mesh_compact
static void mesh_release(VkCtx *c) {   // vk_exec_draw_shutdown
    if (M.buf) vmaDestroyBuffer(c->vma, M.buf, M.mem);
    if (MRB.buf) vmaDestroyBuffer(c->vma, MRB.buf, MRB.mem);
    if (MTB.buf) vmaDestroyBuffer(c->vma, MTB.buf, MTB.mem);
    if (MREC.buf) vmaDestroyBuffer(c->vma, MREC.buf, MREC.mem);
    memset(&MREC, 0, sizeof MREC);
    if (MC.pipe) vkDestroyPipeline(c->device, MC.pipe, NULL);
    if (MC.layout) vkDestroyPipelineLayout(c->device, MC.layout, NULL);
    if (MC.dsl) vkDestroyDescriptorSetLayout(c->device, MC.dsl, NULL);
    memset(&MC, 0, sizeof MC);
    memset(&M, 0, sizeof M), memset(&MRB, 0, sizeof MRB), memset(&MTB, 0, sizeof MTB);
}

// The record buffer, grown when a draw needs more; the old one goes after the submission being recorded.
static int mesh_buffer(VkDeviceSize task_bytes, VkDeviceSize mesh_bytes) {
    VkDeviceSize need = task_bytes + mesh_bytes;
    if (need > M.cap) {
        VkDeviceSize want = M.cap ? M.cap * 2 : 8u << 20;
        while (want < need) want *= 2;
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = want,
                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VkBuffer nb;
        VmaAllocation nm;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &nb, &nm, NULL) != VK_SUCCESS) return 0;
        if (M.buf) {
            TwinBuf *old = malloc(sizeof *old);
            if (!old) rt_fatal("vk: out of memory (mesh records)");
            *old = (TwinBuf){M.buf, M.mem};
            vk_completion_add(vk_sync_issued() + 1, twin_buf_free, old);
        }
        M.buf = nb, M.mem = nm, M.cap = want;
    }
    M.task_size = task_bytes, M.mesh_size = mesh_bytes;
    return 1;
}

// One stage (m 0 task, 1 mesh) as a compute dispatch of `groups` workgroups: its resources and program in the
// compute slots for the length of the dispatch.
static int mesh_dispatch(VkExec *ex, int m, uint32_t groups) {
    const int cs_st = NVN_STAGE_CS;
    static p_bind_buf su[18], ss[16];
    static uint64_t stx[128], ssp[128], sim[32];
    memcpy(su, ex->ubo[cs_st], sizeof su), memcpy(ss, ex->ssbo[cs_st], sizeof ss);
    memcpy(stx, ex->tex[cs_st], sizeof stx), memcpy(ssp, ex->samp[cs_st], sizeof ssp), memcpy(sim, ex->img[cs_st], sizeof sim);
    const prog_t *sprog = ex->prog;
    uint32_t spush = ex->push_cc_len;
    memcpy(ex->ubo[cs_st], ex->mubo[m], sizeof su), memcpy(ex->ssbo[cs_st], ex->mssbo[m], sizeof ss);
    memcpy(ex->tex[cs_st], ex->mtex[m], sizeof stx), memcpy(ex->samp[cs_st], ex->msamp[m], sizeof ssp);
    memcpy(ex->img[cs_st], ex->mimg[m], sizeof sim);
    ex->prog = ex->mesh_prog[m], ex->push_cc_len = 0, ex->tex_serial++, bind_gen++;
    M.table = m ? SM86_MESH_TABLE : SM86_TASK_TABLE;
    int ok = 0;
    const NVNshaderData *e = stage_entry(ex->prog, 6 + m);
    const GfxShader *cs = NULL;
    if (e) {
        const uint8_t *ctl = e->control, *code = (const uint8_t *)(uintptr_t)e->data;
        uint32_t len;
        memcpy(&len, ctl + 0x44, 4);
        GfxShaderAnswers a;
        answers_for(ex, cs_st, ctl, 0, &a);
        cs = gfx_shader_get(GFX_STAGE_CS, ctl, code, len, NULL, &a);
        if (!cs) warn_once(W_MESH, "a %s program failed to translate: mesh draws with it are skipped", m ? "mesh" : "task");
    }
    if (cs && !skip_cs(cs)) {
        C1CUR[cs_st] = cs;
        const VkPipeInfo *info = NULL;
        VkPipeline p = vk_pipeline_compute(cs, &info);
        if (p && info) {
            vk_exec_barrier_compute(ex);   // op_draw_mesh's full barrier precedes the first stage
            bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, p);
            c0_dim[0] = groups, c0_dim[1] = c0_dim[2] = 1;
            B.c0_valid[2] = 0;
            gw_select_mesh(ex, info);
            bind_resources(ex, info);
            gw_n = 0;
            if (m == 0 && M.ind.buffer && !B.c0_valid[2]) {   // no bank 0 to take the count: the bound must not run
                warn_once(W_MESH, "an indirect mesh draw's task program binds no bank 0: skipped");
                cs = NULL;
            } else if (m == 0 && M.ind.buffer) {   // an indirect draw: its count over the bound's in bank 0
                VkBufferCopy cp = {M.ind.offset, B.c0_off[2] + SM86_MESH_C0_COUNT, 4};
                vkCmdCopyBuffer(ex->cb, M.ind.buffer, B.c0_buf[2], 1, &cp);
                ex->recorded++;
                vk_exec_barrier_compute(ex);
            }
        }
        if (cs && p && info) {
            int tp = cst_begin(ex);   // OOT_CS_TIME: the task and mesh stages' GPU time, with the kernels'
            if (m == 1 && M.mi) vkCmdDispatchIndirect(ex->cb, MI.buf, M.mi_off);
            else vkCmdDispatch(ex->cb, groups, 1, 1);
            cst_end(ex, tp, cs, (const uint32_t[3]){groups, 1, 1});
            ex->recorded++;
            ok = 1;
        }
    }
    B.c0_valid[2] = 0;   // the next dispatch gets a bank without the record entries
    memcpy(ex->ubo[cs_st], su, sizeof su), memcpy(ex->ssbo[cs_st], ss, sizeof ss);
    memcpy(ex->tex[cs_st], stx, sizeof stx), memcpy(ex->samp[cs_st], ssp, sizeof ssp), memcpy(ex->img[cs_st], sim, sizeof sim);
    ex->prog = sprog, ex->push_cc_len = spush, ex->tex_serial++, bind_gen++;
    return ok;
}

// Step 3: the records drawn. A stand-in program per mesh program: the pass-through vertex program (sm86_lower.c
// mesh_vs, from a copy of the mesh stage's control section marked SM86_MESH_VS_MAGIC) and the game's fragment stage of
// the same program; one instanced draw of 3 * max-primitives vertices per mesh workgroup, with the replay's current
// state (targets, blend, depth) as the game set it for the mesh draw.
static struct { const void *ctl_src; uint64_t data, fs; uint8_t ctl[0x400]; prog_t prog; } MVS[64];
static uint32_t mvs_next;
static const prog_t *mesh_vs_prog(const prog_t *mesh) {
    const NVNshaderData *me = stage_entry(mesh, 7), *fe = stage_entry(mesh, 1);
    if (!me || !fe) return NULL;
    for (int i = 0; i < 64; i++)
        if (MVS[i].ctl_src == me->control && MVS[i].data == me->data && MVS[i].fs == fe->data) return &MVS[i].prog;
    typeof(MVS[0]) *e = &MVS[mvs_next++ % 64];
    memcpy(e->ctl, me->control, sizeof e->ctl);   // gfx_shader reads CONTROL_BOUND (0x400) bytes behind a control
    uint32_t magic = SM86_MESH_VS_MAGIC;
    e->ctl[0x56] = 0;
    memcpy(e->ctl + 0x48, &magic, 4);
    e->ctl_src = me->control, e->data = me->data, e->fs = fe->data;
    memset(&e->prog, 0, sizeof e->prog);
    e->prog.magic = NVN_PROG_MAGIC, e->prog.nstage = 2;
    e->prog.sd[0] = (NVNshaderData){me->data, e->ctl};
    e->prog.sd[1] = *fe;
    e->prog.serial = nvn_new_serial();
    return &e->prog;
}

// An indirect task draw's launched mesh workgroups as one packed list (shaders/mesh_compact.comp). Flat indexing needs 32 workgroups' records and
// a pass-through instance for every task the list could hold, and only the GPU knows how many tasks there are: a bound of 64 cut the grass lists
// (606 and 400 tasks, 2026-11-09), a bound that fits them cost 8 fps. OOT_MESH_MAP=0: the bound (OOT_MESH_IND_MAX).
#include "mesh_compact_spv.h"
static int mc_ready(void) {
    if (MC.state) return MC.state > 0;
    MC.state = -1;
    const char *e = getenv("OOT_MESH_MAP");
    if (e && e[0] == '0') return 0;
    VkCtx *c = vk_ctx();
    MC.push = vk_pipeline_push_fn();
    VkDescriptorSetLayoutBinding b[2];
    for (uint32_t i = 0; i < 2; i++)
        b[i] = (VkDescriptorSetLayoutBinding){.binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
                                              .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = 2, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 20};
    if (!MC.push || vkCreateDescriptorSetLayout(c->device, &dci, NULL, &MC.dsl) != VK_SUCCESS) return 0;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &MC.dsl,
                                     .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(c->device, &li, NULL, &MC.layout) != VK_SUCCESS) return 0;
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof mesh_compact_spv, .pCode = mesh_compact_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS) return 0;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                .module = m, .pName = "main"},
                                      .layout = MC.layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &MC.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) return 0;
    MC.state = 1;
    rt_log("vk_exec: indirect task draws launch a packed list of mesh workgroups (OOT_MESH_MAP=0: the bound)");
    return 1;
}
static void mesh_compact(VkExec *ex) {
    vk_exec_barrier_compute(ex);   // the task stage's records
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, MC.pipe);
    VkDescriptorBufferInfo bi[2] = {{M.buf, 0, M.task_size + M.mesh_size}, {MI.buf, 0, MI_SLOTS * 32}};
    VkWriteDescriptorSet w[2];
    for (uint32_t i = 0; i < 2; i++)
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = i, .descriptorCount = 1,
                                      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    rec_push(MC.push, ex->cb, VK_PIPELINE_BIND_POINT_COMPUTE, MC.layout, 0, 2, w);
    uint32_t pc[5] = {(uint32_t)(M.task_size / 4), (uint32_t)(M.mi_off / 4), MI_MAX_GROUPS, MESH_MAX_TASKS, SM86_MESH_INDEX / 4};
    vkCmdPushConstants(ex->cb, MC.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(ex->cb, 1, 1, 1);
    ex->recorded += 3;
    vk_exec_barrier_compute(ex);   // the list and the counts, for the mesh stage and as indirect arguments
}

static void op_draw(VkExec *ex, const p_draw *d);
static void mesh_draw(VkExec *ex, const prog_t *mesh, uint32_t groups) {
    const prog_t *vp = mesh_vs_prog(mesh);
    if (!vp) return warn_once(W_MESH, "a mesh program without a fragment stage is not drawn");
    uint32_t prims;
    memcpy(&prims, (const uint8_t *)stage_entry(mesh, 7)->control + 0x60, 4);
    prims >>= 16;
    if (!prims || prims > 256) return;
    static uint64_t n;
    const prog_t *sprog = ex->prog;
    p_bind_buf ss = ex->ssbo[NVN_STAGE_VS][SM86_MESH_VS_SLOT];
    // a stand-in address that differs per draw, so the bind signature never reuses another draw's records
    ex->ssbo[NVN_STAGE_VS][SM86_MESH_VS_SLOT] = (p_bind_buf){NVN_STAGE_VS, SM86_MESH_VS_SLOT, MESH_FAKE_MESH + (++n << 24), M.mesh_size};
    ex->prog = vp, bind_gen++;
    M.drawing = 1;
    // OOT_MESH_NODEPTH=1 (diagnostic): the pass-through draws without a depth test, to tell depth from missing geometry.
    static int nodepth = -1;
    if (nodepth < 0) nodepth = getenv("OOT_MESH_NODEPTH") != NULL;
    extern int vk_force_no_depth;
    if (nodepth) vk_force_no_depth = 1, B.st_valid = 0;
    p_draw d = {4, 0, (int32_t)(3 * prims), 0, (int32_t)groups};   // NVN triangles
    op_draw(ex, &d);
    if (nodepth) vk_force_no_depth = 0, B.st_valid = 0;
    M.drawing = 0;
    ex->prog = sprog, bind_gen++;
    ex->ssbo[NVN_STAGE_VS][SM86_MESH_VS_SLOT] = ss;
}

static void op_draw_mesh(VkExec *ex, const p_draw_mesh *d, int indirect) {    // OOT_MESH (default on since 2026-11-08; OOT_MESH=0 off): run the stages. The first run with them lost the device on the first direct mesh draws
    // of the title (2026-11-07, run-68, cause not known yet), so they stay off until that is understood.
    static int on = -1;
    if (on < 0) on = rt_knob("MESH", "1");
    if (!on) return;
    const prog_t *task = ex->mesh_prog[0], *mesh = ex->mesh_prog[1];
    if (!mesh || !stage_entry(mesh, 7)) return warn_once(W_MESH, "a mesh draw without a mesh program is skipped");
    int has_task = task && stage_entry(task, 6) != NULL;
    cb_res(ex);
    dummies_init();
    // An indirect draw (2026-11-08, once the culling counted: docs/phase3.md): 16-byte arguments, word 0 the count
    // (task workgroups, or mesh workgroups without a task program), word 3 the culling's counter, words 1-2 zero. The
    // count is only on the GPU, so a bound is dispatched (OOT_MESH_IND_MAX, default 64) and the count copied where
    // the programs read it: a task program's workgroups past it exit (bank 0 SM86_MESH_C0_COUNT, task_prologue); a
    // mesh-only draw's stand-in task record gets it as its mesh count, so the mesh workgroups past it draw nothing.
    VkMemSpan ip = {0};
    if (indirect) {
        mwatch_add(d->ind);
        gwr_note(d->ind, 16);
        if (!vk_mem_resolve(d->ind, 16, &ip) || ip.offset % 4)
            return warn_once(W_MESH, "indirect mesh draw arguments at %#llx are not backed or not aligned: skipped", (unsigned long long)d->ind);
        vk_exec_barrier(ex);
        VK_MEM_USE(ip);
        int slot = mesh_rb_slot();   // OOT_MESH_LOG: the arguments as the draw reads them
        if (slot >= 0) {
            VkBufferCopy cp = {ip.offset, (VkDeviceSize)slot * 64, 16};
            vkCmdCopyBuffer(ex->cb, ip.buffer, MRB.buf, 1, &cp);
            ex->recorded++;
            mesh_rb_done(ex, slot, 1, ((uint32_t)d->count & 0xff) | (has_task ? 0x100u : 0), 0, d->ind);
        }
        // OOT_MESH_IND (default on since 2026-11-08; OOT_MESH_IND=0 off): run them. The first run with it lost the device on the title (2026-11-08,
        // run-126, together with a full re-translation; cause not isolated), so it stays off until a run shows it safe.
        static int ind_on = -1;
        if (ind_on < 0) ind_on = rt_knob("MESH_IND", "1");
        if (!ind_on) return;
        // OOT_MESH_IND_FROM=<frame>: only from that frame on (run-126 and run-130 lost the device within 2 s of the
        // first indirect draws on the title). OOT_MESH_IND_SKIP=<hex>[,<hex>...]: not the draws whose task or mesh
        // program's code hash (FNV-1a 64, as kernel.py prints "hash") starts with one of these. Every new pair of
        // programs is logged before it is recorded, so a device loss names its candidates.
        static long long ind_from = -1;
        if (ind_from < 0) { const char *e = getenv("OOT_MESH_IND_FROM"); ind_from = e ? atoll(e) : 0; }
        if ((long long)gfx_stats().frames < ind_from) return;
        uint64_t hp[2] = {0, 0};
        for (int m = 0; m < 2; m++) {
            const NVNshaderData *e = stage_entry(ex->mesh_prog[m], 6 + m);
            if (!e) continue;
            uint32_t len;
            memcpy(&len, (const uint8_t *)e->control + 0x44, 4);
            uint64_t h = 0xcbf29ce484222325ull;
            for (uint32_t i = 0; i < len; i++) h = (h ^ ((const uint8_t *)(uintptr_t)e->data)[i]) * 0x100000001b3ull;
            hp[m] = h;
        }
        // Nothing is skipped by default since 2026-11-09: the mesh program 072eb70d (a11f465b) was drawing at positions from loads that read 0
        // (SM86_PTR_C0_MESH), the black and white blocks of 2026-11-08.
        const char *skip = ENV_ONCE("OOT_MESH_IND_SKIP");
        if (!skip) skip = "";
        for (const char *p = skip; p && *p;) {
            size_t k = strcspn(p, ",");
            char hx[2][17];
            snprintf(hx[0], 17, "%016llx", (unsigned long long)hp[0]), snprintf(hx[1], 17, "%016llx", (unsigned long long)hp[1]);
            if (k && k <= 16 && ((hp[0] && !strncmp(hx[0], p, k)) || !strncmp(hx[1], p, k))) return;
            p += k + (p[k] == ',');
        }
        // OOT_MESH_IND_ONLY=<hex>[,<hex>...]: only the draws whose mesh (or task) program hash starts with one of these.
        const char *only = ENV_ONCE("OOT_MESH_IND_ONLY");
        if (only && *only) {
            int hit = 0;
            for (const char *p = only; *p && !hit;) {
                size_t k = strcspn(p, ",");
                char hx[2][17];
                snprintf(hx[0], 17, "%016llx", (unsigned long long)hp[0]), snprintf(hx[1], 17, "%016llx", (unsigned long long)hp[1]);
                hit = k && k <= 16 && ((hp[0] && !strncmp(hx[0], p, k)) || !strncmp(hx[1], p, k));
                p += k + (p[k] == ',');
            }
            if (!hit) return;
        }
        static struct { uint64_t t, m; } seen[256];
        static int nseen;
        int known = 0;
        for (int i = 0; i < nseen && !known; i++) known = seen[i].t == hp[0] && seen[i].m == hp[1];
        if (!known && nseen < 256) {
            seen[nseen].t = hp[0], seen[nseen].m = hp[1], nseen++;
            rt_log("meshind: f%llu first indirect mesh draw with task %016llx mesh %016llx (arg %d, at %#llx)",
                   (unsigned long long)gfx_stats().frames, (unsigned long long)hp[0], (unsigned long long)hp[1], d->count,
                   (unsigned long long)d->ind);
            fflush(stdout), fflush(stderr);
        }
    } else if (d->first != 0 || d->count <= 0 || (uint32_t)d->count > MESH_MAX_TASKS)
        return warn_once(W_MESH, "a mesh draw from %d of %d task groups is skipped", d->first, d->count);
    static int ind_max = -1;
    if (ind_max < 0) {   // a number: rt_knob reads switches (0/1/ab) and made this bound 1 (2026-11-09)
        const char *e = getenv("OOT_MESH_IND_MAX");
        ind_max = e && *e ? atoi(e) : 64;
        rt_log("vk_exec: indirect mesh draws dispatch a bound of %d task (or mesh) groups (OOT_MESH_IND_MAX)", ind_max);
    }
    if (ind_max < 1) ind_max = 1;
    if (ind_max > (int)MESH_MAX_TASKS) ind_max = (int)MESH_MAX_TASKS;
    // A mesh-only indirect draw takes its count from the GPU for both the dispatch and the pass-through draw (2026-11-08): the plant lists reach
    // 1592 groups, and a bound that large on every one of a frame's draws cost 10x the frame (4 fps). OOT_MESH_MI=0: the bound, as task draws still do.
    static int mi_on = -1;
    if (mi_on < 0) { const char *e = getenv("OOT_MESH_MI"); mi_on = !(e && e[0] == '0'); }
    int mi = mi_on && indirect && !has_task, mt = mi_on && indirect && has_task && mc_ready();
    if ((mi || mt) && !MI.buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = MI_SLOTS * 32,
                                 .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &MI.buf, &MI.mem, NULL) != VK_SUCCESS) mi = mt = mi_on = 0;
    }
    uint32_t n = mi ? (uint32_t)MI_MAX_GROUPS : mt ? MESH_MAX_TASKS : indirect ? (uint32_t)ind_max : (uint32_t)d->count;
    static int mesh_shift = -1;   // OOT_MESH_SHIFT: log2 of the mesh workgroups a task can launch (test of the bound)
    if (mesh_shift < 0) {
        const char *e = getenv("OOT_MESH_SHIFT");
        mesh_shift = e && *e ? atoi(e) : (int)MESH_SHIFT;
        if (mesh_shift < 1 || mesh_shift > 10) mesh_shift = (int)MESH_SHIFT;
        rt_log("vk_exec: a mesh task launches at most %u mesh workgroups (OOT_MESH_SHIFT)", 1u << mesh_shift);
    }
    uint32_t tasks = has_task ? n : 1, shift = has_task ? (uint32_t)mesh_shift : 31;
    uint32_t groups = mt ? (uint32_t)MI_MAX_GROUPS : has_task ? tasks << shift : n;
    if (!mesh_buffer((VkDeviceSize)tasks * SM86_TASK_STRIDE, (VkDeviceSize)groups * SM86_MESH_STRIDE))
        return warn_once(W_MESH, "no memory for %u mesh records: mesh draws are skipped", groups);
    vk_exec_barrier(ex);
    if (!mt) vkCmdFillBuffer(ex->cb, M.buf, 0, M.task_size, 0), ex->recorded++;   // a task record no task wrote asks for no meshes (packed: only tasks that ran are read, and each clears its own)
    if (mt) {   // {x, 1, 1, task count}, then the draw: mesh_compact writes x and the instance count
        uint32_t prims;
        memcpy(&prims, (const uint8_t *)stage_entry(mesh, 7)->control + 0x60, 4);
        prims >>= 16;
        M.mi_off = (VkDeviceSize)(MI.next++ % MI_SLOTS) * 32;
        vkCmdFillBuffer(ex->cb, MI.buf, M.mi_off + 4, 8, 1);
        vkCmdFillBuffer(ex->cb, MI.buf, M.mi_off + 16, 4, 3 * prims);
        vkCmdFillBuffer(ex->cb, MI.buf, M.mi_off + 24, 8, 0);
        VkBufferCopy c1 = {ip.offset, M.mi_off + 12, 4};
        vkCmdCopyBuffer(ex->cb, ip.buffer, MI.buf, 1, &c1);
        ex->recorded += 4;
    }
    static int mesh_clear = -1;   // OOT_MESH_CLEAR=1 (diagnostic): the mesh records too, to tell unwritten vertices from wrong ones
    if (mesh_clear < 0) mesh_clear = getenv("OOT_MESH_CLEAR") != NULL;
    if (mesh_clear) vkCmdFillBuffer(ex->cb, M.buf, M.task_size, M.mesh_size, 0), ex->recorded++;
    if (!has_task) {
        vk_exec_barrier_compute(ex);   // transfers are not ordered among themselves: the fill could land after the update
        // a mesh-only draw: one stand-in task whose count (+4) asks for every mesh group (an indirect draw: its count).
        // A fill or copy, not vkCmdUpdateBuffer: that has no recorder wrapper (vk_rec.h), so it went into the command
        // buffer at once, out of order and from a second thread while the recorder wrote the rest (run-68's device
        // loss had it in every mesh-only draw).
        if (indirect) {
            VkBufferCopy cp = {ip.offset, 4, 4};
            vkCmdCopyBuffer(ex->cb, ip.buffer, M.buf, 1, &cp);
            if (mi) {   // the group count into the dispatch's x and the draw's instance count
                uint32_t prims;
                memcpy(&prims, (const uint8_t *)stage_entry(mesh, 7)->control + 0x60, 4);
                prims >>= 16;
                M.mi_off = (VkDeviceSize)(MI.next++ % MI_SLOTS) * 32;
                vkCmdFillBuffer(ex->cb, MI.buf, M.mi_off + 4, 8, 1);
                vkCmdFillBuffer(ex->cb, MI.buf, M.mi_off + 12, 4, 0);
                vkCmdFillBuffer(ex->cb, MI.buf, M.mi_off + 16, 4, 3 * prims);
                vkCmdFillBuffer(ex->cb, MI.buf, M.mi_off + 24, 8, 0);
                VkBufferCopy c2[2] = {{ip.offset, M.mi_off, 4}, {ip.offset, M.mi_off + 20, 4}};
                vkCmdCopyBuffer(ex->cb, ip.buffer, MI.buf, 2, c2);
                ex->recorded += 5;
                M.mi = 1;
            }
        } else {
            vkCmdFillBuffer(ex->cb, M.buf, 4, 4, ~0u);
        }
        ex->recorded++;
    }
    M.on = 1, M.shift = shift, M.count = tasks;
    M.ind = indirect && has_task ? ip : (VkMemSpan){0};
    M.map = mt;
    int okt = !has_task || mesh_dispatch(ex, 0, tasks);
    if (okt && mt) mesh_compact(ex), M.mi = 1;
    int ok = okt && mesh_dispatch(ex, 1, groups);
    if (ENV_ONCE("OOT_MESH_LOG")) {   // which programs ran, and the task stage's uniform slot 0 as the game left it
        static volatile LONG n;
        if (InterlockedIncrement(&n) <= 12) {
            uint64_t ht = 0, hm = 0;
            for (int m = 0; m < 2; m++) {
                const NVNshaderData *e = stage_entry(ex->mesh_prog[m], 6 + m);
                if (!e) continue;
                uint32_t len;
                memcpy(&len, (const uint8_t *)e->control + 0x44, 4);
                uint64_t h = 0xcbf29ce484222325ull;
                for (uint32_t i = 0; i < len; i++) h = (h ^ ((const uint8_t *)(uintptr_t)e->data)[i]) * 0x100000001b3ull;
                if (m) hm = h; else ht = h;
            }
            const p_bind_buf *u = &ex->mubo[0][0];
            uint32_t w[8] = {0};
            VkMemSpan sp;
            if (u->addr && u->size >= 32 && vk_mem_resolve(u->addr, 32, &sp)) memcpy(w, (const void *)(uintptr_t)u->addr, 32);
            rt_log("meshlog: draw of %d: task %016llx ran %d, mesh %016llx ran %d; task ubo0 %#llx+%#llx: %#x %#x %#x %#x %#x %#x",
                   d->count, (unsigned long long)ht, okt, (unsigned long long)hm, ok, (unsigned long long)u->addr,
                   (unsigned long long)u->size, w[0], w[1], w[2], w[3], w[4], w[5]);
        }
    }
    M.on = 0, M.ind = (VkMemSpan){0};
    vk_exec_barrier(ex);
    if (ok) mesh_tasks_note(ex, indirect ? &ip : NULL, d->ind, has_task ? tasks : 0, shift);
    if (ok && indirect) ms_note(ex, &ip, d->ind);
    if (ok) mesh_rec_note(ex, groups);
    if (vk_dump_armed) {   // frame dumps: which task and mesh programs the next pass-through draw stands for
        uint64_t hq[2] = {0, 0};
        for (int m = 0; m < 2; m++) {
            const NVNshaderData *e = stage_entry(ex->mesh_prog[m], 6 + m);
            if (!e) continue;
            uint32_t len;
            memcpy(&len, (const uint8_t *)e->control + 0x44, 4);
            uint64_t h = 0xcbf29ce484222325ull;
            for (uint32_t i = 0; i < len; i++) h = (h ^ ((const uint8_t *)(uintptr_t)e->data)[i]) * 0x100000001b3ull;
            hq[m] = h;
        }
        vk_dump_note("mesh draw (%s, %d): task %016llx mesh %016llx, %u groups, stages ran %d", indirect ? "indirect" : "direct",
                     d->count, (unsigned long long)hq[0], (unsigned long long)hq[1], groups, ok);
    }
    if (ok && !ENV_ONCE("OOT_MESH_NODRAW")) mesh_draw(ex, mesh, groups);
    M.mi = 0, M.map = 0;
    int slot = ok ? mesh_rb_slot() : -1;
    if (slot >= 0) {
        VkBufferCopy cp[13];
        VkDeviceSize o = (VkDeviceSize)slot * 64;
        cp[0] = (VkBufferCopy){0, o, 16};
        for (uint32_t g = 0; g < 8; g++)
            cp[1 + g] = (VkBufferCopy){M.task_size + (VkDeviceSize)(g < groups ? g : 0) * SM86_MESH_STRIDE + SM86_MESH_MAP, o + 16 + 4 * g, 4};
        for (uint32_t c = 0; c < 4; c++) cp[9 + c] = (VkBufferCopy){M.task_size + 0x80 * c, o + 48 + 4 * c, 4};
        vkCmdCopyBuffer(ex->cb, M.buf, MRB.buf, 13, cp);
        ex->recorded++;
        mesh_rb_done(ex, slot, 0, tasks, groups, 0);
    }
}

// OP_PUSH_CC (NVN2 nvnCommandBufferPushComputeConstants): the pushed bytes into the compute stage's block, which
// c0_info serves as constant bank 0 (the SM86 compute shaders' parameter block: sizes, scales, buffer addresses and
// texture and image handles, each at the offsets the shader reads).
static void op_push_cc(VkExec *ex, const p_update_ub *u) {
    if (!u->size || u->dst + u->size > VK_C0_BYTES) return;
    memcpy(ex->push_cc + u->dst, u->data, u->size);
    if (ex->push_cc_len < u->dst + u->size) ex->push_cc_len = (uint32_t)(u->dst + u->size);
    push_fresh = 1;
    B.c0_valid[2] = 0;
}

// After a dispatch the pushed constants stay for the next ones until the game pushes over them. Measured 2026-11-10:
// the frame's anti-aliasing pushes 116 bytes once, then dispatches ceea6f19, 2a9fe9a8 (indirect) and 1cb856c3, and
// 1cb856c3 takes its last pixel (1919, 1079) and offset from that push. With the push dropped after each dispatch (a
// guess since 56eac50) it read zeros, stored no pixel, and the screen got the scene before tone mapping: too dark in
// dark scenes, white in bright ones. OOT_PUSH_KEEP=0 drops it after each dispatch again (diagnostic).
static void push_after_dispatch(VkExec *ex) {
    static int keep = -1;
    if (keep < 0) { const char *e = getenv("OOT_PUSH_KEEP"); keep = !(e && e[0] == '0'); }
    if (push_hidden) ex->push_cc_len = push_hidden - 1, push_hidden = 0;   // OOT_PUSH_NOINHERIT hid it from one dispatch
    if (!keep) ex->push_cc_len = 0;
    push_fresh = 0;
}

static void op_update_ub(VkExec *ex, const p_update_ub *u) {
    if (!u->size) return;
    VkMemSpan sp;
    if (!vk_mem_resolve(u->dst, u->size, &sp)) return warn_once(W_UB, "UpdateUniformBuffer to %#llx is not backed: dropped", (unsigned long long)u->dst);
    cb_res(ex);
    vk_exec_barrier(ex);   // earlier draws finished with the old contents
    VkBuffer b;
    VkDeviceSize o;
    memcpy(scratch(ex, u->size, &b, &o), u->data, u->size);
    VK_MEM_PICK(sp);
    VkBufferCopy cp = {o, sp.offset, u->size};
    vkCmdCopyBuffer(ex->cb, b, sp.buffer, 1, &cp);
    vk_mem_gpu_wrote(&sp, u->size);
    vk_usnap_drop(u->dst, u->size);
    ex->recorded++;
    vk_mem_count(sp.cls, VK_MUSE_GPU_WRITE, u->size);
    note_written(ex, u->dst, u->size);
    upd_note(ex, u);
}

// ---- bind ops

// A handle's texture id (20 bits) and sampler id (12 bits at bit 20) index the pools bound at this point of the
// stream; the slot tables are shared by every pool, so they become slots here (nvn_api_obj.c nvn_tpool_slot), once,
// and every lookup and memo after this sees slots. kind: 0 texture (both ids), 1 sampler (the sampler id), 2 image.
static uint64_t pool_handle(const VkExec *ex, uint64_t h, int kind) {
    if (!h) return 0;
    uint64_t tid = h & 0xfffff, sid = h >> 20 & 0xfff;   // the game's encoding: 20 and 12 bits
    if (kind != 1) tid = (uint64_t)nvn_tpool_slot(ex->tex_pool, (int)tid);
    if (kind != 2) sid = (uint64_t)nvn_spool_slot(ex->samp_pool, (int)sid);
    return tid | sid << 20;   // the replay's: 20 and 20 bits (NVN_HANDLE_TEX, NVN_HANDLE_SAMP)
}

static void bind_handles(const VkExec *ex, uint64_t *dst, int cap, const p_bind_handles *b, int kind) {
    if (!dst || b->stage < 0 || b->stage >= 8 || b->first < 0 || b->n < 0 || b->n > 32 || b->first + b->n > cap) {
        warn_once(W_BIND, "a bind of %d handles at %d of stage %d is out of range: ignored", b->n, b->first, b->stage);
        return;
    }
    for (int i = 0; i < b->n; i++) dst[b->first + i] = pool_handle(ex, b->h[i], kind);
}

// ---- separable programs
// nvnCommandBufferBindProgram(cmd, program, stages) replaces the program of the stages in the bit mask only (stage bit
// i is NVN stage i: vertex 0, fragment 1, geometry 2, tessellation control 3 and evaluation 4, compute 5). TotK binds
// whole programs; FE3H binds a vertex program and a fragment program separately. The rest of the replay reads one
// program (ex->prog) per draw, so when the stages come from different programs ex->prog is a combined program: the
// shader each stage's program has for that stage, under a serial of its own, so the shader, pipeline and memo caches
// keyed by serial keep working. Combinations are kept in a direct-mapped table; a replaced entry is not freed, as a
// recording made before may still name it (they are small and few).
#define COMBO_N 1024
static struct Combo { const prog_t *src[6]; uint64_t src_serial[6]; prog_t *prog; } combos[COMBO_N];
static SRWLOCK combo_lock = SRWLOCK_INIT;

// The shader data entry of p for NVN stage st, from its control block's stage word (find_stage_scan).
static const NVNshaderData *stage_entry(const prog_t *p, int st) {
    for (uint32_t i = 0; p && i < p->nstage && i < 6; i++) {
        const uint8_t *c = p->sd[i].control;
        if (!c || !p->sd[i].data) continue;
        if (c[0x56] == st) return &p->sd[i];
    }
    return NULL;
}

static const prog_t *combined_program(const prog_t *const sp[6]) {
    const prog_t *one = NULL;
    int mixed = 0;
    for (int st = 0; st < 6; st++) {
        if (!sp[st] || !stage_entry(sp[st], st)) continue;   // a program bound for a stage it has no shader for
        if (one && sp[st] != one) mixed = 1;
        if (!one) one = sp[st];
    }
    if (!mixed) return one;
    uint64_t h = 0;
    for (int st = 0; st < 6; st++) h = (h ^ (uintptr_t)sp[st] ^ (sp[st] ? sp[st]->serial << 1 : 0)) * 0x9E3779B97F4A7C15ull;
    struct Combo *e = &combos[h >> 54];
    AcquireSRWLockExclusive(&combo_lock);
    int hit = e->prog != NULL;
    for (int st = 0; hit && st < 6; st++)
        hit = e->src[st] == sp[st] && e->src_serial[st] == (sp[st] ? sp[st]->serial : 0);
    if (!hit) {
        prog_t *c = calloc(1, sizeof *c);
        if (!c) rt_fatal("vk_exec: out of memory (combined program)");
        c->magic = one->magic;
        for (int st = 0; st < 6; st++) {
            const NVNshaderData *d = sp[st] ? stage_entry(sp[st], st) : NULL;
            if (d && c->nstage < 6) c->sd[c->nstage++] = *d;
        }
        c->serial = nvn_new_serial();
        e->prog = c;
        for (int st = 0; st < 6; st++) e->src[st] = sp[st], e->src_serial[st] = sp[st] ? sp[st]->serial : 0;
    }
    const prog_t *r = e->prog;
    ReleaseSRWLockExclusive(&combo_lock);
    return r;
}

static void bind_program(VkExec *ex, const p_bind_program *b) {
    for (int st = 0; st < 6; st++)
        if (b->stages & (1 << st)) ex->stage_prog[st] = b->prog;
    for (int st = 6; st < 8; st++)   // task and mesh (op_draw_mesh)
        if (b->stages & (1 << st)) ex->mesh_prog[st - 6] = b->prog;
    ex->prog = combined_program(ex->stage_prog), ex->prog_stages = b->stages;
}

void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload) {
    switch (op) {
    case OP_BIND_PROGRAM: {
        bind_program(ex, payload);
        bind_gen++;
        break;
    }
    case OP_BIND_STATE: {
        const p_bind_state *b = payload;
        blend_st bs;
        switch (b->kind) {
        case 0:
            memcpy(&bs, b->bytes, sizeof bs);
            if (bs.target < 8) ex->pipe.blend[bs.target] = bs;
            break;
        case 1: memcpy(&ex->pipe.color, b->bytes, sizeof ex->pipe.color); break;
        case 2: memcpy(&ex->pipe.chmask, b->bytes, sizeof ex->pipe.chmask); break;
        case 3: memcpy(&ex->pipe.ms, b->bytes, sizeof ex->pipe.ms); break;
        case 4: memcpy(&ex->pipe.poly, b->bytes, sizeof ex->pipe.poly); break;
        case 5: memcpy(&ex->pipe.ds, b->bytes, sizeof ex->pipe.ds); break;
        case 6: memcpy(&ex->point_size, b->bytes, sizeof ex->point_size); break;
        }
        break;
    }
    case OP_BIND_VATTR: {
        const p_bind_vattr *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memset(ex->pipe.vattr, 0, sizeof ex->pipe.vattr);   // MNG: the new list replaces the old one
        memcpy(&ex->pipe.vattr[b->first], b->st, sizeof(uint32_t) * (size_t)b->n);
        break;
    }
    case OP_BIND_VSTREAM: {
        const p_bind_vstream *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memset(ex->pipe.vstream, 0, sizeof ex->pipe.vstream);
        memcpy(&ex->pipe.vstream[b->first], b->st, sizeof(vs_st) * (size_t)b->n);
        break;
    }
    case OP_BIND_VBUF: {
        const p_bind_vbuf *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memcpy(&ex->vbuf[b->first], b->r, sizeof(NVNbufferRange) * (size_t)b->n);
        break;
    }
    case OP_BIND_UBO: {
        const p_bind_buf *b = payload;
        if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 18) ex->ubo[b->stage][b->index] = *b;
        else if ((b->stage == 6 || b->stage == 7) && b->index >= 0 && b->index < 18) ex->mubo[b->stage - 6][b->index] = *b;
        bind_gen++;
        break;
    }
    case OP_BIND_SSBO: {
        const p_bind_buf *b = payload;
        if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16) ex->ssbo[b->stage][b->index] = *b;
        else if ((b->stage == 6 || b->stage == 7) && b->index >= 0 && b->index < 16) ex->mssbo[b->stage - 6][b->index] = *b;
        bind_gen++;
        break;
    }
    case OP_BIND_TEX: {
        const p_bind_handles *b = payload;
        bind_handles(ex, b->stage >= 0 && b->stage < 6 ? ex->tex[b->stage] : b->stage == 6 || b->stage == 7 ? ex->mtex[b->stage - 6] : NULL, 128, b, 0);
        ex->tex_serial++;
        bind_gen++;
        break;
    }
    case OP_BIND_SAMP: {
        const p_bind_handles *b = payload;
        bind_handles(ex, b->stage >= 0 && b->stage < 6 ? ex->samp[b->stage] : b->stage == 6 || b->stage == 7 ? ex->msamp[b->stage - 6] : NULL, 128, b, 1);
        break;
    }
    case OP_BIND_IMG: {
        const p_bind_handles *b = payload;
        bind_handles(ex, b->stage >= 0 && b->stage < 6 ? ex->img[b->stage] : b->stage == 6 || b->stage == 7 ? ex->mimg[b->stage - 6] : NULL, 32, b, 2);
        ex->tex_serial++;   // the shader memo's handle check (feedback_scan reads only textures: a rescan at most)
        bind_gen++;
        break;
    }
    case OP_UPDATE_UB: op_update_ub(ex, payload); break;
    case OP_PUSH_CC: op_push_cc(ex, payload); break;
    case OP_DRAW: op_draw(ex, payload); break;
    case OP_DRAW_IDX: op_draw_idx(ex, payload); break;
    case OP_DISPATCH: op_dispatch(ex, payload); push_after_dispatch(ex); break;
    case OP_DRAW_IDX_IND: op_draw_idx_ind(ex, payload); break;
    case OP_DRAW_IND: op_draw_ind(ex, payload); break;
    case OP_DISPATCH_IND: op_dispatch_ind(ex, payload); push_after_dispatch(ex); break;
    case OP_DRAW_MESH: case OP_DRAW_MESH_IND: op_draw_mesh(ex, payload, op == OP_DRAW_MESH_IND); break;
    default: rt_fatal("vk: op %s is not a draw op", nvn_op_name(op));
    }
}

// Transform feedback (nvn_api_cmd.c BindTransformFeedbackBuffer, Begin/EndTransformFeedback), replayed with
// VK_EXT_transform_feedback. This game brackets a few rasterizer-discard draws every few seconds whose vertex shader
// transforms positions for the CPU to read back (the buffer is 1 MiB, the control buffer is not written yet). The
// bindings and the open begin are replay state; the first draw after the begin binds the buffers and begins the
// capture in the open pass (xfb_before_draw, after prepare_draw bound the pipeline that carries the layout), the end
// op or the pass end ends it, and a barrier after the pass hands the vertices to the write-back (bind_range with
// written: the range goes back to guest memory at the fence). The calls go through the recorder (vk_rec.c
// rec_xfb_*), in order with the draws. vk_dump_draw marks the captured draws "xfb" in the pass log.
static PFN_vkCmdBindTransformFeedbackBuffersEXT xfb_bind_fn;
static PFN_vkCmdBeginTransformFeedbackEXT xfb_begin_fn;
static PFN_vkCmdEndTransformFeedbackEXT xfb_end_fn;
static int xfb_fns(void) {
    static int tried, ok;
    if (tried) return ok;
    tried = 1;
    VkDevice dev = vk_ctx()->device;
    if (vk_ctx()->has_transform_feedback) {
        xfb_bind_fn = (PFN_vkCmdBindTransformFeedbackBuffersEXT)vkGetDeviceProcAddr(dev, "vkCmdBindTransformFeedbackBuffersEXT");
        xfb_begin_fn = (PFN_vkCmdBeginTransformFeedbackEXT)vkGetDeviceProcAddr(dev, "vkCmdBeginTransformFeedbackEXT");
        xfb_end_fn = (PFN_vkCmdEndTransformFeedbackEXT)vkGetDeviceProcAddr(dev, "vkCmdEndTransformFeedbackEXT");
        ok = xfb_bind_fn && xfb_begin_fn && xfb_end_fn;
    }
    if (!ok) rt_log("vk_exec: transform feedback is not replayed: the device has no VK_EXT_transform_feedback");
    return ok;
}
static void xfb_before_draw(VkExec *ex) {
    if (!ex->xfb_active || ex->xfb_begun || !ex->in_pass || !xfb_fns()) return;
    VkBuffer bufs[4];
    VkDeviceSize offs[4], sizes[4];
    uint32_t n = 0;
    for (int i = 0; i < 4; i++) {
        if (!ex->xfb_buf[i].addr || !ex->xfb_buf[i].size) break;
        VkDescriptorBufferInfo bi;
        if (!bind_range(ex, ex->xfb_buf[i].addr, ex->xfb_buf[i].size, 4, ex->xfb_buf[i].size, VK_MUSE_GPU_WRITE, 1, &bi)) {
            static int said;
            if (said++ < 4) rt_log("vk_exec: transform feedback buffer %d at %#llx is not backed: not captured", i, (unsigned long long)ex->xfb_buf[i].addr);
            return;
        }
        note_written(ex, ex->xfb_buf[i].addr, bi.range);
        bufs[n] = bi.buffer, offs[n] = bi.offset, sizes[n] = bi.range, n++;
    }
    if (!n) return;
    rec_xfb_bind(xfb_bind_fn, ex->cb, 0, n, bufs, offs, sizes);   // through the recorder, in order with the draws
    rec_xfb_begin(xfb_begin_fn, ex->cb);
    ex->xfb_begun = 1;
    ex->recorded += 2;
    static int said;
    if (said++ < 4)
        rt_log("vk_exec: transform feedback begins into %u buffer%s (%#llx, %llu bytes) in frame %llu", n, n == 1 ? "" : "s",
               (unsigned long long)ex->xfb_buf[0].addr, (unsigned long long)sizes[0], (unsigned long long)gfx_counters.frames);
}
// OOT_DUMP_XFB_PROBE=1 (with OOT_DUMP and OOT_DUMP_DRAWS): every indexed draw of that pass gets its vertex shader
// with the transform feedback layout (spirv_add_xfb: the located outputs in location order; the vk_pipeline log line
// gives it) and captures into a host buffer of its own, written as xfb_p<pass>_d<draw>.bin once the GPU is done:
// what the GPU computed, to set beside a CPU evaluation of the same shader on the dumped inputs.
struct ProbeJob { VkBuffer buf; VmaAllocation alloc; const uint8_t *map; char path[600]; };
static void probe_done(void *arg) {
    ProbeJob *j = arg;
    vmaInvalidateAllocation(vk_ctx()->vma, j->alloc, 0, VK_WHOLE_SIZE);
    FILE *f = fopen(j->path, "wb");
    if (f) fwrite(j->map, 1, 65536, f), fclose(f);
    vmaDestroyBuffer(vk_ctx()->vma, j->buf, j->alloc);
    free(j);
}
static ProbeJob *probe_begin(VkExec *ex, int draw) {
    if (!ex->in_pass || !xfb_fns()) return NULL;
    ProbeJob *j = calloc(1, sizeof *j);
    if (!j) return NULL;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 65536,
                             .usage = VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &j->buf, &j->alloc, &info) != VK_SUCCESS || !info.pMappedData) {
        free(j);
        return NULL;
    }
    j->map = info.pMappedData;
    memset(info.pMappedData, 0xcd, 65536);
    snprintf(j->path, sizeof j->path, "%s/xfb_p%04d_d%03d.bin", vk_dump_dir(), vk_dump_pass_seq(), draw);
    VkDeviceSize off = 0, size = 65536;
    rec_xfb_bind(xfb_bind_fn, ex->cb, 0, 1, &j->buf, &off, &size);
    rec_xfb_begin(xfb_begin_fn, ex->cb);
    ex->recorded += 2;
    return j;
}
static void probe_end(VkExec *ex, ProbeJob *j) {
    if (!j) return;
    rec_xfb_end(xfb_end_fn, ex->cb);
    ex->recorded++;
    ex->xfb_ended = 1;   // the barrier after the pass (vk_exec_xfb_pass_end)
    vk_completion_add(ex->fence, probe_done, j);
}

void vk_exec_xfb_pass_end(VkExec *ex, int after) {
    if (!after) {
        if (!ex->xfb_begun) return;
        rec_xfb_end(xfb_end_fn, ex->cb);
        ex->xfb_begun = 0, ex->xfb_ended = 1, ex->recorded++;
        if (ex->xfb_active) {
            static int said;
            if (said++ < 4) rt_log("vk_exec: a pass ended inside a transform feedback bracket: the next pass captures from offset 0 again");
        }
        return;
    }
    if (!ex->xfb_ended) return;
    ex->xfb_ended = 0;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);   // the recorder's wrapper: after its EndRendering
    ex->recorded++;
}
void vk_exec_op_xfb(VkExec *ex, uint32_t op, const void *payload) {
    switch (op) {
    case OP_XFB_BIND: {
        const p_xfb_bind *b = payload;
        if (b->index >= 0 && b->index < 4) ex->xfb_buf[b->index].addr = b->addr, ex->xfb_buf[b->index].size = b->size;
        break;
    }
    case OP_XFB_BEGIN: ex->xfb_active = 1, ex->xfb_control = ((const p_xfb_ctl *)payload)->control; break;
    case OP_XFB_END: {
        if (ex->xfb_begun) rec_xfb_end(xfb_end_fn, ex->cb), ex->xfb_begun = 0, ex->xfb_ended = 1, ex->recorded++;
        ex->xfb_active = 0;
        break;
    }
    default: break;
    }
}
