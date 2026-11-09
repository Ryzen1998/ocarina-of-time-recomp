// Written for ocarina-of-time-recomp (MIT, Copyright (c) 2026 Jeff Aigner).
// OOT_GPU_ACCT=<first frame>[:<frames>[:<every>]] (diagnostic, off by default): the GPU time of everything the runtime
// submits, for a window of frames (repeated every <every> frames when given), written as one CSV file that
// tools/gpu_acct.py turns into a per-category account of the frame. OOT_PASS_TIME, OOT_CS_TIME and OOT_TRACE=perf each
// see part of a frame; the perf line in particular sums only the replay's command buffers, so a frame of 20 ms showed
// 12.7 ms of GPU time with the card reporting 85% busy (docs/phase3.md "Forest speed: what bounds the frame"). Here the
// command buffers of every submission (replay parts, mirror uploads, upload batches, the present) and what is inside the
// replay's (render passes, dispatches, the mesh stages, copies) are stamped with one clock.
//
// Slots are handed out in order within a window and all read after it, so a window costs one query reset and one read
// and nothing per frame. Timestamps are at ALL_COMMANDS, which on this driver waits for the work before them: the
// numbers are slightly pessimistic for overlapping work. Compare a run with and without the switch.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"

enum { GA_PAIRS = 64000 };
typedef struct { uint64_t frame, id, a, b; uint32_t cat, closed; } GaRec;
static VkQueryPool pool;
static GaRec *rec;
static volatile LONG next_pair;
static volatile int window_on;
static long long first = -1, count = 60, every;
static long long win_from, win_to;
static int ready;
static int state;   // 0 not parsed, 1 parsed and on, -1 off
static char path[260];
static FILE *out;
static double period;
static uint64_t mask;

static void parse(void) {
    state = -1;
    const char *e = getenv("OOT_GPU_ACCT");
    if (!e || !*e) return;
    long long a = -1, b = 60, c = 0;
    if (sscanf(e, "%lld:%lld:%lld", &a, &b, &c) < 1 || a < 0 || b <= 0) {
        rt_log("vk: OOT_GPU_ACCT=%s is not <first>[:<frames>[:<every>]]: off", e);
        return;
    }
    first = a, count = b, every = c;
    win_from = first, win_to = first + count;
    const char *f = getenv("OOT_GPU_ACCT_FILE");
    snprintf(path, sizeof path, "%s", f && *f ? f : "build/logs/gpu-acct.csv");
    state = 1;
}

static int setup(void) {
    if (ready) return ready > 0;
    ready = -1;
    VkCtx *c = vk_ctx();
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
    if (!qp) return 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, qp);
    uint32_t bits = c->gfx_family < nq ? qp[c->gfx_family].timestampValidBits : 0;
    free(qp);
    period = c->props.limits.timestampPeriod;
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                .queryCount = 2 * GA_PAIRS};
    rec = calloc(GA_PAIRS, sizeof *rec);
    if (!bits || period <= 0 || !rec || vkCreateQueryPool(c->device, &qi, NULL, &pool) != VK_SUCCESS) {
        rt_log("vk: OOT_GPU_ACCT: no timestamps here: off");
        return 0;
    }
    mask = bits >= 64 ? ~0ull : (1ull << bits) - 1;
    out = fopen(path, "w");
    if (!out) {
        rt_log("vk: OOT_GPU_ACCT: cannot write %s: off", path);
        return 0;
    }
    fprintf(out, "window,frame,cat,id,a,b,start_ns,end_ns\n");
    rt_log("vk: OOT_GPU_ACCT: frames %lld..%lld (every %lld) timed on the GPU, into %s", first, first + count - 1, every, path);
    ready = 1;
    return 1;
}

int vk_ga_on(void) { return __atomic_load_n(&window_on, __ATOMIC_ACQUIRE); }

// The driver writes a timestamp where the command stream is, without waiting for the rendering still in flight: run-874
// without this drain billed the rendering to whichever barrier came next (mirror copies 13 ms, passes 3 ms). Every stamp
// therefore comes after a full barrier, so an interval closes when its work is done. OOT_GPU_ACCT_DRAIN=0: off.
static void drain(VkCommandBuffer cb) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("OOT_GPU_ACCT_DRAIN"); on = !(e && e[0] == '0'); }
    if (!on) return;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

int vk_ga_open(VkCommandBuffer cb) {
    if (!__atomic_load_n(&window_on, __ATOMIC_ACQUIRE)) return -1;
    LONG p = InterlockedIncrement(&next_pair) - 1;
    if (p >= GA_PAIRS) return -1;
    drain(cb);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool, 2 * (uint32_t)p);
    return (int)p;
}

void vk_ga_close(VkCommandBuffer cb, int pair, int cat, uint64_t id, uint64_t a, uint64_t b) {
    if (pair < 0) return;
    drain(cb);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool, 2 * (uint32_t)pair + 1);
    GaRec *r = &rec[pair];
    r->frame = gfx_counters.frames, r->cat = (uint32_t)cat, r->id = id, r->a = a, r->b = b;
    __atomic_store_n(&r->closed, 1, __ATOMIC_RELEASE);
}

// Reading a window: every submission that holds one of its stamps is complete a few frames later.
static void flush_window(int window) {
    VkCtx *c = vk_ctx();
    LONG n = next_pair > GA_PAIRS ? GA_PAIRS : next_pair;
    uint64_t *v = calloc((size_t)n * 4, sizeof *v);
    if (!v) return;
    vkGetQueryPoolResults(c->device, pool, 0, 2 * (uint32_t)n, (size_t)n * 32, v, 16,
                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    int written = 0, lost = 0;
    for (LONG i = 0; i < n; i++) {
        const GaRec *r = &rec[i];
        const uint64_t *q = v + 4 * i;
        if (!r->closed) continue;
        if (!q[1] || !q[3]) { lost++; continue; }
        uint64_t s = (uint64_t)((double)(q[0] & mask) * period), e = (uint64_t)((double)(q[2] & mask) * period);
        fprintf(out, "%d,%llu,%u,%llx,%llu,%llu,%llu,%llu\n", window, (unsigned long long)r->frame, r->cat,
                (unsigned long long)r->id, (unsigned long long)r->a, (unsigned long long)r->b, (unsigned long long)s,
                (unsigned long long)e);
        written++;
    }
    fflush(out);
    free(v);
    rt_log("vk: OOT_GPU_ACCT: window %d: %d intervals written (%d of %ld slots unused or unfinished, %d not available)",
           window, written, (int)n - written, (long)n, lost);
}

// At each present (vk_backend.c): opens and closes the windows.
void vk_ga_frame(uint64_t frame) {
    static int window;
    static long long drain_at = -1;
    if (!state) parse();
    if (state < 0) return;
    if ((long long)frame >= win_from && (long long)frame < win_to && !window_on && drain_at < 0) {
        if (!setup()) { state = -1; return; }
        vkResetQueryPool(vk_ctx()->device, pool, 0, 2 * GA_PAIRS);
        memset(rec, 0, GA_PAIRS * sizeof *rec);
        next_pair = 0;
        __atomic_store_n(&window_on, 1, __ATOMIC_RELEASE);
    } else if ((long long)frame >= win_to && window_on) {
        __atomic_store_n(&window_on, 0, __ATOMIC_RELEASE);
        drain_at = frame + 12;
    } else if (drain_at >= 0 && (long long)frame >= drain_at) {
        flush_window(window++);
        drain_at = -1;
        if (every > 0) win_from += every, win_to += every;
        else state = -1, fclose(out), out = NULL;
    }
}
