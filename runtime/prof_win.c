// From metroid-prime-4-recomp runtime/prof_win.c @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied with mp4 renamed to oot.
// From fe3h-pc-recomp runtime/prof_win.c @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied with fe3h renamed to mp4 only.
// From totk-pc-recomp runtime/prof_win.c @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), copied with totk renamed to fe3h only.
// Sampling profiler (OOT_PROFILE=<path>) and per-frame timing (OOT_TRACE=perf). tools/prof_report.py reads the profile.
// A host thread wakes every 1 ms and, for every other thread, suspends it, reads Rip and resumes it. The sampler never
// allocates or takes a lock between SuspendThread and ResumeThread and never runs guest code. The profile is text:
//   oot-prof 1 / base <hex> / period_us / samples N / thread <tid> <samples> <in-exe samples> / module <base> <size> <path> /
//   rip <hex> <count> (the top PROF_TOP Rips) / stack <count> <rip> <caller> ... (OOT_PROFILE_RENDER or
//   OOT_PROFILE_CHAINS only, the top PROF_STACKS call chains)
// With OOT_PROFILE_RENDER (the render thread) or OOT_PROFILE_CHAINS=1 (every thread, chains of all threads summed: for
// short frame ranges, as each sample copies a stack) the sampler also copies up to STK_COPY bytes of the suspended
// thread's stack and, after
// ResumeThread, unwinds the copy with the .pdata tables (RtlVirtualUnwind), so it takes no lock while the thread is
// suspended. The unwind stops at the first frame outside the copy; deeper callers are cut.
// OOT_PROFILE_SLOW=<ms> keeps every thread's chains of the last half second in a ring (a thread outside the exe, mostly
// waiting, copies 8 KiB of stack, one in it STK_COPY; a thread that ran no cycle since its previous sample is not
// suspended and repeats its previous chain; a sample that took over 2 ms to stop its thread, which happens while the
// thread is in a long kernel call, is written as `tstall <tid> <ms into the window> <ms stopping> <rip>...` with the
// chain where the thread stopped) and, after a present at least <ms> after the previous one, writes
// that interval's samples (plus the 20 ms before) to <OOT_PROFILE>.f<frame>: the profile format, one `stack 1` line per
// sample, and `tstack <tid> <ms into the window> <rip>...` lines that keep the thread and the time. At most 64 dumps.
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "flight.h"
#include "gfx.h"
#include "host.h"
#include "perfmon.h"

#define PROF_SLOTS (1u << 16)   // Rip hash table, open addressing; a full table drops new Rips
#define PROF_TOP 4000           // more than the 200 asked for: the report sums per function
#define PROF_THREADS 512
#define STK_SLOTS (1u << 15)      // call-chain hash table, open addressing; a full table drops new chains
#define PROF_STACKS 6000
#define STK_COPY (64u << 10)      // stack bytes copied per sample
#define STK_SLACK (64u << 10)     // zeros after the copy: a frame reaching past it reads zeros, and a zero Rip ends the chain

typedef struct { uint64_t rip, n; } RipCount;
#define PROF_DEPTH 24             // frames kept per call chain, the sampled Rip first
// cycles, depth, last: OOT_PROFILE_SLOW only, the thread's cycle count and chain at its previous sample
typedef struct { DWORD tid; HANDLE h; uint64_t n, in_exe, cycles; uint32_t depth; uint64_t last[PROF_DEPTH]; } ThreadSlot;

static RipCount table[PROF_SLOTS];
typedef struct { uint64_t hash, n; uint32_t depth; uint64_t rip[PROF_DEPTH]; } StackCount;
static StackCount stacks[STK_SLOTS];
static uint64_t stacks_dropped;
static uint8_t stk_buf[STK_COPY + STK_SLACK] __attribute__((aligned(16)));   // the slack is never written
static ThreadSlot threads[PROF_THREADS];
static int n_threads;
static uint64_t total, dropped;
static const char *out_path;
static HANDLE sampler;
static volatile LONG stop, written;
static uintptr_t exe_lo, exe_hi;
static volatile DWORD render_tid;   // the thread that presents (perf_present)
static int only_render;             // OOT_PROFILE_RENDER=1: sample the render thread only
static volatile DWORD main_tid;     // the guest main thread (kern_main_thread)
static int only_main;               // OOT_PROFILE_MAIN=1: sample the guest main thread only
static int chains;                  // call chains are kept: OOT_PROFILE_RENDER or OOT_PROFILE_CHAINS
static volatile uint64_t cur_frame; // last presented frame
static uint64_t frame_lo, frame_hi; // OOT_PROFILE_FRAMES=<lo>-<hi>: sample only while presents lo..hi-1 are pending
static int64_t world_lo, world_hi;  // OOT_PROFILE_FRAMES=w<lo>-<hi>: only while the world clock is in [lo, hi) s (us)

#define SLOW_RING (1u << 15)        // OOT_PROFILE_SLOW: about 0.5 s of samples of every thread
#define SLOW_DUMPS 64
typedef struct { uint64_t t; DWORD tid; uint32_t depth; uint64_t rip[PROF_DEPTH]; } Sample;
static Sample ring[SLOW_RING];
static uint64_t ring_n;
#define STALL_RING 1024
static Sample stalls[STALL_RING];   // t: when the stop began; rip[PROF_DEPTH - 1] holds the stop's ns
static uint64_t stall_n;
static uint64_t slow_ns;            // OOT_PROFILE_SLOW in ns, 0 off
static volatile LONG64 dump_lo, dump_hi, dump_frame;   // the render thread asks for a window, the sampler writes it
static volatile DWORD replay_tid;   // asynchronous replay (vk_backend.c): its waits hold up the frame too

static void count_rip(uint64_t rip) {
    uint32_t i = (uint32_t)((rip * 0x9E3779B97F4A7C15ull) >> 48);
    for (uint32_t k = 0; k < 64; k++, i = (i + 1) & (PROF_SLOTS - 1)) {
        if (table[i].rip == rip || !table[i].rip) {
            table[i].rip = rip;
            table[i].n++;
            return;
        }
    }
    dropped++;
}

static void count_stack(const uint64_t *rip, uint32_t depth) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint32_t d = 0; d < depth; d++) h = (h ^ rip[d]) * 0x100000001b3ull;
    h |= 1;   // 0 marks a free slot
    uint32_t i = (uint32_t)(h >> 49);
    for (uint32_t k = 0; k < 64; k++, i = (i + 1) & (STK_SLOTS - 1)) {
        StackCount *s = &stacks[i];
        if (!s->hash) {
            s->hash = h, s->depth = depth;
            memcpy(s->rip, rip, depth * sizeof *rip);
        } else if (s->hash != h || s->depth != depth || memcmp(s->rip, rip, depth * sizeof *rip)) {
            continue;
        }
        s->n++;
        return;
    }
    stacks_dropped++;
}

// Unwinds the stack copied to stk_buf from real address lo (n bytes). Registers that point into the copied range are
// moved into the copy, so frame-pointer frames read the copy too. Returns the number of frames.
static uint32_t unwind_copy(CONTEXT *c, uintptr_t lo, size_t n, uint64_t *rip) {
    uintptr_t buf = (uintptr_t)stk_buf, end = buf + n;
    DWORD64 *regs[] = {&c->Rbx, &c->Rbp, &c->Rsi, &c->Rdi, &c->R12, &c->R13, &c->R14, &c->R15};
    for (unsigned i = 0; i < sizeof regs / sizeof *regs; i++)
        if (*regs[i] >= lo && *regs[i] < lo + n) *regs[i] = *regs[i] - lo + buf;
    c->Rsp = buf;
    uint32_t d = 0;
    while (d < PROF_DEPTH && c->Rip && c->Rsp >= buf && c->Rsp + 8 <= end) {
        rip[d++] = c->Rip;
        DWORD64 base;
        PRUNTIME_FUNCTION f = RtlLookupFunctionEntry(c->Rip, &base, NULL);
        if (f) {
            void *hd;
            DWORD64 est;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, c->Rip, f, c, &hd, &est, NULL);
        } else {   // a leaf function: the return address is on top
            c->Rip = *(DWORD64 *)c->Rsp;
            c->Rsp += 8;
        }
    }
    return d;
}

// Re-lists the process's threads: keeps the handle of a thread seen before, opens new ones, closes the gone ones.
static void refresh_threads(DWORD self) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    static ThreadSlot next[PROF_THREADS];
    int n = 0;
    THREADENTRY32 te = {.dwSize = sizeof te};
    DWORD pid = GetCurrentProcessId();
    for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
        if (te.th32OwnerProcessID != pid || te.th32ThreadID == self || n == PROF_THREADS) continue;
        ThreadSlot *old = NULL;
        for (int i = 0; i < n_threads; i++)
            if (threads[i].tid == te.th32ThreadID) old = &threads[i];
        if (old) {
            next[n++] = *old;
            old->h = NULL;
        } else {
            HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID);
            if (h) next[n++] = (ThreadSlot){.tid = te.th32ThreadID, .h = h};
        }
    }
    CloseHandle(snap);
    for (int i = 0; i < n_threads; i++)
        if (threads[i].h) CloseHandle(threads[i].h);   // exited; its counts are lost with the slot
    memcpy(threads, next, sizeof(ThreadSlot) * n);
    n_threads = n;
}

static void write_modules(FILE *f) {
    HMODULE mods[1024];
    DWORD need = 0;
    if (!EnumProcessModules(GetCurrentProcess(), mods, sizeof mods, &need)) return;
    for (DWORD i = 0; i < need / sizeof(HMODULE) && i < 1024; i++) {
        MODULEINFO mi;
        char path[MAX_PATH];
        if (GetModuleInformation(GetCurrentProcess(), mods[i], &mi, sizeof mi) && GetModuleFileNameA(mods[i], path, sizeof path))
            fprintf(f, "module %llx %lx %s\n", (unsigned long long)(uintptr_t)mi.lpBaseOfDll, mi.SizeOfImage, path);
    }
}

// The sampler, between ticks: the samples of the window the render thread asked for.
static void slow_dump(void) {
    uint64_t lo = (uint64_t)dump_lo, hi = (uint64_t)dump_hi, frame = (uint64_t)dump_frame;
    static int dumps;
    char path[1100];
    snprintf(path, sizeof path, "%s.f%llu", out_path, (unsigned long long)frame);
    FILE *f = dumps < SLOW_DUMPS ? fopen(path, "w") : NULL;
    if (f) {
        dumps++;
        fprintf(f, "oot-prof 1\nbase %llx\nperiod_us 1000\nframe %llu interval_ms %.1f\nmain %lu render %lu replay %lu\n",
                (unsigned long long)(uintptr_t)GetModuleHandleW(NULL), (unsigned long long)frame, (hi - lo) / 1e6 - 20,
                main_tid, render_tid, replay_tid);
        write_modules(f);
        uint64_t first = ring_n > SLOW_RING ? ring_n - SLOW_RING : 0;
        for (int pass = 0; pass < 2; pass++)
            for (uint64_t k = first; k < ring_n; k++) {
                const Sample *s = &ring[k & (SLOW_RING - 1)];
                if (s->t < lo || s->t > hi) continue;
                if (pass) fprintf(f, "tstack %lu %.1f", s->tid, (s->t - lo) / 1e6);
                else fprintf(f, "stack 1");
                for (uint32_t d = 0; d < s->depth; d++) fprintf(f, " %llx", (unsigned long long)s->rip[d]);
                fputc('\n', f);
            }
        for (uint64_t k = stall_n > STALL_RING ? stall_n - STALL_RING : 0; k < stall_n; k++) {
            const Sample *s = &stalls[k & (STALL_RING - 1)];
            if (s->t + s->rip[PROF_DEPTH - 1] < lo || s->t > hi) continue;
            fprintf(f, "tstall %lu %.1f %.1f", s->tid, ((double)s->t - (double)lo) / 1e6, s->rip[PROF_DEPTH - 1] / 1e6);
            for (uint32_t d = 0; d < s->depth; d++) fprintf(f, " %llx", (unsigned long long)s->rip[d]);
            fputc('\n', f);
        }
        fclose(f);
        rt_log("prof: slow frame %llu (%.1f ms) -> %s", (unsigned long long)frame, (hi - lo) / 1e6 - 20, path);
    }
    InterlockedExchange64(&dump_hi, 0);
}

static DWORD WINAPI sampler_main(void *arg) {
    (void)arg;
    DWORD self = GetCurrentThreadId();
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    HANDLE timer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    static CONTEXT ctx __attribute__((aligned(16)));
    for (unsigned tick = 0; !stop; tick++) {
        if (timer) {
            LARGE_INTEGER due = {.QuadPart = -10000};   // 1 ms
            SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE);
            WaitForSingleObject(timer, 10);
        } else {
            Sleep(1);
        }
        if (tick % 64 == 0) refresh_threads(self);
        if (dump_hi) slow_dump();
        if (frame_hi && (cur_frame < frame_lo || cur_frame >= frame_hi)) continue;
        if (world_hi) {   // -1 before the world clock starts
            int64_t w = frame_step_world_us();
            if (w < world_lo || w >= world_hi) continue;
        }
        uint64_t now = slow_ns ? kern_now_ns() : 0;
        for (int i = 0; i < n_threads; i++) {
            ThreadSlot *t = &threads[i];
            if ((only_render && t->tid != render_tid) || (only_main && t->tid != main_tid)) continue;
            uint64_t cyc = 0;
            if (slow_ns && t->depth && QueryThreadCycleTime(t->h, &cyc) && cyc == t->cycles) {   // still waiting
                Sample *s = &ring[ring_n++ & (SLOW_RING - 1)];
                s->t = now, s->tid = t->tid, s->depth = t->depth;
                memcpy(s->rip, t->last, t->depth * sizeof *t->last);
                continue;
            }
            ctx.ContextFlags = chains ? CONTEXT_CONTROL | CONTEXT_INTEGER : CONTEXT_CONTROL;
            uint64_t stop0 = slow_ns ? kern_now_ns() : 0;
            if (SuspendThread(t->h) == (DWORD)-1) continue;
            BOOL ok = GetThreadContext(t->h, &ctx);   // returns once the thread has stopped in user mode
            uint64_t stop_ns = slow_ns ? kern_now_ns() - stop0 : 0;
            size_t n = 0;
            MEMORY_BASIC_INFORMATION mb;
            // a thread stopped between its sub rsp and the stack probe has Rsp in the guard page: touching it would fault
            if (ok && chains && VirtualQuery((void *)ctx.Rsp, &mb, sizeof mb) && mb.State == MEM_COMMIT &&
                !(mb.Protect & (PAGE_GUARD | PAGE_NOACCESS))) {
                n = (uintptr_t)mb.BaseAddress + mb.RegionSize - ctx.Rsp;   // the committed stack above Rsp
                size_t cap = slow_ns && (ctx.Rip < exe_lo || ctx.Rip >= exe_hi) ? 8u << 10 : STK_COPY;
                if (n > cap) n = cap;
                memcpy(stk_buf, (void *)ctx.Rsp, n);
            }
            ResumeThread(t->h);
            if (!ok) continue;
            count_rip(ctx.Rip);
            uint64_t chain[PROF_DEPTH] = {ctx.Rip};
            uint32_t d = n ? unwind_copy(&ctx, ctx.Rsp, n, chain) : 0;
            if (d) count_stack(chain, d);
            if (slow_ns) {
                Sample *s = &ring[ring_n++ & (SLOW_RING - 1)];
                s->t = now, s->tid = t->tid, s->depth = d ? d : 1;
                memcpy(s->rip, chain, s->depth * sizeof *chain);
                if (stop_ns > 2000000) {
                    Sample *w = &stalls[stall_n++ & (STALL_RING - 1)];
                    *w = *s;
                    w->t = stop0;
                    if (w->depth > PROF_DEPTH - 1) w->depth = PROF_DEPTH - 1;
                    w->rip[PROF_DEPTH - 1] = stop_ns;
                }
                QueryThreadCycleTime(t->h, &t->cycles);   // after ResumeThread: the cycles the sample itself cost
                t->depth = s->depth;
                memcpy(t->last, chain, s->depth * sizeof *chain);
            }
            total++;
            t->n++;
            if (ctx.Rip >= exe_lo && ctx.Rip < exe_hi) t->in_exe++;
        }
    }
    return 0;
}

static int cmp_desc(const void *a, const void *b) {
    uint64_t x = ((const RipCount *)a)->n, y = ((const RipCount *)b)->n;
    return (x < y) - (x > y);
}

static int cmp_stack_desc(const void *a, const void *b) {
    uint64_t x = ((const StackCount *)a)->n, y = ((const StackCount *)b)->n;
    return (x < y) - (x > y);
}

void prof_write(void) {
    perf_summary_flush();
    if (!out_path || InterlockedExchange(&written, 1)) return;
    InterlockedExchange(&stop, 1);
    if (WaitForSingleObject(sampler, 2000) != WAIT_OBJECT_0) return;   // sampler stuck: no half-written counts
    FILE *f = fopen(out_path, "w");
    if (!f) {
        rt_log("prof: cannot write %s", out_path);
        return;
    }
    fprintf(f, "oot-prof 1\nbase %llx\nperiod_us 1000\nsamples %llu\ndropped %llu\n", (unsigned long long)(uintptr_t)GetModuleHandleW(NULL),
            (unsigned long long)total, (unsigned long long)dropped);
    for (int i = 0; i < n_threads; i++)
        fprintf(f, "thread %lu %llu %llu\n", threads[i].tid, (unsigned long long)threads[i].n, (unsigned long long)threads[i].in_exe);
    write_modules(f);
    int cnt = 0;
    for (uint32_t i = 0; i < PROF_SLOTS; i++)
        if (table[i].rip) table[cnt++] = table[i];
    qsort(table, cnt, sizeof *table, cmp_desc);
    for (int i = 0; i < cnt && i < PROF_TOP; i++) fprintf(f, "rip %llx %llu\n", (unsigned long long)table[i].rip, (unsigned long long)table[i].n);
    int ns = 0;
    for (uint32_t i = 0; i < STK_SLOTS; i++)
        if (stacks[i].hash) stacks[ns++] = stacks[i];
    qsort(stacks, ns, sizeof *stacks, cmp_stack_desc);
    for (int i = 0; i < ns && i < PROF_STACKS; i++) {
        fprintf(f, "stack %llu", (unsigned long long)stacks[i].n);
        for (uint32_t d = 0; d < stacks[i].depth; d++) fprintf(f, " %llx", (unsigned long long)stacks[i].rip[d]);
        fputc('\n', f);
    }
    if (stacks_dropped) fprintf(f, "stacks_dropped %llu\n", (unsigned long long)stacks_dropped);
    fclose(f);
    rt_log("prof: %llu samples, %d Rips -> %s", (unsigned long long)total, cnt, out_path);
}

// OOT_TRACE=perf, OOT_FLIGHT=1 or the full performance overlay: event counts are taken. Wait times are always taken
// (frame-time summary). perf_base: the part fixed at start.
static int perf_on, perf_base;
static int overlay;   // performance overlay mode (perf_set_overlay): 0 off, 1 fps, 2 full

void perf_set_overlay(int mode) {
    __atomic_store_n(&overlay, mode, __ATOMIC_RELAXED);
    __atomic_store_n(&perf_on, perf_base || mode == 2, __ATOMIC_RELAXED);
}

int perf_gpu_timing(void) { return (rt_trace & RT_TRACE_PERF) || __atomic_load_n(&overlay, __ATOMIC_RELAXED) == 2; }

static volatile LONG64 gpu_busy_total;

void perf_gpu_busy(uint64_t ns) { InterlockedAdd64(&gpu_busy_total, (LONG64)ns); }

uint64_t perf_gpu_busy_total(void) { return (uint64_t)__atomic_load_n(&gpu_busy_total, __ATOMIC_RELAXED); }

void prof_start(void) {
    flight_init();
    atexit(perf_summary_flush);   // boot.c's _exit path goes through prof_write instead
    perf_base = (rt_trace & RT_TRACE_PERF) || flight_on;
    perf_on = perf_base || overlay == 2;
    out_path = cfg_get("OOT_PROFILE");
    if (!out_path || !*out_path) {
        out_path = NULL;
        return;
    }
    only_render = cfg_get("OOT_PROFILE_RENDER") != NULL;
    only_main = !only_render && cfg_get("OOT_PROFILE_MAIN") != NULL;
    const char *ch = cfg_get("OOT_PROFILE_CHAINS");
    const char *sl = cfg_get("OOT_PROFILE_SLOW");
    slow_ns = sl && atoi(sl) > 0 ? (uint64_t)atoi(sl) * 1000000ull : 0;
    chains = only_render || only_main || slow_ns || (ch && *ch && strcmp(ch, "0"));
    const char *fr = cfg_get("OOT_PROFILE_FRAMES");
    double wl, wh;
    if (fr && fr[0] == 'w' && sscanf(fr + 1, "%lf-%lf", &wl, &wh) == 2 && wl >= 0 && wl < wh)
        world_lo = (int64_t)(wl * 1e6), world_hi = (int64_t)(wh * 1e6);
    else if (fr && sscanf(fr, "%llu-%llu", (unsigned long long *)&frame_lo, (unsigned long long *)&frame_hi) != 2) frame_hi = 0;
    MODULEINFO mi;
    if (GetModuleInformation(GetCurrentProcess(), GetModuleHandleW(NULL), &mi, sizeof mi)) {
        exe_lo = (uintptr_t)mi.lpBaseOfDll;
        exe_hi = exe_lo + mi.SizeOfImage;
    }
    sampler = CreateThread(NULL, 0, sampler_main, NULL, 0, NULL);
    if (!sampler) {
        rt_log("prof: cannot start the sampler (error %lu)", GetLastError());
        out_path = NULL;
        return;
    }
    atexit(prof_write);   // a --seconds run ends in _exit: boot.c calls prof_write itself first
}

// ---- OOT_TRACE=perf: frame interval and the time the render thread spends waiting, logged every 60 presents. The
// flight recorder (OOT_FLIGHT=1, flight.c) takes the same numbers per frame.
static uint64_t wait_ns[PERF_N], frame_ns[PERF_N], worst_ns[PERF_N];   // window totals, this frame, the slowest frame

uint64_t perf_begin(void) { return kern_now_ns(); }

void perf_replay_thread(void) { replay_tid = GetCurrentThreadId(); }
void perf_main_thread(void) { main_tid = GetCurrentThreadId(); }

void perf_end(int kind, uint64_t t0) {
    if (!t0) return;
    DWORD tid = GetCurrentThreadId();
    if (tid != render_tid && tid != replay_tid) return;
    uint64_t d = kern_now_ns() - t0;
    if (tid == render_tid && !replay_tid) {   // one thread adds: no atomics on the synchronous path
        wait_ns[kind] += d;
        frame_ns[kind] += d;
        return;
    }
    __atomic_fetch_add(&wait_ns[kind], d, __ATOMIC_RELAXED);
    __atomic_fetch_add(&frame_ns[kind], d, __ATOMIC_RELAXED);
}

static volatile LONG64 other_wait_ns;     // perf_gpu_wait on threads other than the render thread, over the window
static volatile LONG64 render_wait_ns;    // the same on the render thread
void (*perf_gpu_stats)(PerfGpu *out);

void perf_gpu_wait(uint64_t t0) {
    if (!t0 || !__atomic_load_n(&perf_on, __ATOMIC_RELAXED)) return;
    LONG64 d = (LONG64)(kern_now_ns() - t0);
    InterlockedAdd64(GetCurrentThreadId() == render_tid ? &render_wait_ns : &other_wait_ns, d);
}

static volatile LONG64 replay_ns;         // perf_replay: summed over the window, any thread
static volatile LONG64 frame_replay_ns;   // the same since the previous present (frame-time summary)

void perf_replay(uint64_t t0) {
    if (!t0) return;
    LONG64 d = (LONG64)(kern_now_ns() - t0);
    InterlockedAdd64(&frame_replay_ns, d);
    if (__atomic_load_n(&perf_on, __ATOMIC_RELAXED)) InterlockedAdd64(&replay_ns, d);
}

static volatile LONG64 counts[PERF_CNT_N];   // running totals: the perf log and the flight recorder take differences

void perf_count(int kind) {
    if (__atomic_load_n(&perf_on, __ATOMIC_RELAXED)) InterlockedIncrement64(&counts[kind]);
}

// ---- Frame-time summary, always on: every 60 s of play and at exit one line of frame count, mean fps, 1% low and the
// frames over 25/50/100 ms, then the worst frames of the period with the render thread's waits in them. Per frame: a
// histogram increment, the compare against the worst list, a memset of the wait array. No allocation.
#define FT_BINS 1024              // 0.25 ms bins up to 256 ms; slower frames land in the last bin
#define FT_WORST 5
#define FT_PERIOD_NS 60000000000ull
typedef struct { uint64_t frame, ns, wait[PERF_N], replay; } FtWorst;
static struct {
    uint64_t t0, frames, sum, over25, over50, over100;
    uint32_t hist[FT_BINS];
    FtWorst worst[FT_WORST];   // slowest first
    int nworst;
} ft;
static uint64_t summary_t0;   // start of the current period (0: not started)

static void ft_report(uint64_t now, const char *tag) {
    if (ft.frames && ft.sum) {
        uint64_t cut = (ft.frames + 99) / 100, acc = 0, low = 0;   // the slowest 1% of frames
        for (int b = FT_BINS - 1; b >= 0; b--) {
            acc += ft.hist[b];
            if (acc >= cut) { low = (uint64_t)(b + 1) * 250000ull; break; }   // upper edge of the bin
        }
        rt_log("frametime: %s: %llu frames in %.1f s, mean %.1f fps, 1%% low %.1f fps, over 25 ms %llu, 50 ms %llu, 100 ms %llu",
               tag, (unsigned long long)ft.frames, (now - ft.t0) / 1e9, ft.frames * 1e9 / ft.sum, low ? 1e9 / low : 0.0,
               (unsigned long long)ft.over25, (unsigned long long)ft.over50, (unsigned long long)ft.over100);
        for (int i = 0; i < ft.nworst; i++) {
            const FtWorst *w = &ft.worst[i];
            rt_log("frametime:   worst frame %llu: %.1f ms; shader translation %.1f, pipeline build %.1f, texture upload %.1f (astc %.1f), replay %.1f, fence/present wait %.1f, pacing %.1f",
                   (unsigned long long)w->frame, w->ns / 1e6, w->wait[PERF_XLATE] / 1e6, w->wait[PERF_PIPE] / 1e6,
                   w->wait[PERF_TEX] / 1e6, w->wait[PERF_ASTC] / 1e6, w->replay / 1e6,
                   (w->wait[PERF_FENCE] + w->wait[PERF_DRAIN]) / 1e6, w->wait[PERF_PACE] / 1e6);
        }
    }
    memset(&ft, 0, sizeof ft);
    ft.t0 = now;
}

__attribute__((weak)) void gfx_exit_log(void) {}

void perf_summary_flush(void) {   // at exit (prof_write)
    if (summary_t0) {
        ft_report(kern_now_ns(), "exit");
        gfx_exit_log();
    }
    summary_t0 = 0;
}

// OOT_FRAME_CSV=<file> (diagnostic, off by default): one line per frame with the interval, the render thread's waits,
// the replay and the texture uploads of the frame. The 60 s frametime lines name only the five worst frames of a period.
static volatile LONG64 csv_tex_new, csv_tex_re, csv_tex_bytes, csv_tex_alloc, csv_tex_swz;
void perf_tex_note(int re, uint64_t bytes, uint64_t alloc_ns, uint64_t swizzle_ns) {
    InterlockedIncrement64(re ? &csv_tex_re : &csv_tex_new);
    InterlockedAdd64(&csv_tex_bytes, (LONG64)bytes);
    InterlockedAdd64(&csv_tex_alloc, (LONG64)alloc_ns);
    InterlockedAdd64(&csv_tex_swz, (LONG64)swizzle_ns);
}
static void csv_frame(uint64_t frame, uint64_t now, uint64_t d, uint64_t rep) {
    static FILE *f;
    static int state;
    static uint64_t t0, gpu_prev;
    if (state < 0) return;
    if (!state) {
        const char *p = getenv("OOT_FRAME_CSV");
        if (!p || !*p || !(f = fopen(p, "w"))) { state = -1; return; }
        state = 1, t0 = now;
        fprintf(f, "frame,t_s,ms,pace,fence,drain,pipe,xlate,tex,astc,replay,gpu,tex_new,tex_re,tex_mib,tex_alloc,tex_swizzle,vram_mib\n");
    }
    static uint64_t vram;
    if (!(frame % 30)) {
        extern __attribute__((weak)) int vk_mem_vram(uint64_t *used, uint64_t *budget, int *exact);
        uint64_t u = 0, b; int ex;
        if (vk_mem_vram && vk_mem_vram(&u, &b, &ex)) vram = u;
    }
    uint64_t gpu = perf_gpu_busy_total();
    LONG64 nw = InterlockedExchange64(&csv_tex_new, 0), re = InterlockedExchange64(&csv_tex_re, 0), by = InterlockedExchange64(&csv_tex_bytes, 0);
    LONG64 al = InterlockedExchange64(&csv_tex_alloc, 0), sw = InterlockedExchange64(&csv_tex_swz, 0);
    fprintf(f, "%llu,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%lld,%lld,%.2f,%.2f,%.2f,%.0f\n", (unsigned long long)frame, (now - t0) / 1e9, d / 1e6,
            frame_ns[PERF_PACE] / 1e6, frame_ns[PERF_FENCE] / 1e6, frame_ns[PERF_DRAIN] / 1e6, frame_ns[PERF_PIPE] / 1e6, frame_ns[PERF_XLATE] / 1e6,
            frame_ns[PERF_TEX] / 1e6, frame_ns[PERF_ASTC] / 1e6, rep / 1e6, (gpu - gpu_prev) / 1e6, (long long)nw, (long long)re, by / 1048576.0, al / 1e6, sw / 1e6, vram / 1048576.0);
    gpu_prev = gpu;
    if (!(frame % 60)) fflush(f);
}

// Returns the replay time since the previous present.
static uint64_t ft_frame(uint64_t frame, uint64_t now, uint64_t d) {
    LONG64 rep = InterlockedExchange64(&frame_replay_ns, 0);
    csv_frame(frame, now, d, (uint64_t)rep);
    if (!summary_t0) {   // the first present starts the period; there is no interval yet
        summary_t0 = now;
        ft.t0 = now;
        return (uint64_t)rep;
    }
    ft.frames++;
    ft.sum += d;
    uint64_t b = d / 250000ull;
    ft.hist[b < FT_BINS ? b : FT_BINS - 1]++;
    if (d > 25000000ull) ft.over25++;
    if (d > 50000000ull) ft.over50++;
    if (d > 100000000ull) ft.over100++;
    if (d > 25000000ull && (ft.nworst < FT_WORST || d > ft.worst[FT_WORST - 1].ns)) {
        int i = ft.nworst < FT_WORST ? ft.nworst++ : FT_WORST - 1;
        while (i > 0 && ft.worst[i - 1].ns < d) ft.worst[i] = ft.worst[i - 1], i--;
        ft.worst[i].frame = frame;
        ft.worst[i].ns = d;
        memcpy(ft.worst[i].wait, frame_ns, sizeof frame_ns);
        ft.worst[i].replay = (uint64_t)rep;
    }
    if (now - summary_t0 >= FT_PERIOD_NS) {
        ft_report(now, "60 s");
        summary_t0 = now;
    }
    return (uint64_t)rep;
}

// The second line of the 60-frame window: GPU busy time (timestamps), the time threads waited for GPU fences, VRAM and
// process memory, and which side holds the frame back. pace: the frame waited for pacing (vsync, the frame limiter) at
// least 15% of the window; gpu: the GPU was busy for 80% of the frame interval or more (without timestamps: threads
// waited for fences for half the interval or more); cpu otherwise.
static void perf_gpu_line(uint64_t frame, double interval_ms, double pace_share, uint64_t n) {
    PerfGpu g;
    memset(&g, 0, sizeof g);
    if (perf_gpu_stats) perf_gpu_stats(&g);
    double rw = InterlockedExchange64(&render_wait_ns, 0) / 1e6 / n, ow = InterlockedExchange64(&other_wait_ns, 0) / 1e6 / n;
    double gpu_ms = g.have_gpu ? g.gpu_ns / 1e6 / n : 0;
    char gs[96], vs[96];
    if (g.have_gpu) snprintf(gs, sizeof gs, "gpu avg %.2f ms (%llu parts timed, %llu not)", gpu_ms,
                             (unsigned long long)g.gpu_parts, (unsigned long long)g.gpu_missed);
    else snprintf(gs, sizeof gs, "gpu avg n/a");
    if (g.have_vram) snprintf(vs, sizeof vs, "vram %.0f of %.0f MiB%s", g.vram_used / 1048576.0, g.vram_budget / 1048576.0,
                              g.vram_exact ? "" : " (own allocations of the heap size)");
    else snprintf(vs, sizeof vs, "vram n/a");
    PROCESS_MEMORY_COUNTERS_EX pm = {.cb = sizeof pm};
    double ws = 0, priv = 0;
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pm, sizeof pm))
        ws = pm.WorkingSetSize / 1048576.0, priv = pm.PrivateUsage / 1048576.0;
    const char *bound = pace_share >= 0.15 ? "pace"
                      : g.have_gpu ? (gpu_ms >= 0.8 * interval_ms ? "gpu" : "cpu")
                      : (rw + ow >= 0.5 * interval_ms ? "gpu" : "cpu");
    rt_log("perf: frame %llu: %s; fence wait per frame: render thread %.2f ms, other threads %.2f ms; %s; ram working set "
           "%.0f MiB, private %.0f MiB; bound: %s",
           (unsigned long long)frame, gs, rw, ow, vs, ws, priv, bound);
}

int rt_ab(void) {
    static int64_t period = -1;   // us of world clock; 0: off
    if (period < 0) {
        const char *e = cfg_get("OOT_AB");
        period = e && *e ? (int64_t)(atof(e) * 1e6) : 0;
    }
    int64_t w = period ? frame_step_world_us() : -1;
    if (w < 0) return 0;   // before the world clock: A
    int64_t k = w / period % 4;
    return k == 1 || k == 2;
}

int rt_knob(const char *name, const char *def) {
    char key[64];
    snprintf(key, sizeof key, "OOT_%s", name);
    const char *e = cfg_get(key);
    if (!e || !*e) e = def;
    return !strcmp(e, "ab") ? 2 : strcmp(e, "0") != 0;
}

// OOT_ONE_PER_CORE=1: the process keeps the first logical processor of each physical core, so busy threads (the
// render thread, the guest cores, the job workers) never share a core's execution units with another; "ab" switches
// it with the A/B blocks. Off by default until measured here; Savage Wind Recomp
// reported +4-10% from the same rule on an 8-core SMT CPU.
static void cpu_placement(void) {
    static int mode = -1, applied = -1;
    static DWORD_PTR all, one;
    if (mode < 0) {
        mode = rt_knob("ONE_PER_CORE", "0");
        DWORD_PTR sys;
        GetProcessAffinityMask(GetCurrentProcess(), &all, &sys);
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *buf = len ? malloc(len) : NULL;
        int cores = 0;
        if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore, buf, &len))
            for (DWORD off = 0; off < len;) {
                SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *e = (void *)((char *)buf + off);
                KAFFINITY m = e->Processor.GroupMask[0].Mask & all;
                if (e->Processor.GroupMask[0].Group == 0 && m) one |= m & -m, cores++;   // the lowest of the core's
                off += e->Size;
            }
        free(buf);
        if (mode && one && one != all)
            rt_log("cpu: %d cores, %d logical processors: one per core%s", cores, __builtin_popcountll(all),
                   mode == 2 ? " in the B blocks" : "");
        else
            mode = 0;
    }
    int on = RT_KNOB_ON(mode);
    if (on == applied) return;
    applied = on;
    SetProcessAffinityMask(GetCurrentProcess(), on ? one : all);
}

void perf_present(uint64_t frame) {
    static uint64_t last, sum, max, n, win_start, ab_b;
    static LONG64 win_prev[PERF_CNT_N];
    render_tid = GetCurrentThreadId();
    cur_frame = frame;
    if (rt_trace & RT_TRACE_SCHED) kern_sched_report(frame);
    uint64_t now = kern_now_ns(), d = last ? now - last : 0;
    if (slow_ns && last && d >= slow_ns && !dump_hi) {
        dump_lo = (LONG64)(now - d - 20000000ull);
        dump_frame = (LONG64)frame;
        InterlockedExchange64(&dump_hi, (LONG64)now);
    }
    uint64_t rep = ft_frame(frame, now, d);   // d is 0 on the first present: it only starts the period
    ab_b += rt_ab();
    cpu_placement();
    if (__atomic_load_n(&overlay, __ATOMIC_RELAXED))   // the overlay's only cost while it is off: this branch
        perfmon_frame(now, d, frame_ns, rep, __atomic_load_n(&counts[PERF_CNT_TEX_UP], __ATOMIC_RELAXED));
    kern_core_diag_frame(frame, d);
    if (flight_on) flight_frame(frame, now, d, frame_ns, counts);
    if (!(rt_trace & RT_TRACE_PERF)) {
        memset(frame_ns, 0, sizeof frame_ns);
        last = now;
        return;
    }
    static uint64_t slow_ns;   // OOT_PERF_SLOW_MS (default 50): frames at least this long get a line of their own
    if (!slow_ns) {
        const char *e = cfg_get("OOT_PERF_SLOW_MS");
        slow_ns = (uint64_t)(e && atoi(e) > 0 ? atoi(e) : 50) * 1000000ull;
    }
    if (last) {
        sum += d;
        n++;
        if (d > max) max = d, memcpy(worst_ns, frame_ns, sizeof worst_ns);
        if (d >= slow_ns)
            rt_log("perf: slow frame %llu: %.1f ms; blocked pipeline %.1f xlate %.1f texture %.1f astc %.1f fence %.1f pacing %.1f",
                   (unsigned long long)frame, d / 1e6, frame_ns[PERF_PIPE] / 1e6, frame_ns[PERF_XLATE] / 1e6,
                   frame_ns[PERF_TEX] / 1e6, frame_ns[PERF_ASTC] / 1e6, frame_ns[PERF_FENCE] / 1e6,
                   frame_ns[PERF_PACE] / 1e6);
    }
    memset(frame_ns, 0, sizeof frame_ns);
    last = now;
    if (!win_start) win_start = now;
    if (frame % 60) return;
    uint64_t span = now - win_start;
    LONG64 win[PERF_CNT_N];
    for (int i = 0; i < PERF_CNT_N; i++) {
        LONG64 v = __atomic_load_n(&counts[i], __ATOMIC_RELAXED);
        win[i] = v - win_prev[i];
        win_prev[i] = v;
    }
    LONG64 async = win[PERF_CNT_PIPE_ASYNC], skips = win[PERF_CNT_PIPE_SKIP], up = win[PERF_CNT_TEX_UP];
    LONG64 reup = win[PERF_CNT_TEX_REUP], sskip = win[PERF_CNT_SHADER_SKIP], waited = win[PERF_CNT_PIPE_WAIT];
    LONG64 oneshot = win[PERF_CNT_ONESHOT_WAIT];
    uint64_t shader_keys, shader_xlated;
    gfx_shader_counts(&shader_keys, &shader_xlated);
    if (n && span)
        rt_log("perf: frame %llu: interval avg %.2f ms max %.2f ms (%.1f fps); render thread blocked: pacing %.0f%% fence %.0f%% drain %.0f%% pipeline build %.0f%% shader xlate %.0f%% texture %.0f%% (astc %.0f%%); pipelines built in background %lld, draws skipped pending %lld, waited %lld, waiting for a shader %lld, one-shot waited %lld; textures uploaded %lld re-uploaded %lld; slowest frame: xlate %.1f ms texture %.1f ms (astc %.1f) pipeline %.1f ms fence %.1f ms; shaders: %llu keys, %llu translated; render busy avg %.2f ms; replay avg %.2f ms; ab %llu/%llu; world %.1f s",
               (unsigned long long)frame, sum / 1e6 / n, max / 1e6, n * 1e9 / sum, 100.0 * wait_ns[PERF_PACE] / span,
               100.0 * wait_ns[PERF_FENCE] / span, 100.0 * wait_ns[PERF_DRAIN] / span,
               100.0 * wait_ns[PERF_PIPE] / span, 100.0 * wait_ns[PERF_XLATE] / span,
               100.0 * wait_ns[PERF_TEX] / span, 100.0 * wait_ns[PERF_ASTC] / span, (long long)async, (long long)skips, (long long)waited, (long long)sskip, (long long)oneshot,
               (long long)up, (long long)reup, worst_ns[PERF_XLATE] / 1e6, worst_ns[PERF_TEX] / 1e6,
               worst_ns[PERF_ASTC] / 1e6, worst_ns[PERF_PIPE] / 1e6, worst_ns[PERF_FENCE] / 1e6,
               (unsigned long long)shader_keys, (unsigned long long)shader_xlated,
               (span > wait_ns[PERF_PACE] ? span - wait_ns[PERF_PACE] : 0) / 1e6 / n,
               InterlockedExchange64(&replay_ns, 0) / 1e6 / n, (unsigned long long)ab_b, (unsigned long long)n, frame_step_world_us() / 1e6);
    if (n && span) perf_gpu_line(frame, sum / 1e6 / n, (double)wait_ns[PERF_PACE] / span, n);
    sum = max = n = ab_b = 0;
    memset(wait_ns, 0, sizeof wait_ns);
    win_start = now;
}
