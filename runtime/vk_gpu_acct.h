// Written for ocarina-of-time-recomp (MIT, Copyright (c) 2026 Jeff Aigner).
// OOT_GPU_ACCT (vk_gpu_acct.c): a timestamp pair around work in a command buffer. open returns the pair, -1 outside a
// window. Include after vk.h.
#pragma once

enum { GA_PART, GA_MIRROR_UP, GA_MIRROR_DOWN, GA_UPLOAD, GA_PRESENT, GA_PASS, GA_CS, GA_TASK, GA_MESH, GA_COMPACT, GA_XFER };
int vk_ga_on(void);   // a window is open
int vk_ga_open(VkCommandBuffer cb);
void vk_ga_close(VkCommandBuffer cb, int pair, int cat, uint64_t id, uint64_t a, uint64_t b);
void vk_ga_frame(uint64_t frame);
