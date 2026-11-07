# From metroid-prime-4-recomp tools/gen_nvn_procs.py @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied, with the NVN2 procs OoT asks for (NVN2_OOT).
# From fe3h-pc-recomp tools/gen_nvn_procs.py @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied unchanged.
# From totk-pc-recomp tools/gen_nvn_procs.py @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), renamed to fe3h; FE3H also loads the two names TotK 1.2.1 does not.
"""Write runtime/nvn_procs.inc: the NVN proc names the game loads.

Source names come from the MacNativeGames checkout in refs/ (MIT, Copyright (c) 2026 MacNativeGames contributors):
NVN/names.txt plus the NVN_EXTRA definitions of totk_nvn_ext.c. TotK 1.2.1 leaves out two of them; FE3H (SDK 6.5.0,\nan older NVN) asks for both: nvnDeviceApplyDeferredFinalizes and nvnQueueAcquireTexture. The file
list format follows d2r-mobile dm/gfx/tools/gen_protos.py (MIT). No game data is read.

Run with .venv: python tools/gen_nvn_procs.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NVN_DIR = ROOT / "refs/MacNativeGames/games/totk/third_party/d2r_gfx/gfx/nvn"
OUT = ROOT / "runtime/nvn_procs.inc"

NOT_LOADED = set()   # TotK 1.2.1: {"nvnDeviceApplyDeferredFinalizes", "nvnQueueAcquireTexture"}
SUBSDK0 = [  # section 0.1: asked by the movie library through nvnDeviceGetProcAddress
    "nvnDeviceGetProcAddress",
    "nvnSyncFinalize",
    "nvnSyncGetNvRmSyncNVX",
    "nvnSyncInitializeFromNvRmSyncNVX",
    "nvnTextureGetNvRmSurfaceNVX",
]
MAIN_COUNT = 536
# NVN2 names OoT's proc loader (main+0x129e290, 1.1.1) asks for and nothing implements yet. Each gets a stub that logs
# its first calls and returns 0 (runtime/nvn_procs.c NVN_STUB), so a run shows the call order instead of stopping at
# the first call through a NULL pointer; a name moves to NVN2_OOT and an implementation once the game is seen to use it.
NVN2_STUBS = [
    "nvnCommandBufferBeginDeviceCommands",
    "nvnCommandBufferBindAccelerationStructure",
    "nvnCommandBufferBindAccelerationStructures",
    "nvnCommandBufferBindAccelerationStructuresDeferred",
    "nvnCommandBufferBindDeviceMessageBuffer",
    "nvnCommandBufferBindDeviceMessageBufferDeferred",
    "nvnCommandBufferBindShadingRateImage",
    "nvnCommandBufferBuildAccelerationStructures",
    "nvnCommandBufferCompactAccelerationStructure",
    "nvnCommandBufferComputeDirtyTileMap",
    "nvnCommandBufferComputeDirtyTileMapEx",
    "nvnCommandBufferConvertCooperativeVectorMatrix",
    "nvnCommandBufferCopyAccelerationStructure",
    "nvnCommandBufferDispatchRays",
    "nvnCommandBufferDispatchRaysIndirect",
    "nvnCommandBufferDrawMeshTasks",
    "nvnCommandBufferDrawMeshTasks3D",
    "nvnCommandBufferDrawMeshTasks3DIndirect",
    "nvnCommandBufferDrawMeshTasksIndirect",
    "nvnCommandBufferEnableExclusiveScissor",
    "nvnCommandBufferEndDeviceCommands",
    "nvnCommandBufferEvaluateDlss",
    "nvnCommandBufferGetAccelerationStructureVisualizationData",
    "nvnCommandBufferInsertDebugMarkerCompute",
    "nvnCommandBufferInsertDebugMarkerGraphics",
    "nvnCommandBufferMultiDrawMeshTasks3DIndirect",
    "nvnCommandBufferMultiDrawMeshTasks3DIndirectCount",
    "nvnCommandBufferMultiDrawMeshTasksIndirect",
    "nvnCommandBufferMultiDrawMeshTasksIndirectCount",
    "nvnCommandBufferPopDebugGroupCompute",
    "nvnCommandBufferPopDebugGroupGraphics",
    "nvnCommandBufferPushComputeConstantsDeferred",
    "nvnCommandBufferPushDebugGroupCompute",
    "nvnCommandBufferPushDebugGroupGraphics",
    "nvnCommandBufferPushGraphicsConstants",
    "nvnCommandBufferPushGraphicsConstantsDeferred",
    "nvnCommandBufferRelocateAccelerationStructure",
    "nvnCommandBufferResetDeviceMessageBuffer",
    "nvnCommandBufferResetDeviceMessageBufferDeferred",
    "nvnCommandBufferSetComputeEnable",
    "nvnCommandBufferSetComputeEnableConditional",
    "nvnCommandBufferSetComputeOverlapMode",
    "nvnCommandBufferSetComputeWorkloadPriority",
    "nvnCommandBufferSetConservativeRasterMode",
    "nvnCommandBufferSetCopyEnable",
    "nvnCommandBufferSetCopyEnableConditional",
    "nvnCommandBufferSetExclusiveScissor",
    "nvnCommandBufferSetExclusiveScissors",
    "nvnCommandBufferSetFloatBlendClamp",
    "nvnCommandBufferSetGeometryProcessingMode",
    "nvnCommandBufferSetGraphicsWorkloadPriority",
    "nvnCommandBufferSetRecordingMode",
    "nvnCommandBufferSetRepresentativeFragmentTest",
    "nvnCommandBufferSetResourceOccupancy",
    "nvnCommandBufferSetShadingRatePaletteMode",
    "nvnCommandBufferShadingRateImagePalette",
    "nvnCommandBufferSubmitDeviceCommands",
    "nvnDeviceCalculateAccelerationStructureMemoryRequirements",
    "nvnDeviceCommandBufferGetHandle",
    "nvnDeviceCommandBufferSetCommandMemorySize",
    "nvnDeviceCommandBufferSetComputeMemorySize",
    "nvnDeviceCommandBufferSetDefaults",
    "nvnDeviceCommandBufferSetDevice",
    "nvnDeviceCommandBufferSetStorage",
    "nvnDeviceCompactAccelerationStructure",
    "nvnDeviceConvertCooperativeVectorMatrix",
    "nvnDeviceCopyAccelerationStructure",
    "nvnDeviceGetCommandHandleMemorySize",
    "nvnDeviceGetCooperativeVectorMatrixSize",
    "nvnDeviceGetUniformBufferHandle",
    "nvnDeviceMessageBufferBeginMessages",
    "nvnDeviceMessageBufferEndMessages",
    "nvnDeviceMessageBufferGetNextMessage",
    "nvnDeviceMessageBufferReset",
    "nvnDeviceMessageBufferSetDefaults",
    "nvnDeviceMessageBufferSetDevice",
    "nvnDeviceMessageBufferSetStorage",
    "nvnDeviceRelocateAccelerationStructure",
    "nvnDirtyTileMapParametersGetConstantColorTable",
    "nvnDirtyTileMapParametersGetDepthStencilMask",
    "nvnDirtyTileMapParametersGetExpandFootprint",
    "nvnDirtyTileMapParametersGetInputs",
    "nvnDirtyTileMapParametersGetOutputs",
    "nvnDirtyTileMapParametersGetTexture",
    "nvnDirtyTileMapParametersSetConstantColorTable",
    "nvnDirtyTileMapParametersSetDefaults",
    "nvnDirtyTileMapParametersSetDepthStencilMask",
    "nvnDirtyTileMapParametersSetExpandFootprint",
    "nvnDirtyTileMapParametersSetInputs",
    "nvnDirtyTileMapParametersSetOutputs",
    "nvnDirtyTileMapParametersSetTexture",
    "nvnDlssNetworkBuilderGetDevice",
    "nvnDlssNetworkBuilderGetFlags",
    "nvnDlssNetworkBuilderGetMaxEvaluationEntries",
    "nvnDlssNetworkBuilderGetMemory",
    "nvnDlssNetworkBuilderGetMemorySize",
    "nvnDlssNetworkBuilderGetMethod",
    "nvnDlssNetworkBuilderGetOptimalSettings",
    "nvnDlssNetworkBuilderGetOutputResolution",
    "nvnDlssNetworkBuilderGetPreset",
    "nvnDlssNetworkBuilderGetStorageSize",
    "nvnDlssNetworkBuilderGetTextureIDs",
    "nvnDlssNetworkBuilderGetTexturePool",
    "nvnDlssNetworkBuilderSetDefaults",
    "nvnDlssNetworkBuilderSetDevice",
    "nvnDlssNetworkBuilderSetFlags",
    "nvnDlssNetworkBuilderSetMaxEvaluationEntries",
    "nvnDlssNetworkBuilderSetMethod",
    "nvnDlssNetworkBuilderSetOutputResolution",
    "nvnDlssNetworkBuilderSetPreset",
    "nvnDlssNetworkBuilderSetStorage",
    "nvnDlssNetworkBuilderSetTextureIDs",
    "nvnDlssNetworkFinalize",
    "nvnDlssNetworkInitialize",
    "nvnDlssNetworkResize",
    "nvnDlssParametersGetBiasCurrentColorSubrectangleBase",
    "nvnDlssParametersGetDepthSubrectangleBase",
    "nvnDlssParametersGetEvaluationIndex",
    "nvnDlssParametersGetHistoryReset",
    "nvnDlssParametersGetInputColorSubrectangleBase",
    "nvnDlssParametersGetInputSubrectangleSize",
    "nvnDlssParametersGetJitterOffset",
    "nvnDlssParametersGetMotionVectorScale",
    "nvnDlssParametersGetMotionVectorSubrectangleBase",
    "nvnDlssParametersGetOutputSubrectangleBase",
    "nvnDlssParametersGetPreExposure",
    "nvnDlssParametersSetBiasCurrentColorMask",
    "nvnDlssParametersSetBiasCurrentColorSubrectangleBase",
    "nvnDlssParametersSetDefaults",
    "nvnDlssParametersSetDepth",
    "nvnDlssParametersSetDepthSubrectangleBase",
    "nvnDlssParametersSetEvaluationIndex",
    "nvnDlssParametersSetExposureTexture",
    "nvnDlssParametersSetHistoryReset",
    "nvnDlssParametersSetInputColor",
    "nvnDlssParametersSetInputColorSubrectangleBase",
    "nvnDlssParametersSetInputSubrectangleSize",
    "nvnDlssParametersSetJitterOffset",
    "nvnDlssParametersSetMotionVectorScale",
    "nvnDlssParametersSetMotionVectorSubrectangleBase",
    "nvnDlssParametersSetMotionVectors",
    "nvnDlssParametersSetOutputColor",
    "nvnDlssParametersSetOutputSubrectangleBase",
    "nvnDlssParametersSetOverlayDebugLayer",
    "nvnDlssParametersSetOverlayFullScreen",
    "nvnDlssParametersSetOverlayJitterDebug",
    "nvnDlssParametersSetOverlayShowNans",
    "nvnDlssParametersSetPreExposure",
    "nvnMultisampleStateGetCoverageTruncationEnable",
    "nvnMultisampleStateGetSampleLocations",
    "nvnMultisampleStateSetCoverageTruncationEnable",
    "nvnProgramBuilderGetMaxCallableRecursionDepth",
    "nvnProgramBuilderGetMaxTraceRecursionDepth",
    "nvnProgramBuilderGetProgramRayTracingFlags",
    "nvnProgramBuilderSetMaxCallableRecursionDepth",
    "nvnProgramBuilderSetMaxTraceRecursionDepth",
    "nvnProgramBuilderSetProgramRayTracingFlags",
    "nvnProgramGetDeviceComputeMemorySize",
    "nvnProgramGetScratchBytesPerWarp",
    "nvnProgramGetShaderGroupHandle",
    "nvnSamplerPoolWriteDescriptorTableData",
    "nvnTexturePoolWriteDescriptorTableData",
]
# NVN2 (SDK 22, Switch 2) procs Ocarina of Time 1.1.1 asks for at its graphics bring-up and the Switch 1 list lacks
# (build/logs/nvn-missing-loaded.txt, 2026-11-05); runtime/nvn_api_nvn2.c and nvn_api_core.c implement them. The
# game names 187 more NVN2 procs it has not asked for yet (docs/plan.md, "NVN2 survey").
NVN2_OOT = [
    "nvnCommandBufferPushComputeConstants",   # the compute kernels' parameter block, read as constant bank 0 (NVN2)
    "nvnDeviceGetFlags", "nvnDeviceRegisterFastClearColorIndex", "nvnDeviceRegisterFastClearColoriIndex",
    "nvnDeviceRegisterFastClearColoruiIndex", "nvnDeviceRegisterFastClearDepthIndex",
    "nvnDeviceRegisterFastClearStencil", "nvnDeviceRegisterFastClearStencilIndex",
    "nvnProgramBuilderGetDevice", "nvnProgramBuilderGetSampleShading", "nvnProgramBuilderSetDebugDataHashes",
    "nvnProgramBuilderSetDefaults", "nvnProgramBuilderSetDevice", "nvnProgramBuilderSetSampleShading",
    "nvnQueueBuilderGetPrimaryQueue", "nvnQueueBuilderGetRayTracingMemorySize", "nvnQueueBuilderSetPrimaryQueue",
    "nvnQueueBuilderSetRayTracingMemorySize", "nvnQueueGetTotalRayTracingMemoryUsed",
    "nvnWindowBuilderGetAcquireTimeout", "nvnWindowBuilderGetColorSpace", "nvnWindowBuilderGetHdrContentMetadata",
    "nvnWindowBuilderGetHdrDisplayMetadata", "nvnWindowBuilderSetAcquireTimeout", "nvnWindowBuilderSetColorSpace",
    "nvnWindowBuilderSetHdrContentMetadata", "nvnWindowBuilderSetHdrDisplayMetadata",
    "nvnWindowGetAcquireTimeout", "nvnWindowGetHdrContentMetadata", "nvnWindowGetHdrDisplayMetadata",
    "nvnWindowSetAcquireTimeout", "nvnWindowSetHdrContentMetadata", "nvnWindowSetHdrDisplayMetadata",
    # the program path, asked for next (nvnProgramInitializeFromBuilder is called through a NULL pointer otherwise)
    "nvnProgramBuilderSetShaders", "nvnProgramBuilderGetShaders", "nvnProgramBuilderGetShaderCount",
    "nvnProgramBuilderSetRayTracingModules", "nvnProgramBuilderGetRayTracingModuleCount",
    "nvnProgramBuilderGetRayTracingModuleData", "nvnProgramBuilderSetSubroutineLinkage",
    "nvnProgramBuilderGetSubroutineLinkage", "nvnProgramBuilderGetSubroutineLinkageCount",
    "nvnProgramBuilderGetDebugDataHashes", "nvnProgramInitializeFromBuilder",
    # the mesh draws (nvn_api_cmd.c): the level's ground is drawn through them
    "nvnCommandBufferDrawMeshTasks", "nvnCommandBufferDrawMeshTasksIndirect",
]


def main() -> int:
    names = [n.strip() for n in (NVN_DIR / "names.txt").read_text(encoding="utf-8").splitlines() if n.strip()]
    extras = re.findall(r"^NVN_EXTRA\((\w+)", (NVN_DIR / "totk_nvn_ext.c").read_text(encoding="utf-8"), re.M)
    if len(extras) != 13:
        print(f"expected 13 NVN_EXTRA names, found {len(extras)}", file=sys.stderr)
        return 1
    main_names = [n for n in names if n not in NOT_LOADED] + extras + NVN2_OOT
    if len(main_names) != len(set(main_names)) or len(main_names) != MAIN_COUNT + len(NVN2_OOT):
        print(f"expected {MAIN_COUNT} distinct main names, found {len(main_names)} "
              f"({len(set(main_names))} distinct)", file=sys.stderr)
        return 1
    # nvnDeviceGetProcAddress is in both lists; keep one row per name.
    extra_sub = [n for n in SUBSDK0 if n not in main_names]
    rows = sorted(main_names) + sorted(extra_sub)
    text = ("// Generated by tools/gen_nvn_procs.py: the NVN proc names FE3H loads (TotK 1.2.1's plus two older ones) and the NVN2 ones OoT asks for.\n"
            + "".join(f"NVN_PROC({n})\n" for n in rows))
    OUT.write_text(text, encoding="utf-8", newline="\n")
    stub_names = [n for n in NVN2_STUBS if n not in main_names]
    (OUT.parent / "nvn_procs_stub.inc").write_text(
        "// Generated by tools/gen_nvn_procs.py: NVN2 procs the game loads that have no implementation yet (NVN2_STUBS).\n"
        + "".join(f"NVN_STUB({n})\n" for n in stub_names), encoding="utf-8", newline="\n")
    print(f"runtime/nvn_procs_stub.inc: {len(stub_names)} stubs")
    print(f"{OUT.relative_to(ROOT)}: {len(main_names)} main names + {len(extra_sub)} subsdk0-only names "
          f"({len(SUBSDK0)} asked by subsdk0), {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
