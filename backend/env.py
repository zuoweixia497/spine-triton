import hashlib
import tempfile
import sysconfig
import os
import shutil
import re
from ctypes import CDLL, RTLD_GLOBAL
import triton
import os

SPINE_MLIR_BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def get_spine_mlir_cc_debug() -> bool:
    debug_or_not = int(os.getenv("SPINE_MLIR_DEBUG_MODE", "0"))
    return debug_or_not == 1


def get_spine_mlir_opt_path() -> str:
    path = os.getenv("SPINE_MLIR_OPT_PATH", "")
    if path == "":
        return os.path.join(SPINE_MLIR_BASE_PATH, "bin", "spine-opt")
    return path


def get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        return os.path.join(SPINE_MLIR_BASE_PATH, "bin", bin_name)
    return os.path.join(path, bin_name)


def get_spine_triton_opt_path() -> str:
    spine_triton_opt_path = os.path.join(SPINE_MLIR_BASE_PATH, "bin", "spine-triton-opt")
    if os.path.isfile(spine_triton_opt_path):
        path = spine_triton_opt_path
    else:
        print("Warning: spine-triton-opt not found in the Triton installation path, getting SPINE_TRITON_OPT_PATH environment variable")
        path = os.getenv("SPINE_TRITON_OPT_PATH", "")
        if path == "":
            raise Exception("SPINE_TRITON_OPT_PATH is not set.")
    return path


def dump_ir_if_needed(files, kernel_name=None):
    path = os.getenv("SPINE_TRITON_DUMP_PATH", "")
    if not path:
        return
    for f in files:
        if kernel_name != None:
            shutil.copy(f, os.path.join(path, kernel_name + "_" + os.path.basename(f)))
        else:
            shutil.copy(f, os.path.join(path) + os.path.basename(f))


def extract_kernel_name(pattern, ir):
    matches = re.findall(pattern, ir)
    assert len(matches) == 1
    kernel_name = matches[0]
    return kernel_name


try:
    spine_mlir_opt_path = get_spine_mlir_opt_path()
    if os.path.isfile(spine_mlir_opt_path):
        spine_mlir_lib_dir = os.path.join(SPINE_MLIR_BASE_PATH, "lib")
        libspeirruntime_path = os.path.join(
            spine_mlir_lib_dir, "libSpeIRRuntimeLibs.so"
        )
        libspeirruntime = CDLL(libspeirruntime_path, mode=RTLD_GLOBAL)
except Exception as e:
    raise ImportError("can not find libspeirruntime. {}".format(e))


try:
    triton_path = os.path.dirname(triton.__file__)
    libtritonruntime_path = os.path.join(triton_path, "_C", "libSpineTritonRuntime.so")
    if os.path.isfile(libtritonruntime_path):
        libtritonruntime = CDLL(libtritonruntime_path, mode=RTLD_GLOBAL)
    else:
        spine_triton_opt_path = get_spine_triton_opt_path()
        if os.path.isfile(spine_triton_opt_path):
            spine_triton_lib_dir = os.path.join(
                os.path.dirname(spine_triton_opt_path), "triton/_C"
            )
            libtritonruntime_path = os.path.join(
                spine_triton_lib_dir, "libSpineTritonRuntime.so"
            )
            libtritonruntime = CDLL(libtritonruntime_path, mode=RTLD_GLOBAL)
except Exception as e:
    raise ImportError("can not find libtritonruntime. {}".format(e))
