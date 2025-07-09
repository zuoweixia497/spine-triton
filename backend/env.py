import hashlib
import tempfile
import sysconfig
import os
import shutil
from ctypes import CDLL, RTLD_GLOBAL


def get_spine_mlir_cc_debug() -> bool:
    debug_or_not = int(os.getenv("SPINE_MLIR_DEBUG_MODE", "0"))
    return debug_or_not == 1


def get_spine_mlir_opt_path() -> str:
    path = os.getenv("SPINE_MLIR_OPT_PATH", "")
    if path == "":
        print("SPINE_MLIR_OPT_PATH is not set.")
    return path


def get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)


def get_triton_shared_opt_path() -> str:
    path = os.getenv("TRITON_SHARED_OPT_PATH", "")
    if path == "":
        raise Exception("TRITON_SHARED_OPT_PATH is not set.")
    return path


def dump_ir_if_needed(files):
    path = os.getenv("TRITON_SHARED_DUMP_PATH", "")
    if not path:
        return
    for f in files:
        shutil.copy(f, os.path.join(path, os.path.basename(f)))


try:
    spine_mlir_opt_path = get_spine_mlir_opt_path()
    if os.path.isfile(spine_mlir_opt_path):
        spine_mlir_lib_dir = os.path.join(
            os.path.dirname(os.path.dirname(spine_mlir_opt_path)), "lib"
        )
        libspeirruntime_path = os.path.join(
            spine_mlir_lib_dir, "libSpeIRRuntimeLibs.so"
        )
        libspeirruntime = CDLL(libspeirruntime_path, mode=RTLD_GLOBAL)
except Exception as e:
    raise ImportError("can not find libspeirruntime. {}".format(e))
