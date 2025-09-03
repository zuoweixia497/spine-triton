from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import re
import shutil
import subprocess
import functools
import platform
from pathlib import Path
from . import (
    get_triton_shared_opt_path,
    dump_ir_if_needed,
    get_llvm_bin_path,
    get_spine_mlir_opt_path,
    extract_kernel_name,
)

def _ttir_to_ttsharedir(mod, metadata):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        dump_ir_if_needed([src_path], metadata['name'])
        triton_shared_opt_path = get_triton_shared_opt_path()
        subprocess.check_call(
            [
                triton_shared_opt_path,
                src_path,
                "--triton-to-linalg-experimental",
                "-o",
                dst_path
            ])
        dump_ir_if_needed([dst_path], metadata['name'])
        return Path(dst_path).read_text()


def _optimize_ttsharedir(ttsharedir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return ttsharedir


def _ttsharedir_to_llir(ttsharedir: str, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(ttshared_path).write_text(ttsharedir)
        mlir_opt_path = get_llvm_bin_path("mlir-opt")
        # TritonShared-MLIR to LLVM-MLIR
        subprocess.check_call(
            [
                mlir_opt_path,
                ttshared_path,
                "--convert-linalg-to-affine-loops",
                # Note: eliminate-empty-tensors fails when there are multiple func.return ops
                # in a single kernel which are the results of early returns.
                # See python/examples/test_early_return.py for examples.
                # We disable this pass for now since performance on CPU isn't the main
                # focus at the moment.
                # "--eliminate-empty-tensors",
                "--empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=allow-return-allocs-from-loops=true",
                "--lower-affine",
                "--convert-linalg-to-loops",
                "--expand-strided-metadata",
                "--convert-scf-to-cf",
                "--convert-arith-to-llvm",
                "--convert-math-to-llvm",
                "--convert-complex-to-llvm",
                "--convert-vector-to-llvm",
                "--convert-index-to-llvm",
                "--memref-expand",
                "--finalize-memref-to-llvm",
                "--convert-func-to-llvm",
                "--convert-cf-to-llvm",
                # Lowering memrefs creates more affine.apply ops.
                # Lowering these affine ops again creates further arith ops,
                # so we have to run these two passes again here.
                "--lower-affine",
                "--convert-arith-to-llvm",
                # Remove all unrealized casts created
                "--reconcile-unrealized-casts",
                "--mlir-print-debuginfo",
                "-o",
                llmlir_path,
            ]
        )

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = get_llvm_bin_path("mlir-translate")
        subprocess.check_call(
            [mlir_translate_path, llmlir_path, "--mlir-to-llvmir", "-o", llir_path]
        )
        dump_ir_if_needed([llmlir_path, llir_path], metadata['name'])
        return Path(llir_path).read_text()


def _spine_mlir_ttsharedir_to_llir(ttsharedir: str, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        ttshared_path = os.path.join(tmpdir, "ttshared.mlir")
        llmlir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(ttshared_path).write_text(ttsharedir)
        spine_mlir_path = get_spine_mlir_opt_path()
        # TritonShared-MLIR to LLVM-MLIR
        subprocess.check_call(
            [
                spine_mlir_path,
                ttshared_path,
                "--spine-triton-pipeline",
                "-o",
                llmlir_path,
            ]
        )
        dump_ir_if_needed([llmlir_path], metadata['name'])

        llmlir_new_path = llmlir_path
        base_path = os.getenv("TRITON_SHARED_DUMP_PATH", "")
        if base_path:
            llmlir_new_path = os.path.join(tmpdir, "ll_with_debuginfo.mlir")
            subprocess.check_call(
                [
                    spine_mlir_path,
                    os.path.join(base_path, metadata['name'] + "_" + os.path.basename(llmlir_path)),
                    "--ensure-debug-info-scope-on-llvm-func",
                    "-mlir-print-debuginfo",
                    "-o",
                    llmlir_new_path,
                ]
            )

        # LLVM-MLIR to LLVM-IR
        mlir_translate_path = get_llvm_bin_path("mlir-translate")
        subprocess.check_call(
            [
                mlir_translate_path,
                llmlir_new_path,
                "--mlir-to-llvmir",
                "-o",
                llir_path
            ]
        )
        dump_ir_if_needed([llir_path], metadata['name'])
        return Path(llir_path).read_text()


def _optimize_llir(llir: str):
    # We don't apply any optimizations now, but we can add passes if needed.
    return llir


def _llir_to_so(llir: str, metadata):
    cpu_arch = platform.machine()
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        dst_path = os.path.join(tmpdir, "kernel.o")
        Path(src_path).write_text(llir)
        llc_path = get_llvm_bin_path("llc")
        llc_flags = ["-O3", "--float-abi=hard", "--relocation-model=pic"]
        if cpu_arch == "riscv64":
            llc_flags.extend(
                [
                    "--march=riscv64",
                    "--mattr=64bit,a,b,c,d,f,i,m,v,zfh,zicbop,zicbom,zicboz",
                ]
            )

        subprocess.check_call(
            [
                llc_path,
                src_path,
                *llc_flags,
                "-filetype=obj",
                "-o",
                dst_path
            ]
        )
        dump_ir_if_needed([dst_path], metadata['name'])
        import sys
        py_version = sys.version_info
        cpu_arch = platform.machine()
        if platform.system() == "Windows":
            py_include_dir = os.path.join(sys.base_prefix, 'include')
            py_lib_dir = os.path.join(sys.base_prefix, 'libs')
            py_lib = '{name}{major}{minor}.lib'.format(name="python", major=py_version.major, minor=py_version.minor)
        else:
            py_include_dir = os.path.join(sys.base_prefix, 'include', f'python{sys.version_info.major}.{sys.version_info.minor}')
            py_lib_dir = os.path.join(sys.base_prefix, 'lib')
            py_lib = '{name}{major}.{minor}'.format(name="python", major=py_version.major, minor=py_version.minor)
        cpu_backend_path = Path(__file__).resolve().parent
        include_dir = os.path.join(cpu_backend_path, "include")
        so_path = os.path.join(tmpdir, "kernel.so")
        gcc_flags = []
        if cpu_arch == "riscv64":
            gcc_flags.extend(
                [
                "-march=rv64gcv_zfh_zba_zicbop",
                "-mabi=lp64d",
                "-O3"
                ]
            )
        gcc_flags.append("-fopenmp")
        subprocess.check_call([
        "g++", "-std=c++17", *gcc_flags, dst_path,
        f"-I{py_include_dir}", f"-I{include_dir}", f"-L{py_lib_dir}",
        "-shared", f"-l{py_lib}", "-fPIC", "-o", so_path
        ])
        dump_ir_if_needed([so_path], metadata['name'])
        with open(so_path, "rb") as f:
            return f.read()


@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    # Disable FP8 here since this is a sample CPU backend.
    # Target specific backends can eanble it with supported types.
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    sanitize_overflow: bool = True

    def __post_init__(self):
        pass

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    binary_ext = "so"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        args = {"arch": self.target.arch}
        args.update(
            {k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts}
        )
        return CPUOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name,
        )

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `triton_shared.cc`
    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        cache_sizes = get_cache_sizes()
        mod.set_attr("tt.cache_sizes", ir.make_attr(cache_sizes, mod.context))
        tt_pattern = r"tt\.func\s+public\s+@(\w+)\s*\("
        kernel_name = extract_kernel_name(tt_pattern, str(mod))
        metadata['name'] = kernel_name
        return mod

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: _optimize_ttsharedir(
            _ttir_to_ttsharedir(src, metadata)
        )

        spine_mlir_path = get_spine_mlir_opt_path()

        if os.path.isfile(spine_mlir_path):
            stages["llir"] = lambda src, metadata: _optimize_llir(
                _spine_mlir_ttsharedir_to_llir(src, metadata)
            )
        else:
            stages["llir"] = lambda src, metadata: _optimize_llir(
                _ttsharedir_to_llir(src, metadata)
            )

        stages["so"] = lambda src, metadata: _llir_to_so(src, metadata)

    @functools.lru_cache()
    def hash(self):
        return self.target

    # The CPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}


def get_cache_sizes():

    unit_map = {"k": 1024, "m": 1024**2, "g": 1024**3, "": 1}

    cache_cmd = {
        "L1": "lscpu | grep -E 'L1d|Cache|一级数据' | grep -v 'combined' | awk '{print $3, $4, $5, $6}'",
        "L2": "lscpu | grep -E 'L2|Cache|二级数据' | grep -v 'combined' | awk '{print $3, $4, $5, $6}'",
        "L3": "lscpu | grep -E 'L3|Cache|三级数据' | grep -v 'combined' | awk '{print $3, $4, $5, $6}'",
    }

    results = []
    for cache_level in ["L1", "L2", "L3"]:  # Enforce order
        try:
            output = subprocess.check_output(
                cache_cmd[cache_level], shell=True
            ).decode()
        except Exception as e:
            print(f"Command execution failed: {e}")
            results.append(0)
            continue

        match = re.search(r"(\d+)\s*([KMG]?i?B)\s*\((\d+)\s*instances\)", output)
        matchNoinstances = re.search(r"(\d+)\s*([KMG]?i?B)", output)
        if matchNoinstances:
            total_size, unit = matchNoinstances.groups()
            instances = 1
        elif match:
            total_size, unit, instances = match.groups()
        else:
            results.append(0)
            continue

        unit = unit.lower().rstrip("ib")
        bytes_per_instance = (int(total_size) * unit_map[unit]) // int(instances)
        results.append(bytes_per_instance)

    return results  # Format: [L1_size, L2_size, L3_size] in bytes
