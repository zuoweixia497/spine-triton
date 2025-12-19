## spine-triton

spine-triton is forked from [microsoft/triton-shared](https://github.com/microsoft/triton-shared), which is a Shared Middle-Layer for Triton Compilation.

## QuickStart
1. env setup
~~~
pip install PyYAML sympy torch opencv-python pybind11 --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
~~~

2. prebuild whl
~~~
pip install triton --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
~~~

3. mm demo
~~~
# python/examples/test_smt_mm.py
# export SPINE_TRITON_DUMP_PATH=./ir_dumps # for ir dump
python3 python/examples/test_smt_mm.py
~~~

## Build
1. llvm
~~~
# at spine-triton
# pull llvm-project
# llvm-project hash-tag at spine-triton/triton/cmake/llvm-hash.txt
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout a992f29451b9e140424f35ac5e20177db4afbdc0

mkdir -p build-llvm-riscv64
pushd build-llvm-riscv64
cmake -G Ninja ../llvm-project/llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
   -DLLVM_TARGETS_TO_BUILD="RISCV" \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_BUILD_UTILS=ON \
   -DLLVM_ENABLE_LIBEDIT=OFF \
   -DLLVM_INSTALL_UTILS=ON \
   -DLLVM_INCLUDE_TESTS=OFF \
   -DLLVM_BUILD_TESTS=OFF \
   -DLLVM_ENABLE_LLD=ON \
   -DCMAKE_INSTALL_PREFIX=installed \
   -DCMAKE_TOOLCHAIN_FILE=../cmake/linux_riscv64.clang.toolchain.cmake

cmake --build . --target install --parallel 40
popd
~~~

## License

spine-triton is licensed under the [MIT license](/LICENSE).
