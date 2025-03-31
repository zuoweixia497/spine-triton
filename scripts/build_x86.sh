# !/bin/bash
LLVM_INSTALL_DIR=${1}

echo "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}"

export TRITON_PLUGIN_DIRS=${PWD}

pushd triton/python

TRITON_BUILD_PROTON=false TRITON_BUILD_WITH_CLANG_LLD=false \
    TRITON_BUILD_WITH_CCACHE=false LLVM_ROOT_DIR=${LLVM_INSTALL_DIR} MAX_JOBS=20 \
    python3 setup.py build_py

popd

mkdir -p build

pushd build

cp -r ../triton/python/build/lib*/triton ./

cp ../triton/python/build/cmake*/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt ./

cp -r ${TRITON_PLUGIN_DIRS}/backend ./triton/triton_shared

popd