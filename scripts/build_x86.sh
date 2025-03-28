# !/bin/bash
LLVM_INSTALL_DIR=${1}

echo "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}"

export TRITON_PLUGIN_DIRS=${PWD}

pushd triton/python

TRITON_BUILD_PROTON=false TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=false LLVM_ROOT_DIR=${LLVM_INSTALL_DIR} \
    python3 setup.py build_py

popd