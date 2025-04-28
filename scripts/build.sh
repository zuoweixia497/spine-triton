# !/bin/bash
LLVM_INSTALL_DIR=${1}
BUILD_DIR=build-${2}

echo "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}"

export TRITON_PLUGIN_DIRS=${PWD}

mkdir -p ${TRITON_PLUGIN_DIRS}/${BUILD_DIR}

pushd triton/python

TRITON_BUILD_PROTON=false TRITON_BUILD_WITH_CLANG_LLD=false \
TRITON_BUILD_WITH_CCACHE=false LLVM_ROOT_DIR=${LLVM_INSTALL_DIR} MAX_JOBS=20 \
python3 setup.py install --prefix=${TRITON_PLUGIN_DIRS}/${BUILD_DIR}
popd

rm -rf ${BUILD_DIR}/triton
if [ "${2}" = "x86_64" ]; then
    cp -r ${BUILD_DIR}/lib/python*/site-packages/triton ${BUILD_DIR}/
    rm -rf ${BUILD_DIR}/lib
elif [ "${2}" = "riscv64" ]; then
    cp -r ${BUILD_DIR}/local/lib/python*/dist-packages/triton ${BUILD_DIR}/
    rm -rf ${BUILD_DIR}/local
fi

cp triton/python/build/cmake.linux-${2}*/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt ${BUILD_DIR}