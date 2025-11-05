# !/bin/bash
LLVM_INSTALL_DIR=${1}
BUILD_DIR=build-${2}
CUR_DIR=${PWD}

echo "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}"

HASH_FILE="triton-src-hash.txt"
COMMIT_ID=$(grep -v '^\s*$\|^\s*#' "$HASH_FILE" | head -n 1 | tr -d '[:space:]')

export TRITON_PLUGIN_DIRS=${PWD}

mkdir -p ${TRITON_PLUGIN_DIRS}/${BUILD_DIR}

pushd triton
git checkout .
ls ${CUR_DIR}/patch/*.patch | xargs -n1 git apply

# git checkout ${COMMIT_ID} 可以省略

TRITON_BUILD_PROTON=false TRITON_BUILD_WITH_CLANG_LLD=false TRITON_BUILD_UT=false TRITON_OFFLINE_BUILD=true \
TRITON_BUILD_WITH_CCACHE=false LLVM_ROOT_DIR=${LLVM_INSTALL_DIR} MAX_JOBS=20 \
python3 setup.py install --prefix=${TRITON_PLUGIN_DIRS}/${BUILD_DIR}
popd

rm -rf ${BUILD_DIR}/triton

if ls -d ${BUILD_DIR}/lib/python*/site-packages/triton >/dev/null 2>&1; then
    cp -r ${BUILD_DIR}/lib/python*/site-packages/triton ${BUILD_DIR}/
    rm -rf ${BUILD_DIR}/lib
elif  ls -d ${BUILD_DIR}/local/lib/python*/dist-packages/triton >/dev/null 2>&1; then
    cp -r ${BUILD_DIR}/local/lib/python*/dist-packages/triton ${BUILD_DIR}/
    rm -rf ${BUILD_DIR}/local
else
    echo "Error: Cannot find triton package"
    exit 1
fi

cp triton/build/cmake.linux-${2}*/third_party/spine_triton/tools/spine-triton-opt/spine-triton-opt ${BUILD_DIR}