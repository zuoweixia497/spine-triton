# !/bin/bash
# bash build.sh ${LLVM_INSTALL_DIR} {arch/x86_64/riscv64} {spine-mlir-install-dir}

LLVM_INSTALL_DIR=${1}
BUILD_DIR=build-wheel-${2}
SPINE_MLIR_INSTALL_DIR=${3}

CUR_DIR=${PWD}
VERSION_NUMBER=$(cat VERSION_NUMBER)

echo "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}"

HASH_FILE="triton-src-hash.txt"
COMMIT_ID=$(grep -v '^\s*$\|^\s*#' "$HASH_FILE" | head -n 1 | tr -d '[:space:]')

export TRITON_PLUGIN_DIRS=${PWD}

mkdir -p ${TRITON_PLUGIN_DIRS}/${BUILD_DIR}

pushd triton
git checkout .
ls ${CUR_DIR}/patch/*.patch | xargs -n1 git apply

export SPINE_MLIR_INSTALL_DIR=${SPINE_MLIR_INSTALL_DIR}
export SPINE_TRITON_VERSION_NUMBER=${VERSION_NUMBER}
TRITON_BUILD_PROTON=false TRITON_BUILD_WITH_CLANG_LLD=false TRITON_BUILD_UT=false TRITON_OFFLINE_BUILD=true \
TRITON_BUILD_WITH_CCACHE=false LLVM_ROOT_DIR=${LLVM_INSTALL_DIR} MAX_JOBS=20 \
python3 setup.py bdist_wheel --plat=linux-${2}

cp dist/*.whl ${CUR_DIR}/${BUILD_DIR}/
popd
