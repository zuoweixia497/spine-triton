#!/bin/bash

set -e
# python3 -m pytest tests/test_blas_ops.py --ref=cpu
# python3 -m pytest tests/test_binary_pointwise_ops.py --ref=cpu
# python3 -m pytest tests/test_norm_ops.py --ref=cpu
# python3 -m pytest tests/test_reduction_ops.py --ref=cpu
# python3 -m pytest tests/test_unary_pointwise_ops.py --ref=cpu
# python3 -m pytest tests/test_distribution_ops.py --ref=cpu
# python3 -m pytest tests/test_special_ops.py --ref=cpu
python3 -m pytest tests/test_v1_ops.py --ref=cpu
python3 -m pytest tests/test_v2_ops.py --ref=cpu
python3 -m pytest tests/test_v3_ops.py --ref=cpu
python3 tests/model_resnet18_test.py