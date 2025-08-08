#!/bin/bash

python3 -m pytest examples/test_blas_ops.py

if [ $? -eq 0 ]; then
    echo "✅ 所有测试通过！结果正确。"
    exit 0
else
    echo "❌ 测试失败！某些结果不正确。"
    exit 1
fi