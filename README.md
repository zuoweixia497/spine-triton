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

## License

spine-triton is licensed under the [MIT license](/LICENSE).
