from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


HERE = Path(__file__).resolve().parent
RANS_INCLUDE = HERE / "third_party" / "ryg_rans"

setup(
    name="hybrid-parallel-rans",
    version="0.1.0",
    ext_modules=[
        Pybind11Extension(
            "_parallel_rans",
            [str(HERE / "parallel_rans.cpp")],
            include_dirs=[str(RANS_INCLUDE)],
            cxx_std=17,
            extra_compile_args=["-O3", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
