from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
from pathlib import Path
import subprocess

this_dir = os.path.dirname(os.path.abspath(__file__))

if os.path.isdir(".git"):
    subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)

# if 'MODAL_IMAGE_ID' in os.environ:
#     cutlass_include_dirs = ["/root/flash-attention-turing/cutlass/include", "/root/flash-attention-turing/cutlass/tools/util/include"]
# elif 'COLAB_GPU' in os.environ:
#     cutlass_include_dirs = ["/content/flash-attention-turing/cutlass/include", "/content/flash-attention-turing/cutlass/tools/util/include"]
# elif 'GOOGLE_CLOUD_PROJECT' in os.environ:
#     cutlass_include_dirs = ["/home/stevesiu1013/flash-attention-turing/cutlass/include", "/home/stevesiu1013/flash-attention-turing/cutlass/tools/util/include"]

#from cpp_extension.setup import nvcc_flags

#modal
#cutlass_include_dirs = ["/root/flash-attention-turing/cutlass/include", "/root/flash-attention-turing/cutlass/tools/util/include"]

# colab
#cutlass_include_dirs = ["/content/flash-attention-turing/cutlass/include", "/content/flash-attention-turing/cutlass/tools/util/include"]

# google cloud
# cutlass_include_dirs = ["/home/stevesiu1013/flash-attention-turing/cutlass/include", "/home/stevesiu1013/flash-attention-turing/cutlass/tools/util/include"]

nvcc_flags = ["-std=c++17",
              "--expt-relaxed-constexpr",
              "-arch=sm_75",
              "-O3",
              "-lineinfo"]

setup(
    name="flash_attn_turing",
    ext_modules=[
        CUDAExtension(
            name="flash_attn_turing",
            sources=["csrc/flash_attn/flash_api.cpp",
                     "csrc/flash_attn/src/flash_fwd_kernel.cu"
                     ],
            #include_dirs=cutlass_include_dirs,
            include_dirs=[
                Path(this_dir) / "csrc" / "flash_attn",
                Path(this_dir) / "csrc" / "flash_attn" / "src",
                Path(this_dir) / "csrc" / "cutlass" / "include",
                Path(this_dir) / "csrc" / "cutlass" / "tools/util/include"
                ],
            extra_compile_args={'nvcc': nvcc_flags}
        )
    ],
    install_requires=["torch"],
    cmdclass={"build_ext": BuildExtension}
)