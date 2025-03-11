from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

if 'MODAL_IMAGE_ID' in os.environ:
    cutlass_include_dirs = ["/root/cuda/flash_attn_turing/cutlass/include", "/root/cuda/flash_attn_turing/cutlass/tools/util/include"]
elif 'COLAB_GPU' in os.environ:
    cutlass_include_dirs = ["/content/cuda/flash_attn_turing/cutlass/include", "/content/cuda/flash_attn_turing/cutlass/tools/util/include"]
elif 'GOOGLE_CLOUD_PROJECT' in os.environ:
    cutlass_include_dirs = ["/home/stevesiu1013/cuda/flash_attn_turing/cutlass/include", "/home/stevesiu1013/cuda/flash_attn_turing/cutlass/tools/util/include"]

#from cpp_extension.setup import nvcc_flags

#modal
#cutlass_include_dirs = ["/root/cuda/flash_attn_turing/cutlass/include", "/root/cuda/flash_attn_turing/cutlass/tools/util/include"]

# colab
#cutlass_include_dirs = ["/content/cuda/flash_attn_turing/cutlass/include", "/content/cuda/flash_attn_turing/cutlass/tools/util/include"]

# google cloud
# cutlass_include_dirs = ["/home/stevesiu1013/cuda/flash_attn_turing/cutlass/include", "/home/stevesiu1013/cuda/flash_attn_turing/cutlass/tools/util/include"]

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
            sources=["flash_attn_turing.cpp",
                     #"flash_fwd_v0.cu",
                     #"flash_fwd_v1.cu",
                     #"flash_fwd_v2.cu",
                     #"flash_fwd_v3.cu",
                     #"flash_fwd_v4.cu",
                     #"flash_fwd_v5.cu",
                     #"flash_fwd_v6.cu",
                     #"flash_fwd_v7.cu",
                     #"flash_fwd_v8.cu",
                     #"flash_fwd_v9.cu",
                     #"flash_fwd_v11.cu"
                     #"flash_fwd_v12.cu"
                     #"flash_fwd_v13.cu"
                     #"flash_fwd_v14.cu"
                     #"flash_fwd_v15.cu"
                     #"flash_fwd_v16.cu"
                     "flash_fwd_v17.cu"
                     #"flash_fwd_v18.cu"
                     ],
            include_dirs=cutlass_include_dirs,
            extra_compile_args={'nvcc': nvcc_flags}
        )
    ],
    install_requires=["torch"],
    cmdclass={"build_ext": BuildExtension}
)