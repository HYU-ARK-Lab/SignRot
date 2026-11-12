# Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
from setuptools import setup
import torch.utils.cpp_extension as torch_cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import pathlib
setup_dir = os.path.dirname(os.path.realpath(__file__))
HERE = pathlib.Path(__file__).absolute().parent

def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass

def get_cuda_arch_flags():
    # SIGNROT_CUDA_ARCH=86 builds for the local GPU only, which cuts the build
    # from ~4 archs down to 1 while iterating on the kernels.
    archs = os.environ.get('SIGNROT_CUDA_ARCH')
    if archs:
        flags = []
        for arch in archs.replace(',', ' ').split():
            flags += ['-gencode', f'arch=compute_{arch},code=sm_{arch}']
        return flags
    return [
        '-gencode', 'arch=compute_75,code=sm_75',  # Turing
        '-gencode', 'arch=compute_80,code=sm_80',  # Ampere
        '-gencode', 'arch=compute_86,code=sm_86',  # Ampere
        '-gencode', 'arch=compute_90,code=sm_90',  # Hopper
    ]
    
def third_party_cmake():
    import subprocess, sys, shutil
    
    cmake = shutil.which('cmake')
    if cmake is None:
            raise RuntimeError('Cannot find CMake executable.')

    retcode = subprocess.call([cmake, HERE])
    if retcode != 0:
        sys.stderr.write("Error: CMake configuration failed.\n")
        sys.exit(1)

    # install fast hadamard transform
    hadamard_dir = os.path.join(HERE, 'third-party/fast-hadamard-transform')
    pip = shutil.which('pip')
    retcode = subprocess.call([pip, 'install', '-e', hadamard_dir])

if __name__ == '__main__':
    # third_party_cmake()
    remove_unwanted_pytorch_nvcc_flags()
    setup(
        name='signrot',
        ext_modules=[
            CUDAExtension(
                name='signrot._CUDA',
                sources=[
                    'signrot/kernels/bindings.cpp',
                    'signrot/kernels/gemm.cu',
                    'signrot/kernels/gemm_grouped.cu',
                    'signrot/kernels/moe.cu',
                    'signrot/kernels/quant.cu',
                    'signrot/kernels/fused.cu',
                    'signrot/kernels/flashinfer.cu',
                ],
                include_dirs=[
                    os.path.join(setup_dir, 'signrot/kernels/include'),
                    os.path.join(setup_dir, 'third-party/cutlass/include'),
                    os.path.join(setup_dir, 'third-party/cutlass/tools/util/include')
                ],
                extra_compile_args={
                    'cxx': [],
                    'nvcc': get_cuda_arch_flags(),
                }
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
