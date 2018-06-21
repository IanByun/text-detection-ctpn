#!python
#cython: language_level=3

from Cython.Build import cythonize
import os
from os.path import join as pjoin
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

def find_in_path(name, path):
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    # first check if the CUDA_PATH env variable is in use
    if 'CUDA_PATH' in os.environ:
        home = os.environ['CUDA_PATH']
        nvcc = pjoin(home, 'bin')
    else:
        # otherwise, search the PATH for NVCC
        default_path = pjoin(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path('nvcc', os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDA_PATH')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib', 'x64')}
    for k, v in cudaconfig.items():
    #for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))
    return cudaconfig

CUDA = locate_cuda()

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

#스크립트 원본:  
#https://github.com/rmcgibbo/npcuda-example/blob/master/cython/setup.py
#https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/setup.py

#윈도우 버전: https://github.com/zchrissirhcz/incubator-mxnet/blob/b13d913a797b5c11e42c9355410d5b302048a442/example/rcnn/rcnn/cython/setup_windows_cuda.py

import sys
import os
from distutils.spawn import spawn, find_executable
PATH = os.environ.get('PATH')

class CUDA_build_ext(build_ext):
    """
    Custom build_ext command that compiles CUDA files.
    Note that all extension source files will be processed with this compiler.
    """
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        self.compiler.set_executable('compiler_so', 'nvcc')
        self.compiler.set_executable('linker_so', 'nvcc --shared')
        if hasattr(self.compiler, '_c_extensions'):
            self.compiler._c_extensions.append('.cu')  # needed for Windows
        self.compiler.spawn = self.spawn
        build_ext.build_extensions(self)

    def spawn(self, cmd, search_path=1, verbose=0, dry_run=0):
        """
        Perform any CUDA specific customizations before actually launching
        compile/link etc. commands.
        """
        if (sys.platform == 'darwin' and len(cmd) >= 2 and cmd[0] == 'nvcc' and
                cmd[1] == '--shared' and cmd.count('-arch') > 0):
            # Versions of distutils on OSX earlier than 2.7.9 inject
            # '-arch x86_64' which we need to strip while using nvcc for
            # linking
            while True:
                try:
                    index = cmd.index('-arch')
                    del cmd[index:index+2]
                except ValueError:
                    break
        elif self.compiler.compiler_type == 'msvc':
            # There are several things we need to do to change the commands
            # issued by MSVCCompiler into one that works with nvcc. In the end,
            # it might have been easier to write our own CCompiler class for
            # nvcc, as we're only interested in creating a shared library to
            # load with ctypes, not in creating an importable Python extension.
            # - First, we replace the cl.exe or link.exe call with an nvcc
            #   call. In case we're running Anaconda, we search cl.exe in the
            #   original search path we captured further above -- Anaconda
            #   inserts a MSVC version into PATH that is too old for nvcc.
            cmd[:1] = ['nvcc', '--compiler-bindir',
                       os.path.dirname(find_executable("cl.exe", "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\bin")) #PATH를 쓰면 \bin -> x08in으로 깨지는 버그
                       or cmd[0]]
            # - Secondly, we fix a bunch of command line arguments.
            for idx, c in enumerate(cmd):
                # create .dll instead of .pyd files
                #if '.pyd' in c: cmd[idx] = c = c.replace('.pyd', '.dll')  #20160601, by MrX
                # replace /c by -c
                if c == '/c': cmd[idx] = '-c'
                # replace /DLL by --shared
                elif c == '/DLL': cmd[idx] = '--shared'
                # remove --compiler-options=-fPIC
                elif '-fPIC' in c: del cmd[idx]
                # replace /Tc... by ...
                elif c.startswith('/Tc'): cmd[idx] = c[3:]
                #https://github.com/frerich/clcache/pull/290
                elif '/Tp' in c: cmd[idx] = cmd[idx].replace('/Tp', ' ')
                elif 'ID=2' in c: cmd[idx] = cmd[idx].replace('ID=2', ' ')
                # replace /Fo... by -o ...
                elif c.startswith('/Fo'): cmd[idx:idx+1] = ['-o', c[3:]]
                # replace /LIBPATH:... by -L...
                elif c.startswith('/LIBPATH:'): cmd[idx] = '-L' + c[9:]
                # replace /OUT:... by -o ...
                elif c.startswith('/OUT:'): cmd[idx:idx+1] = ['-o', c[5:]]
                # remove /EXPORT:initlibcudamat or /EXPORT:initlibcudalearn
                elif c.startswith('/EXPORT:'): del cmd[idx]
                # replace cublas.lib by -lcublas
                elif c == 'cublas.lib': cmd[idx] = '-lcublas'
            # - Finally, we pass on all arguments starting with a '/' to the
            #   compiler or linker, and have nvcc handle all other arguments
            if '--shared' in cmd:
                pass_on = '--linker-options='
                # we only need MSVCRT for a .dll, remove CMT if it sneaks in:
                cmd.append('/NODEFAULTLIB:libcmt.lib')
            else:
                pass_on = '--compiler-options='
            cmd = ([c for c in cmd if c[0] != '/'] +
                   [pass_on + ','.join(c for c in cmd if c[0] == '/')])
            print("\ncmd is \n", cmd,"\n")
            # For the future: Apart from the wrongly set PATH by Anaconda, it
            # would suffice to run the following for compilation on Windows:
            # nvcc -c -O -o <file>.obj <file>.cu
            # And the following for linking:
            # nvcc --shared -o <file>.dll <file1>.obj <file2>.obj -lcublas
            # This could be done by a NVCCCompiler class for all platforms.
        spawn(cmd, search_path, verbose, dry_run)

nvcc_compile_args = ['-O', '--ptxas-options=-v', '-arch=compute_61', '-code=sm_61', '-c', '--compiler-options=-fPIC']
nvcc_compile_args = os.environ.get('NVCCFLAGS', '').split() + nvcc_compile_args

ext_modules = [
    Extension(
        "utils.bbox",
        ["bbox.pyx"],
        extra_compile_args={},
        include_dirs = [numpy_include]
    ),
    Extension(
        "utils.cython_nms",
        ["cython_nms.pyx"],
        extra_compile_args={},
        include_dirs = [numpy_include]
    ),
    Extension('utils.gpu_nms',
        ['nms_kernel.cu', 'gpu_nms.pyx'],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[],
        extra_compile_args=nvcc_compile_args,
        include_dirs = [numpy_include, CUDA['include']]
    ),
]
#runtime_library_dirs should be set to [] .
#https://github.com/Unidata/netcdf4-python/issues/460

setup(
    ext_modules=ext_modules,
	cmdclass={'build_ext': CUDA_build_ext},
)
