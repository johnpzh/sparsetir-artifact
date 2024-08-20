set -e # exit the shell script if any command failed.
set -u # exit the shell script if there was any undefined variables.
set -o pipefail # If a command in a pipeline failed, the pipeline would return the non-zero exit status.
set -x # print each command before executing it.

export LOCAL_PATH="/home/thom895/local"
export WORK_ROOT="/home/thom895/CGO25/sparseTIR/sparsetir-artifact"
export TMP_PATH="/home/thom895/CGO25/sparseTIR/sparsetir-artifact/Downloads"

export PATH="${LOCAL_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:/usr/local/cuda/lib64:${LOCAL_PATH}/lib:/home/thom895/.local/lib/python3.10/site-packages/numpy.libs/"

mkdir -p "${LOCAL_PATH}/bin"

export CXXFLAGS="-fPIC -I${LOCAL_PATH}/include"
export CFLAGS="-fPIC -I${LOCAL_PATH}/include"
export CPLUS_INCLUDE_PATH="${LOCAL_PATH}/include"
export C_INCLUDE_PATH="${LOCAL_PATH}/include"

# [docker only] Ubuntu install core

# [if needed] install cmake >=3.24
# cd ${TMP_PATH}
# wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3.tar.gz
# tar zxvf cmake-3.28.3.tar.gz
# cd cmake-3.28.3
# mkdir build
# cd build
# ../bootstrap --prefix=${LOCAL_PATH} -- -DCMAKE_USE_OPENSSL=OFF
# make -j 
# make install

# Install Ninja
# cd ${TMP_PATH}
# wget https://github.com/ninja-build/ninja/archive/refs/tags/v1.11.1.tar.gz -O ninja-1.11.1.tar.gz
# tar zxvf ninja-1.11.1.tar.gz
# cd ninja-1.11.1
# mkdir build
# cd build
# cmake .. -DCMAKE_INSTALL_PREFIX="${LOCAL_PATH}" -DCMAKE_COLOR_DIAGNOSTICS=ON
# make -j
# make install

# # Install NCurses needed by TVM
# cd ${TMP_PATH}
# wget https://invisible-island.net/archives/ncurses/ncurses-6.4.tar.gz
# tar zxvf ncurses-6.4.tar.gz
# cd ncurses-6.4
# ./configure --prefix=${LOCAL_PATH} --enable-ext-colors --enable-sp-funcs --enable-term-driver --enable-shared
# make -j
# make install


# # install python 3.9
# # Install miniconda
# # Link: https://docs.anaconda.com/free/miniconda/
# cd ${TMP_PATH}
# mkdir -p "${LOCAL_PATH}/miniconda3"
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${LOCAL_PATH}/miniconda3/miniconda.sh
# bash ${LOCAL_PATH}/miniconda3/miniconda.sh -b -u -p ${LOCAL_PATH}/miniconda3
# rm -rf ${LOCAL_PATH}/miniconda3/miniconda.sh
# ${LOCAL_PATH}/miniconda3/bin/conda init bash
# ${LOCAL_PATH}/miniconda3/bin/conda init zsh
# # Create virtual environment
# conda create --name sparsetir
# conda activate sparsetir
# conda install python=3.9
# conda install pip
# pip3 install setuptools

# Install python packages
cd ${WORK_ROOT}/install
# bash ubuntu_install_python_package.sh
# install libraries for python package on ubuntu
pip3 install --upgrade \
    attrs \
    cloudpickle \
    cython==0.29.33 \
    decorator \
    mypy \
    orderedset \
    packaging \
    Pillow \
    psutil \
    pytest==7.4.4 \
    tlcpack-sphinx-addon==0.2.1 \
    pytest-profiling \
    pytest-xdist \
    requests \
    Jinja2 \
    synr==0.6.0 \
    junitparser==2.4.2 \
    six \
    tornado \
    pytest-lazy-fixture \
    pandas==1.4.4 \
    ogb \
    rdflib==6.2.0 \
    transformers==4.22.1
# pipe install --upgrade numpy
# Note: Cython 3.0 would not work for TVM.
# pip3 install --upgrade cython==0.29.33
# pip3 install --upgrade pytest==7.4.4
# ogb-1.3.5 has a bug, so ogb-1.3.6 or higher is needed.
# pip3 install --upgrade ogb

# Install PyTorch with CUDA, successed for CUDA 12.2
# Note: 
# I uninstalled all torch* packages at first, then installed the author's torch-1.12.0, then installed Triton.
# pip3 install --upgrade torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
## Test commands
# python3 -c "import torch;print(torch.cuda.nccl.version())"
# python3 -c "import torch; print(torch.__version__)"
# ldconfig -p | grep libnccl
#### Python command to check if PyTorch works with GPU
# >>> import torch
# 
# >>> torch.cuda.is_available()
# True
# 
# >>> torch.cuda.device_count()
# 1
# 
# >>> torch.cuda.current_device()
# 0
# 
# >>> torch.cuda.device(0)
# <torch.cuda.device at 0x7efce0b03be0>
# 
# >>> torch.cuda.get_device_name(0)
# 'GeForce GTX 950M'
#### End

# # Install cuDNN
# # Link: https://developer.nvidia.com/cudnn-downloads
# cd ${TMP_PATH}
# wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz
# tar xvf cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz
# mv cudnn-linux-x86_64-9.0.0.312_cuda12-archive ${LOCAL_PATH}/repos/cudnn-9.0.0.312
# ln -s ${LOCAL_PATH}/repos/cudnn-9.0.0.312/lib/* ${LOCAL_PATH}/lib/
# ln -s ${LOCAL_PATH}/repos/cudnn-9.0.0.312/include/* ${LOCAL_PATH}/include/

# # Install dgl and pytorch geometric
# # cd ${WORK_ROOT}/install
# # bash install_dgl.sh
# # bash install_pyg.sh
# pip3 install  dgl -f https://data.dgl.ai/wheels/cu121/repo.html
# # pip3 install dgl==1.0.1 -f https://data.dgl.ai/wheels/cu121/repo.html
# # pip3 install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
# # pip install  dgl==1.0.1 -f https://data.dgl.ai/wheels/cu117/repo.html
# mkdir -p ~/.dgl/
# echo '{"backend": "pytorch"}' > ~/.dgl/config.json
# python3 -c 'import dgl; print(dgl.__path__)'
# pip3 install torch_geometric
# pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# # Install LLVM 13
# # ref: https://github.com/uwsampl/sparsetir-artifact/blob/main/install/ubuntu2004_install_llvm.sh
# cd ${TMP_PATH}
# git clone --depth 1 --branch llvmorg-13.0.1 https://github.com/llvm/llvm-project.git
# cd llvm-project
# git switch -c v13.0.1
# rm -rf build
# mkdir build
# cd build
# cmake -G Ninja ../llvm \
#     -DLLVM_ENABLE_PROJECTS="clang;lld;mlir;openmp" \
#     -DLLVM_TARGETS_TO_BUILD="AArch64;X86" \
#     -DCMAKE_OSX_ARCHITECTURES="arm64" \
#     -DLLVM_ENABLE_ASSERTIONS=ON \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DLLVM_INSTALL_UTILS=ON \
#     -DCMAKE_INSTALL_PREFIX="${LOCAL_PATH}/repos/llvm-13.0.1" \
#     -DCMAKE_COLOR_DIAGNOSTICS=ON
# ninja && ninja install
# ln -s ${LOCAL_PATH}/repos/llvm-13.0.1/bin/* ${LOCAL_PATH}/bin


# Install SparseTIR
# git clone --recursive https://github.com/uwsampl/SparseTIR.git
# cd sparsetir
# cd ${WORK_ROOT}/3rdparty/SparseTIR
# echo set\(USE_LLVM \"llvm-config --ignore-libllvm --link-static\"\) >> config.cmake
# echo set\(HIDE_PRIVATE_SYMBOLS ON\) >> config.cmake
# echo set\(USE_CUDA ON\) >> config.cmake
# echo set\(USE_CUBLAS ON\) >> config.cmake
# echo set\(USE_CUDNN ON\) >> config.cmake
# # Set cuDNN paths
# echo "set(CUDA_CUDNN_LIBRARY \"${LOCAL_PATH}/repos/cudnn-9.0.0.312/lib/libcudnn.so.9.0.0\")" >> config.cmake
# echo "include_directories(\"${LOCAL_PATH}/repos/cudnn-9.0.0.312/include\")" >> config.cmake
# mkdir -p build
# cd build
# cmake -G Ninja .. -DCMAKE_COLOR_DIAGNOSTICS=ON
# ninja

# cd ../python
# pip3 install -e .


# Install glog
# cd ${WORK_ROOT}/3rdparty/glog
# rm -rf build
# mkdir build
# cd build
# # -DWITH_TLS=OFF is needed
# # ref: https://github.com/google/glog/issues/409#issuecomment-455836857
# cmake -G Ninja .. -DWITH_TLS=OFF -DCMAKE_INSTALL_PREFIX="${LOCAL_PATH}" -DCMAKE_COLOR_DIAGNOSTICS=ON 
# ninja
# ninja install
# # CMAKE_INSTALL_PREFIX
# # CMAKE_FIND_PATH
# cmake -G Ninja .. -DCMAKE_INSTALL_PREFIX="${LOCAL_PATH}" -DCMAKE_COLOR_DIAGNOSTICS=ON

# # Install libsparsehash-dev
# # https://github.com/sparsehash/sparsehash
# cd ${TMP_PATH}
# wget https://github.com/sparsehash/sparsehash/archive/refs/tags/sparsehash-2.0.4.tar.gz
# tar zxvf sparsehash-2.0.4.tar.gz
# cd sparsehash-sparsehash-2.0.4
# ./configure --prefix=${LOCAL_PATH}
# make -j
# make install


# # Install torchsparse, needing sparsehash, pytorch
# # for Sparse Convolution
# cd ${WORK_ROOT}/3rdparty/torchsparse
# pip3 install -e .

# # Install kineto
# cd ${WORK_ROOT}/3rdparty/kineto
# cd libkineto
# rm -rf build
# mkdir build
# cd build
# cmake -G Ninja .. -DCMAKE_INSTALL_PREFIX="${LOCAL_PATH}" -DCMAKE_COLOR_DIAGNOSTICS=ON
# ninja 
# ninja install

# # Install graphiler
# # for RGCN
# cd ${WORK_ROOT}/3rdparty/graphiler
# rm -rf build
# mkdir build
# cd build
# cmake -G Ninja .. -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)');${LOCAL_PATH}" -DCMAKE_COLOR_DIAGNOSTICS=ON
# ninja
# mkdir -p ~/.dgl
# mv libgraphiler.so ~/.dgl/
# cd ..
# pip3 install -e .
# python3 -c "from graphiler import EdgeBatchDummy, NodeBatchDummy, mpdfg_builder, update_all"

# # Install Sputnik
# # for SpMM
# cd ${WORK_ROOT}/3rdparty/sputnik
# rm -rf build
# mkdir build
# cd build
# cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DCUDA_ARCHS="60;70;75;80;86" -DCMAKE_PREFIX_PATH="${LOCAL_PATH}" -DCMAKE_INSTALL_PREFIX="${LOCAL_PATH}" -DCMAKE_COLOR_DIAGNOSTICS=ON
# ninja
# ninja install

# # Install dgSPARSE
# # for SpMM
# cd ${WORK_ROOT}/3rdparty/dgsparse
# rm -rf build
# mkdir build
# cd build
# cmake -G Ninja .. -DCMAKE_INSTALL_PREFIX="${LOCAL_PATH}" -DCMAKE_COLOR_DIAGNOSTICS=ON
# ninja
# ninja install

# # Install Triton
# # for Sparse Transformers
# # Might have errors because of libstdc++ version.
# # strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
# # rm ${LOCAL_PATH}/repos/miniconda3/envs/sparsetir/bin/../lib/libstdc++.so.6
# # ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${LOCAL_PATH}/repos/miniconda3/envs/sparsetir/bin/../lib/libstdc++.so.6
# cd ${WORK_ROOT}/3rdparty/triton
# cd python
# pip3 install -e .
# ## Test command. Even if some tests failed, it would probably be okay.
# python3 -m pytest test/unit


# # Install TACO
# # for SpMM
# cd ${WORK_ROOT}/3rdparty/taco
# ## in CMakeLists.txt, change CUDA_ARCHITECTURES values to native
# cd eval_prepared_gpu
# sed -i 's/"70;75;80;86"/native/g' CMakeLists.txt
# # set_target_properties(taco-sddmm PROPERTIES CUDA_ARCHITECTURES native)
# # set_target_properties(taco-spmm PROPERTIES CUDA_ARCHITECTURES native)
# cd ..
# rm -rf build
# mkdir build
# cd build
# cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release -DCUDA=ON -DCMAKE_INSTALL_PREFIX="${LOCAL_PATH}" -DCMAKE_COLOR_DIAGNOSTICS=ON
# ninja
# ninja install

# Install sparsetir_artifact
cd ${WORK_ROOT}/python
pip3 install -e .


# Datasets
## Test command
# python3 -c "from ogb.nodeproppred import DglNodePropPredDataset"
# cd ${WORK_ROOT}
# bash sparse-conv/download_data.sh && echo "y\ny\n" | python3 spmm/download_data.py && echo "y\n" | python3 rgcn/download_data.py && python3 pruned-bert/download_model.py

# Install TexLive for epstopdf
# Install instructions: https://www.tug.org/texlive/quickinstall.html
# omit the --no-interaction
# Select collections:
# a [X] Essential programs and files
# c [X] TeX auxiliary programs
# g [X] Graphics and font utilities
# cd ${TMP_PATH} # working directory of your choice
# wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz # or curl instead of wget
# zcat < install-tl-unx.tar.gz | tar xf -
# cd install-tl-*
# perl ./install-tl
# # Nees about 500 MB 

# # Ghostscript for epstopdf
# # Link: https://ghostscript.com/releases/gsdnld.html
# cd ${TMP_PATH}
# wget https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs10021/ghostscript-10.02.1.tar.gz
# tar zxvf ghostscript-10.02.1.tar.gz
# cd ghostscript-10.02.1
# ./configure --prefix=${LOCAL_PATH} 
# make -j
# make install

# # GNUPlot
# # Build GNUPlot without QT
# cd ${TMP_PATH}
# wget https://psychz.dl.sourceforge.net/project/gnuplot/gnuplot/5.4.10/gnuplot-5.4.10.tar.gz
# tar zxvf gnuplot-5.4.10.tar.gz
# cd gnuplot-5.4.10
# ./configure --without-qt --prefix=${LOCAL_PATH}
# make -j
# make install

# # Copy gnuplot-palettes to ~/
# cd ${WORK_ROOT}
# cp -r 3rdparty/gnuplot-palettes ~/

# set environment variable FLUSH_L2
export FLUSH_L2=ON