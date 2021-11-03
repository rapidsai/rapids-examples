
export git_branch=r21.10
rm -rf python_backend
git clone https://github.com/triton-inference-server/python_backend -b $git_branch
cd python_backend
mkdir build && cd build
cmake -DTRITON_ENABLE_GPU=ON -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make triton-python-backend-stub

