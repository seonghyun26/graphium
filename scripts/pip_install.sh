TORCH_VERSION=2.7.0
CUDA_VERSION=128

echo "pip install  dgl -f https://data.dgl.ai/wheels/cu${CUDA_VERSION}/repo.html"

echo "pip install torch==${TORCH_VERSION} torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}"

echo "pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html"


