conda env create -f conda/environment.yaml
git clone https://github.com/dptech-corp/Uni-Core.git tmp/Uni-core
cd tmp/Uni-core && pip install --no-build-isolation . && cd ..