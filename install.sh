# sudo apt install -y libcairo2-dev pkg-config python3-dev
# pip3 install -r requirements.txt
# pip3 install gpustat
# pip3 install byted-cruise==0.7.3

cd /mnt/bn/vgfm2/test_dit/geneval/mmdetection
pip install -v -e .
pip3 install -v -e .
pip3 install open-clip-torch
pip3 install clip-benchmark
pip3 install -U openmim
mim install mmcv-full
cd /mnt/bn/vgfm2/test_dit/weijia/code_cycle/