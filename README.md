# healthcare_benchmark



## install
```
conda create -n health python==3.10
conda activate health
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install paddleocr==2.7.3
pip install -r requirements.txt
python -m pip install git+https://github.com/KaiyangZhou/deep-person-reid.git#egg=torchreid --no-build-isolation

```

## Preprocess video
python process_video.py

## Generate database

python tool/video_preprocessing.py

## Generate Summary on GPU

python tool/generate_summary.py

## Generate Summary on Chip

python tool/generate_summary_on_chip.py


# Demonstration

## single video
python single_video_app.py
