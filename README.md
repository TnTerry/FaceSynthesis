# Preparation

Clone the repo.
```
git clone https://github.com/TnTerry/FaceSynthesis.git
cd FaceSynthesis
```

Install the packages required.
```
pip install -r requirements.txt
```

Download the model weights following the IP-Adapter's instruction.
```
# install ip-adapter
pip install git+https://github.com/tencent-ailab/IP-Adapter.git

# download the models
cd IP-Adapter
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
```

# Inference
```
python inference.py
```
