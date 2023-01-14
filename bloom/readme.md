# Bloom

Install Bloom in bolt

## Info
- [Understand BLOOM, the Largest Open-Access AI, and Run It on Your Local Computer](https://towardsdatascience.com/run-bloom-the-largest-open-access-ai-model-on-your-desktop-computer-f48e1e2a9a32)

## Install

1. Install rquirements: `pip3 install -r requirements.txt`
2. Install torch for cpu: `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu`
3. Install git lfs: `sudo apt-get install git-lfs`
4. Download pretrained model:
   ```bash
   git lfs install
   export GIT_LFS_SKIP_SMUDGE=1
   git clone https://huggingface.co/bigscience/bloom
   cd bloom
   git lfs fetch origin 2a3d62e
   git lfs checkout