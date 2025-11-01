# Azure GPU VM Setup and Configuration Guide

This document provides a step-by-step guide to creating and configuring an Azure GPU VM (A10) with persistent storage, correct kernel version, NVIDIA GRID driver installation, and a Conda-based ML environment.

---

## 🖥️ Azure VM Creation Notes

While creating your Azure VM, make sure to configure the following:

1. ✅ Enable **Spot Discount** to save cost.
2. 🚫 **Do not select Secure Boot** (it prevents NVIDIA GRID driver installation).
3. 💾 **Add an additional Standard SSD** for persistent data storage.
4. 🚫 **Do not select Accelerated Networking** (can cause driver conflicts).

---

## ⚙️ Kernel Management

Check the installed kernel and downgrade to `6.8` if a higher version (like `6.14`) is installed.

```bash
uname -a
```

### Install the 6.8 kernel (6.11 or later is unsupported)
```bash
sudo apt install linux-image-6.8.0-1015-azure
```

### List installed kernels
```bash
dpkg --list | egrep -i --color 'linux-image|linux-headers|linux-modules' | awk '{ print $2 }'
```

### Remove unwanted (6.14/6.11) kernels
```bash
sudo apt purge linux-headers-6.14.0-1012-azure  linux-image-6.14.0-1012-azure  linux-modules-6.14.0-1012-azure
```

### Verify only 6.8 kernels remain
```bash
dpkg --list | egrep -i --color 'linux-image|linux-headers|linux-modules' | awk '{ print $2 }'
```

Expected output:
```
linux-headers-6.8.0-1015-azure
linux-image-6.8.0-1015-azure
linux-modules-6.8.0-1015-azure
```

### Update GRUB to boot with kernel 6.8
Edit the GRUB config:

```bash
sudo nano /etc/default/grub
```

Modify the following line:
```
GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-1015-azure"
```

Then update and reboot:
```bash
sudo update-grub && sudo update-grub2
sudo reboot
```

---

## 🧩 Install Required Packages and Headers

```bash
uname -r
dpkg -l | grep linux-headers
sudo apt-get install linux-headers-$(uname -r)
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install build-essential ubuntu-desktop -y
```

---

## 🚫 Disable Nouveau Driver

Create and edit a file to blacklist Nouveau:

```bash
sudo nano /etc/modprobe.d/nouveau.conf
```

Add the following lines:

```
blacklist nouveau
blacklist lbm-nouveau
```

Then reboot:
```bash
sudo reboot
```

Verify Nouveau is disabled (should fail):
```bash
sudo systemctl stop lightdm.service
```

---

## 🧱 Install NVIDIA GRID 16.5 Driver

Download and install GRID 16.5 for Azure:

```bash
wget -O NVIDIA-Linux-x86_64-535.161.08-grid-azure.run   "https://download.microsoft.com/download/8/d/a/8da4fb8e-3a9b-4e6a-bc9a-72ff64d7a13c/NVIDIA-Linux-x86_64-535.161.08-grid-azure.run"

chmod +x NVIDIA-Linux-x86_64-535.161.08-grid-azure.run
sudo ./NVIDIA-Linux-x86_64-535.161.08-grid-azure.run
```

Edit the GRID configuration:

```bash
sudo cp /etc/nvidia/gridd.conf.template /etc/nvidia/gridd.conf
sudo nano /etc/nvidia/gridd.conf
```

Add:
```
IgnoreSP=FALSE
EnableUI=FALSE
```

Remove:
```
FeatureType=0
```

Save, exit, and verify:
```bash
nvidia-smi
sudo reboot
```

---

## 💾 Configure Persistent SSD

Identify your attached disks:

```bash
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT
```

Assume `/dev/sdc` is your 512 GB persistent disk.

```bash
sudo parted -s /dev/sdc mklabel gpt
sudo parted -s /dev/sdc mkpart primary ext4 0% 100%
sudo mkfs.ext4 -L datadisk /dev/sdc1
sudo mkdir -p /mnt/ssd
sudo mount /dev/sdc1 /mnt/ssd
sudo chown -R $USER:$USER /mnt/ssd
df -hT /mnt/ssd
```

Expected output:
```
Filesystem     Type  Size  Used Avail Use% Mounted on
/dev/sdc1      ext4  512G   ...  ...    /mnt/ssd
```

Record UUID:
```bash
sudo blkid /dev/sdc1
```

Edit `/etc/fstab` to auto-mount:

```bash
sudo nano /etc/fstab
```

Add the following line (replace `<UUID>`):
```
UUID=<UUID>   /mnt/ssd   ext4   defaults,nofail,discard   0   2
```

Test mount:
```bash
sudo mount -a
df -hT /mnt/ssd
```

---

## 🐍 Install Conda on Persistent SSD

```bash
cd /mnt/ssd
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /mnt/ssd/miniconda3

echo 'export PATH=/mnt/ssd/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
conda --version
```

### Configure Conda to use `/mnt/ssd` for all data and model cahce
```bash
mkdir -p /mnt/ssd/conda/pkgs /mnt/ssd/conda/envs
mkdir -p /mnt/ssd/caches/{huggingface,torch,pip,xdg}
```

Create config files:
```bash
cat > ~/.condarc <<'EOF'
pkgs_dirs:
  - /mnt/ssd/conda/pkgs
envs_dirs:
  - /mnt/ssd/conda/envs
EOF
```

Update `.bashrc` for cache locations:
```bash
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/mnt/ssd/caches/huggingface
export TRANSFORMERS_CACHE=/mnt/ssd/caches/huggingface
export HF_HUB_CACHE=/mnt/ssd/caches/huggingface
export TORCH_HOME=/mnt/ssd/caches/torch
export PIP_CACHE_DIR=/mnt/ssd/caches/pip
export XDG_CACHE_HOME=/mnt/ssd/caches/xdg
EOF
source ~/.bashrc
```

---

## 🧠 Create ML Environment and Install Libraries

```bash
conda create -y -n <your_env> python=3.11
conda activate <your_env>
```

Use pip to avoid MKL / OpenMP conflicts:

```bash
pip install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets accelerate safetensors sentencepiece einops bitsandbytes peft triton tqdm huggingface_hub
pip install pandas numpy scipy ipykernel rich typer fastapi uvicorn
```

---

## ⚡ Test GPU Access

```bash
python - <<'PY'
import torch, platform
print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("Matmul OK:", y.shape)
PY
```

If everything runs successfully, you’re all set! 🎉

---

**Enjoy your Azure GPU environment!**
