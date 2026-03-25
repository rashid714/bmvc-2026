# AWS EC2 ONE-PAGE QUICK REFERENCE

## 🚀 Launch Instance (5 min)

**Via AWS Console** (easiest):
1. Go to: https://console.aws.amazon.com/ec2/
2. Click "Launch Instance"
3. Choose AMI: Deep Learning AMI (Ubuntu 22.04)
4. Instance Type: `g4dn.12xlarge` (4 GPUs, ~5-6 hrs training)
5. Storage: 256 GB gp3 volume
6. Security: Allow SSH (port 22)
7. Download key pair (.pem file)

**Via AWS CLI** (automation):
```bash
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g4dn.12xlarge \
  --key-name my-key \
  --security-groups default
```

---

## 🔗 Connect to Instance (1 min)

```bash
chmod 600 my-key.pem
ssh -i my-key.pem ec2-user@YOUR-PUBLIC-IP

# Or Ubuntu AMI
ssh -i my-key.pem ubuntu@YOUR-PUBLIC-IP
```

---

## 📦 Setup Environment (2 min)

**Option 1: Auto-setup (recommended)**
```bash
cd ~ && wget https://raw.githubusercontent.com/YOUR-REPO/setup_aws.py
python3 setup_aws.py
```

**Option 2: Manual setup**
```bash
sudo yum update -y
git clone https://github.com/YOUR-USERNAME/bmvc-2026.git
cd bmvc-2026
pip install -r requirements.txt
```

---

## ✅ Verify Setup (30 sec)

```bash
nvidia-smi                    # Check GPUs detected
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

---

## 🎯 Run Training (5-6 hours)

```bash
cd ~/bmvc-2026

# Smoke test first (1 epoch, 1 min)
make multimodal-smoke

# Full training
torchrun --nproc_per_node=4 \
  scripts/train_multimodal_cloud.py \
  --config configs/multimodal_cloud.json \
  --output-dir checkpoints/results-final
```

## 📁 Add MINE Google Drive Dataset (AWS only)

Use this only on the AWS machine to avoid large local downloads.

```bash
cd ~/bmvc-2026
pip install gdown
mkdir -p data/mine_gdrive

# MINE dataset file id: 1tdmHOwanxZLigt7_0c3M3kKYZ2rjSauQ
gdown --fuzzy "https://drive.google.com/file/d/1tdmHOwanxZLigt7_0c3M3kKYZ2rjSauQ/view" -O /tmp/mine_dataset.zip
unzip -o /tmp/mine_dataset.zip -d data/mine_gdrive

# Optional: ensure explicit path override
export MINE_GDRIVE_ROOT="$PWD/data/mine_gdrive"
```

Then run training as usual. The pipeline now reads source `mine_gdrive` from `data/mine_gdrive` and will skip it safely if not present.

**What to expect:**
- Training starts immediately
- Prints progress every batch
- Total ~5-6 hours (4 epochs × 3 seeds)
- Creates `checkpoints/results-final/summary.json`

---

## 📊 Monitor Progress (Real-time)

**In another SSH window:**
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Watch training log
tail -f ~/bmvc-2026/checkpoints/results-final/training.log

# Check CPU/memory
top
```

---

## 💾 Download Results (5 min)

**On your local machine:**
```bash
scp -r -i my-key.pem ec2-user@YOUR-IP:~/bmvc-2026/checkpoints/results-final ./

# View results
cat results-final/summary.json | python -m json.tool
```

---

## 💰 Cost Breakdown

| Action | Cost |
|--------|------|
| Instance running (per hour) | ~$3.06 |
| Full training (5-6 hours) | ~$15-18 |
| Storage (EBS, per month) | ~$10 |
| **TOTAL** | **~$18 USD** |

---

## 🛑 Stop Instance After Training

```bash
# SSH exit
exit

# Stop instance (keeps EBS, saves ~99% cost)
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Terminate (deletes everything)
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot connect SSH" | Check security group allows port 22 |
| "CUDA not found" | Instance has no GPUs attached |
| "Out of memory" | Reduce batch_size in config |
| "Connection timeout" | Check instance is running: `aws ec2 describe-instances` |

---

## 📞 Support

- **Full guide**: `AWS_DEPLOYMENT_GUIDE.md`
- **Documentation**: `SUBMISSION_GUIDE_FOR_PROFESSOR.md`
- **Python version**: `python3 --version` (should be 3.10+)

---

## 🎓 For Multiple Users

Run each experiment in separate directory:
```bash
mkdir -p /data/results-$(date +%s)
cd /data/results-$(date +%s)
torchrun --nproc_per_node=4 ~/bmvc-2026/scripts/train_multimodal_cloud.py ...
```

---

**Ready?** After landing on instance: `python3 setup_aws.py`
