# ✅ AWS EC2 Step-by-Step for Professor

**Total time: ~10 minutes setup + ~5-6 hours training**

---

## STEP 1: Create AWS Account (if needed)

- [ ] Go to: https://aws.amazon.com/
- [ ] Click "Create AWS Account"
- [ ] Add payment method
- [ ] (Free tier works but is slow; better to use paid for faster training)

---

## STEP 2: Launch EC2 Instance (2 minutes)

1. Go to: https://console.aws.amazon.com/ec2/
2. Click "Instances" (left sidebar)
3. Click "Launch Instance" (orange button)

**Configuration:**

| Field | Value |
|-------|-------|
| **Name** | `BMVC-2026-Training` |
| **AMI** | Deep Learning AMI (Ubuntu 22.04) |
| **Instance Type** | `g4dn.12xlarge` (4 T4 GPUs) |
| **Key Pair** | Create new → name: `bmvc-key` → Download! |
| **Storage** | 256 GB gp3 |
| **Security Group** | SSH port 22 allowed |

4. Click "Launch Instance"
5. ⏳ Wait 2 minutes for instance to start
6. Screenshot the "Public IP" address

---

## STEP 3: Connect to Instance (2 minutes)

**On your local computer terminal:**

```bash
# 1. Make key file readable
chmod 600 ~/Downloads/bmvc-key.pem

# 2. SSH into instance (replace IP with your Public IP from Step 2)
ssh -i ~/Downloads/bmvc-key.pem ec2-user@YOUR-PUBLIC-IP

# 3. Confirm: you should see: ec2-user@ip-172-31-xxx-xxx $
```

- [ ] Successfully SSH'd into EC2

---

## STEP 4: Setup Environment (2 minutes)

**In the SSH terminal, run:**

```bash
cd ~ \
&& wget https://raw.githubusercontent.com/YOUR-GITHUB-USERNAME/bmvc-2026/main/setup_aws.py \
&& python3 setup_aws.py
```

Wait for script to complete. You'll see:
```
✅ AWS Setup Complete!
```

- [ ] Setup script completed

---

## STEP 5: Verify GPUs (30 seconds)

**Still in SSH terminal:**

```bash
nvidia-smi
```

You should see **4 GPUs** (Tesla T4). If not, something is wrong.

- [ ] All 4 GPUs visible

---

## STEP 6: Run Quick Test (1 minute)

**In SSH terminal:**

```bash
cd ~/bmvc-2026 && make aws-smoke
```

Should complete in ~1 minute and show:
```
✅ Smoke test complete!
```

- [ ] Smoke test passed

---

## STEP 7: Start Training (5-6 hours)

**In SSH terminal:**

```bash
cd ~/bmvc-2026 && make aws-train
```

You'll see training progress:
```
Epoch 1 | Batch 0/100 | Loss: 2.1234
Epoch 1 | Batch 10/100 | Loss: 1.9876
...
```

**_Do NOT close this terminal._** Training will continue for 5-6 hours.

- [ ] Training started (you can now minimize terminal)

---

## STEP 8: Monitor Progress (Optional)

**Open ANOTHER terminal** and SSH again:

```bash
ssh -i ~/Downloads/bmvc-key.pem ec2-user@YOUR-PUBLIC-IP

# Watch real-time log
tail -f ~/bmvc-2026/checkpoints/aws-results/training.log
```

---

## STEP 9: Training Complete! Download Results

**When you see this message:**
```
✅ TRAINING COMPLETE - FINAL RESULTS
```

In the ORIGINAL training terminal, **training is done!**

---

## STEP 10: Download Results to Computer (2 minutes)

**On your LOCAL computer terminal (NOT SSH):**

```bash
# Replace YOUR-PUBLIC-IP with the IP from Step 2
scp -r -i ~/Downloads/bmvc-key.pem \
  ec2-user@YOUR-PUBLIC-IP:~/bmvc-2026/checkpoints/aws-results \
  ~/Desktop/bmvc-results/

# View results
cat ~/Desktop/bmvc-results/summary.json
```

Results will look like:
```json
{
  "test_emotion_accuracy_mean": 0.6234,
  "test_emotion_accuracy_std": 0.0145,
  "test_intention_f1_mean": 0.5612,
  ...
}
```

- [ ] Results downloaded

---

## STEP 11: Stop Instance (Save Money!)

**On AWS Console:**

1. Go to: https://console.aws.amazon.com/ec2/instances/
2. Select your instance
3. Click "Instance State" → "Stop Instance"

Cost while stopped: ~$0.07/hour (instead of $3.06/hour)

- [ ] Instance stopped

---

## ✅ YOU'RE DONE!

**What you now have:**
- ✅ Trained multimodal model (4 epochs, 3 seeds)
- ✅ Publication-ready results (mean ± std)
- ✅ Checkpoints saved (`best_model.pt`)
- ✅ Training logs
- ✅ Spending ~$18 total

---

## 📊 Expected Results

After training completes, `summary.json` will show:

```
Emotion Accuracy:        60-65% ± 1-2%
Intention F1 Score:      55-60% ± 0-1%
Action F1 Score:         40-45% ± 1-2%
```

---

## ❌ Troubleshooting

| Problem | Fix |
|---------|-----|
| Can't SSH | Check key file permissions: `chmod 600 bmvc-key.pem` |
| No GPUs showing | Wrong instance type selected. Use `g4dn.12xlarge` |
| Out of memory error | Reduce batch size in `configs/multimodal_cloud.json` |
| "Module not found" | Run `python3 setup_aws.py` again |
| Slow training | Check `nvidia-smi` shows 4 GPUs using 100% |

---

## 🛑 When Finished

1. Download all results to your computer
2. Stop the instance (don't terminate unless you're certain)
3. Review results in `summary.json`
4. Do NOT leave instance running!

---

## 📞 Need Help?

- AWS Setup Guide: `AWS_DEPLOYMENT_GUIDE.md`
- Quick Reference: `AWS_QUICK_REFERENCE.md`
- Technical Details: `SUBMISSION_GUIDE_FOR_PROFESSOR.md`

**You're all set!** Ready for Step 1? → Start at **STEP 1**.
