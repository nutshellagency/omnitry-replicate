# How to Deploy OmniTry to Replicate (Serverless Option B)

Since you are running Windows locally, building a 28GB+ Linux Docker image locally using `cog` is extremely difficult and requires complex WSL2/GPU setups. 

The easiest and cheapest way to deploy OmniTry to Replicate is by using **GitHub Actions**. Replicate provides a system that automatically builds your serverless API directly in their cloud whenever you push to a GitHub repository.

## Step-by-Step Deployment Guide

### 1. Set Up the GitHub Repository
1. Go to your [GitHub Account](https://github.com/) and create a **New, Public Repository**. Let's name it `omnitry-replicate`.
2. Do **not** initialize it with a README, .gitignore, or license.
3. Open your terminal in the new folder we just created for you:
   Path: `C:\Users\Hassan\Apps\Virtual-Try-On\omnitry-replicate`
4. Run the following commands to initialize and push your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for OmniTry Cog deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/omnitry-replicate.git
   git push -u origin main
   ```

### 2. Prepare the Source Code (Important!)
Inside the `predict.py` file in this folder, I have placed placeholder logic because OmniTry's specific `inference.py` API is complex. 
- You will need to customize `predict.py` to match exactly how the `Kunbyte-AI/OmniTry` pipeline executes based on their documentation.

### 3. Connect GitHub to Replicate
1. Go to [Replicate.com](https://replicate.com) and log in.
2. Go to **Dashboard** -> **Models** -> **Create a model**.
3. Name your model (e.g., `omnitry-vton`), set it to Public or Private, and create it.
4. On the model page, click **Settings** -> **GitHub integration**.
5. Connect your GitHub account and select your `omnitry-replicate` repository.
6. Click **Enable integration**.

### 4. Build and Monitor
1. Make a small edit to any file in your repository (like adding a space to this README) and push it to  GitHub:
   ```bash
   git commit -am "Trigger Replicate Build"
   git push
   ```
2. Go to your repository on GitHub and click the **Actions** tab. You will see a workflow running!
3. Replicate is now automatically downloading a Linux machine with GPUs, installing `torch`, `CUDA`, OpenCV, cloning the OmniTry source code, and downloading 28GB of weights.
   *(This step will take 30-45 minutes the first time).*

### 5. Finalize the Backend
Once the GitHub Action finishes successfully, go back to your Replicate Model Page. It will say "Ready"!

You now have a production endpoint. 
Update your `.env` string with your model:
```
# For example, if your replicate username is "nutshellagency"
OMNITRY_MODEL_STRING="nutshellagency/omnitry-vton:version_hash_from_replicate"
```
You can then replace the hardcoded `cuuupid/idm-vton` string in our `modelSelector.ts` to instantly route all Option B traffic to your new API!
