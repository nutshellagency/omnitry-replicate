import os
from huggingface_hub import snapshot_download

# Optimize download speeds
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def download_omnitry():
    print("Downloading OmniTry models (this may take a while, ~20+ GB)...")
    
    # We download to /src/weights assuming this is where predict.py expects it
    weights_dir = "/src/weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Example repo name (adjust if the actual weights repo is different)
    REPO_ID = "Kunbyte-AI/OmniTry" 
    
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=weights_dir,
        ignore_patterns=["*.md", ".gitattributes"],
        # allow_patterns=["*.safetensors", "*.json", "*.txt"]
    )
    
    print("Download complete.")

if __name__ == "__main__":
    download_omnitry()
