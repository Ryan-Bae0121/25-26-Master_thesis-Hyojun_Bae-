#!/usr/bin/env python3

import os
import sys
from huggingface_hub import hf_hub_download, list_repo_files
import torch

def download_omiclip_model():
    """Download OmiCLIP model from Hugging Face"""
    
    # Set up paths
    data_dir = '/home/students/hbae/Loki/data/basic_usage'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Searching for OmiCLIP model on Hugging Face...")
    
    # Try to find OmiCLIP model repositories
    possible_repos = [
        "guangyuwanglab/OmiCLIP",
        "guangyuwanglab/Loki", 
        "guangyuwanglab/omiclip",
        "omiclip/omiclip",
        "loki/omiclip"
    ]
    
    model_downloaded = False
    
    for repo_id in possible_repos:
        try:
            print(f"Checking repository: {repo_id}")
            files = list_repo_files(repo_id)
            print(f"Available files: {files}")
            
            # Look for checkpoint files
            checkpoint_files = [f for f in files if 'checkpoint' in f.lower() and f.endswith('.pt')]
            
            if checkpoint_files:
                print(f"Found checkpoint files: {checkpoint_files}")
                
                # Download the first checkpoint file
                checkpoint_file = checkpoint_files[0]
                print(f"Downloading {checkpoint_file} from {repo_id}...")
                
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=checkpoint_file,
                    local_dir=data_dir,
                    local_dir_use_symlinks=False
                )
                
                # Rename to checkpoint.pt
                final_path = os.path.join(data_dir, 'checkpoint.pt')
                if downloaded_path != final_path:
                    os.rename(downloaded_path, final_path)
                
                print(f"Model downloaded successfully to: {final_path}")
                print(f"File size: {os.path.getsize(final_path) / (1024*1024):.1f} MB")
                
                model_downloaded = True
                break
                
        except Exception as e:
            print(f"Error with repository {repo_id}: {e}")
            continue
    
    if not model_downloaded:
        print("Could not find OmiCLIP model on Hugging Face.")
        print("Trying alternative approach...")
        
        # Try to download from open_clip directly
        try:
            from open_clip import create_model_from_pretrained, get_tokenizer
            
            print("Attempting to download CoCa ViT-L-14 model...")
            
            # This will download the model weights automatically
            model, preprocess = create_model_from_pretrained(
                "coca_ViT-L-14", 
                pretrained="laion2b_s13b_b90k",
                device='cpu'
            )
            
            # Save the model
            checkpoint_path = os.path.join(data_dir, 'checkpoint.pt')
            torch.save(model.state_dict(), checkpoint_path)
            
            print(f"Model downloaded and saved to: {checkpoint_path}")
            print(f"File size: {os.path.getsize(checkpoint_path) / (1024*1024):.1f} MB")
            
            model_downloaded = True
            
        except Exception as e:
            print(f"Error downloading from open_clip: {e}")
    
    return model_downloaded

if __name__ == "__main__":
    success = download_omiclip_model()
    if success:
        print("\n✅ OmiCLIP model download completed successfully!")
    else:
        print("\n❌ Failed to download OmiCLIP model.")
        print("Please check the model repository or try manual download.")
