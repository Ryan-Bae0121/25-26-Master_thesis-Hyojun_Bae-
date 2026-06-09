# =====================================
# 🔹 Download Loki pretrained checkpoint from Hugging Face
# =====================================

"""
Goal:
Download the pretrained 'checkpoint.pt' from the Hugging Face repository 
WangGuangyuLab/Loki (≈7.6 GB) using your Hugging Face token, 
so that it can be used for fine-tuning or inference in Loki PredEx / Align / Annotate modules.

Usage:
    # Method 1: Set environment variable (recommended)
    export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    python download_loki_checkpoint.py
    
    # Method 2: Pass as argument
    python download_loki_checkpoint.py --token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    # Method 3: Interactive login (may not work in some environments)
    python download_loki_checkpoint.py --interactive

Requirements:
- Hugging Face access approved (you already have permission)
- Hugging Face token (get from https://huggingface.co/settings/tokens)
"""

from huggingface_hub import login, snapshot_download
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Download Loki pretrained checkpoint from Hugging Face')
    parser.add_argument('--token', type=str, help='Hugging Face token')
    parser.add_argument('--interactive', action='store_true', help='Use interactive login')
    args = parser.parse_args()
    
    print('🔽 Downloading Loki pretrained checkpoint from Hugging Face...')
    print('=' * 80)
    
    # Get token
    hf_token = args.token or os.getenv('HF_TOKEN')
    
    # Try to login
    if hf_token:
        print('🔐 Logging in with HF_TOKEN...')
        login(token=hf_token)
        print('✅ Login successful!')
    elif args.interactive:
        print('📝 Interactive login mode...')
        print('(Find your token at https://huggingface.co/settings/tokens)\n')
        try:
            login()
            print('✅ Login successful!')
        except (EOFError, KeyboardInterrupt) as e:
            print(f'\n⚠️  Interactive login not supported in this environment')
            print('Please use --token argument or set HF_TOKEN environment variable\n')
            return
    else:
        print('⚠️  No HF_TOKEN found. Please provide token:')
        print('\n   Option 1: Set environment variable')
        print('      export HF_TOKEN=\'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\'')
        print('\n   Option 2: Pass as argument')
        print('      python download_loki_checkpoint.py --token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print('\n   Get your token from: https://huggingface.co/settings/tokens\n')
        return
    
    # Download repository
    print('\n📥 Downloading Loki repository (WangGuangyuLab/Loki)...')
    print('   This may take a few minutes (≈7.6 GB)...')
    repo_id = 'WangGuangyuLab/Loki'
    
    try:
        download_dir = snapshot_download(repo_id=repo_id, repo_type='model')
        
        print(f'\n✅ Download complete!')
        print(f'📁 Files saved to: {download_dir}')
        
        # List downloaded files
        from pathlib import Path
        download_path = Path(download_dir)
        print(f'\n📂 Downloaded files:')
        for item in sorted(download_path.iterdir()):
            if item.is_file():
                size_gb = item.stat().st_size / (1024**3)
                print(f'   {item.name}: {size_gb:.2f} GB')
            else:
                print(f'   {item.name}/')
        
        # Find checkpoint.pt
        checkpoint_files = list(download_path.rglob('checkpoint.pt'))
        if checkpoint_files:
            print(f'\n✅ checkpoint.pt found:')
            for ckpt in checkpoint_files:
                size_gb = ckpt.stat().st_size / (1024**3)
                print(f'   {ckpt}')
                print(f'   Size: {size_gb:.2f} GB')
        
        print(f'\n🎉 Loki checkpoint ready for fine-tuning!')
        print(f'\n📝 Next step: Fine-tune Loki with your HNSCC data!')
        print(f'   Checkpoint path: {checkpoint_files[0] if checkpoint_files else download_dir}')
        
    except Exception as e:
        print(f'\n❌ Download failed: {e}')
        import traceback
        traceback.print_exc()
        print('\nPossible issues:')
        print('1. Token does not have access to the repository')
        print('2. Network connection issue')
        print('3. Repository name or path is incorrect')

if __name__ == '__main__':
    main()
