#!/usr/bin/env python3
"""
Upload DCRNN datasets to Hugging Face Hub.
This script creates repositories automatically and uploads the generated datasets.
"""

import os
import sys

from huggingface_hub import HfApi, create_repo


def check_login():
    """Check if user is logged into Hugging Face."""
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']} (@{user['name']})")
        return True
    except Exception:
        print("❌ Not logged into Hugging Face. Please run:")
        print("   huggingface-cli login")
        print("   or set HF_TOKEN environment variable")
        return False


def create_repository(repo_id, dataset_name):
    """Create a new dataset repository on Hugging Face."""
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True,  # Don't fail if repo already exists
        )
        print(f"✅ Repository created/verified: {repo_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to create repository {repo_id}: {e}")
        return False


def upload_dataset(dataset_name, local_path, username="witgaw"):
    """Upload a single dataset to Hugging Face Hub."""
    repo_id = f"{username}/{dataset_name}"

    # Check if local path exists
    if not os.path.exists(local_path):
        print(f"❌ Local path not found: {local_path}")
        return False

    # Check required files
    required_files = ["README.md", "train.parquet", "val.parquet", "test.parquet"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(local_path, file)):
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing required files in {local_path}: {missing_files}")
        return False

    print(f"📂 Uploading {dataset_name} from {local_path}...")

    # Create repository
    if not create_repository(repo_id, dataset_name):
        return False

    try:
        # Upload files
        api = HfApi()
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {dataset_name} traffic prediction dataset with Parquet files",
            ignore_patterns=[
                "*.npz",
                "*.pkl",
                "__pycache__",
                ".DS_Store",
            ],  # Ignore unnecessary files
        )
        print(f"✅ Successfully uploaded {dataset_name}")
        print(f"🔗 View at: https://huggingface.co/datasets/{repo_id}")
        return True

    except Exception as e:
        print(f"❌ Failed to upload {dataset_name}: {e}")
        return False


def main():
    """Main upload function."""
    print("🚀 DCRNN Dataset Upload to Hugging Face")
    print("=" * 50)

    # Check if logged in
    if not check_login():
        sys.exit(1)

    # Define datasets to upload
    datasets = [
        {"name": "METR-LA", "path": "data/hf_datasets/METR-LA"},
        {"name": "PEMS-BAY", "path": "data/hf_datasets/PEMS-BAY"},
    ]

    # Upload each dataset
    success_count = 0
    for dataset in datasets:
        print(f"\n📦 Processing {dataset['name']}...")
        if upload_dataset(dataset["name"], dataset["path"]):
            success_count += 1
        else:
            print(f"⚠️  Skipping {dataset['name']} due to errors")

    # Summary
    print("\n" + "=" * 50)
    print(
        f"📊 Upload Summary: {success_count}/{len(datasets)} datasets uploaded successfully"
    )

    if success_count == len(datasets):
        print("🎉 All datasets uploaded successfully!")
        print("\n📖 Test loading with:")
        for dataset in datasets:
            print("   from datasets import load_dataset")
            print(f"   ds = load_dataset('witgaw/{dataset['name']}')")
    else:
        print("⚠️  Some uploads failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
