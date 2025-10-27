#!/usr/bin/env python3
"""
Delete private Hugging Face repositories in bulk.
Use with caution - this will permanently delete repositories!
"""

import sys
import time

from huggingface_hub import HfApi


def list_private_repos(username=None, repo_type="model", include_public=False):
    """List all private (and optionally public) repositories for the authenticated user."""
    try:
        api = HfApi()

        # Get current user if username not provided
        if username is None:
            user_info = api.whoami()
            username = user_info["name"]

        repo_visibility = "public and private" if include_public else "private"
        print(f"🔍 Listing {repo_visibility} {repo_type}s for user: {username}")

        # List repositories using the correct method
        if repo_type == "model":
            repos = api.list_models(author=username)
        elif repo_type == "dataset":
            repos = api.list_datasets(author=username)
        elif repo_type == "space":
            repos = api.list_spaces(author=username)
        else:
            print(f"❌ Unknown repository type: {repo_type}")
            return [], username

        # Convert to list and filter repos based on visibility preference
        repos_list = list(repos)
        if include_public:
            # Include both private and public repos
            filtered_repos = repos_list
        else:
            # Only include private repos
            filtered_repos = [
                repo for repo in repos_list if getattr(repo, "private", False)
            ]

        print(f"Found {len(filtered_repos)} {repo_visibility} {repo_type}s")
        return filtered_repos, username

    except Exception as e:
        print(f"❌ Error listing repositories: {e}")
        return [], None


def preview_repos(repos, repo_type="model", include_public=False):
    """Show a preview of repositories that will be deleted."""
    visibility = "public and private" if include_public else "private"
    print(f"\n📋 {visibility.title()} {repo_type}s that will be DELETED:")
    print("-" * 60)

    for i, repo in enumerate(repos[:10], 1):  # Show first 10
        visibility_icon = "🔓" if not getattr(repo, "private", True) else "🔒"
        print(f"{i:3d}. {visibility_icon} {repo.id}")

    if len(repos) > 10:
        print(f"    ... and {len(repos) - 10} more")

    print("-" * 60)
    print(f"Total: {len(repos)} {visibility} {repo_type}s")


def confirm_deletion(count, repo_type="model", include_public=False):
    """Get user confirmation for deletion."""
    visibility = "PUBLIC AND PRIVATE" if include_public else "PRIVATE"
    print(
        f"\n⚠️  WARNING: This will PERMANENTLY DELETE {count} {visibility} {repo_type}s!"
    )
    if include_public:
        print("🔓 This INCLUDES PUBLIC repositories that others may be using!")
    print("This action cannot be undone.")

    response = input(
        f"\nType 'DELETE {count} {visibility} {repo_type.upper()}S' to confirm: "
    )
    expected = f"DELETE {count} {visibility} {repo_type.upper()}S"

    return response.strip() == expected


def delete_repos(repos, repo_type="model", dry_run=False):
    """Delete repositories with progress tracking."""
    api = HfApi()

    if dry_run:
        print(f"\n🧪 DRY RUN: Would delete {len(repos)} {repo_type}s")
        for repo in repos:
            print(f"   Would delete: {repo.id}")
        return

    print(f"\n🗑️  Deleting {len(repos)} private {repo_type}s...")

    deleted_count = 0
    failed_count = 0
    failed_repos = []

    for i, repo in enumerate(repos, 1):
        try:
            print(f"[{i:3d}/{len(repos)}] Deleting {repo.id}...", end=" ")

            api.delete_repo(repo_id=repo.id, repo_type=repo_type, missing_ok=True)

            print("✅")
            deleted_count += 1

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        except Exception as e:
            print(f"❌ Failed: {e}")
            failed_count += 1
            failed_repos.append(repo.id)

    # Summary
    print("\n📊 Deletion Summary:")
    print(f"✅ Successfully deleted: {deleted_count}")
    print(f"❌ Failed to delete: {failed_count}")

    if failed_repos:
        print("\n❌ Failed repositories:")
        for repo_id in failed_repos:
            print(f"   - {repo_id}")


def main():
    """Main deletion function."""
    print("🧹 Hugging Face Private Repository Cleanup Tool")
    print("=" * 60)

    # Check authentication
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Logged in as: {user_info['name']}")
    except Exception:
        print("❌ Not logged into Hugging Face. Please run:")
        print("   huggingface-cli login")
        sys.exit(1)

    # Choose repository type
    print("\nSelect repository type to delete:")
    print("1. Models (default)")
    print("2. Datasets")
    print("3. Spaces")
    print("4. All types")

    choice = input("Enter choice (1-4, default=1): ").strip()

    repo_types = {
        "1": ["model"],
        "2": ["dataset"],
        "3": ["space"],
        "4": ["model", "dataset", "space"],
        "": ["model"],  # default
    }

    if choice not in repo_types:
        print("❌ Invalid choice")
        sys.exit(1)

    types_to_process = repo_types[choice]

    # Repository visibility options
    print("\nRepository visibility:")
    print("1. Private only (default, safer)")
    print("2. Public and private (DANGEROUS)")

    visibility_choice = input("Enter choice (1-2, default=1): ").strip()
    include_public = visibility_choice == "2"

    if include_public:
        print("⚠️  WARNING: You chose to include PUBLIC repositories!")
        print("   This will delete repositories that others may be using.")
        confirm = input("Are you sure? Type 'YES' to continue: ")
        if confirm != "YES":
            print("❌ Operation cancelled")
            sys.exit(1)

    # Options
    print("\nOptions:")
    print("1. Preview only (safe)")
    print("2. Dry run (show what would be deleted)")
    print("3. Actually delete (DANGEROUS)")

    action = input("Enter choice (1-3, default=1): ").strip()

    dry_run = action == "2"
    preview_only = action in ["1", ""]
    actually_delete = action == "3"

    # Process each repository type
    for repo_type in types_to_process:
        repos, username = list_private_repos(
            repo_type=repo_type, include_public=include_public
        )

        if not repos:
            visibility_text = "public or private" if include_public else "private"
            print(f"No {visibility_text} {repo_type}s found.")
            continue

        preview_repos(repos, repo_type, include_public)

        if preview_only:
            print(f"👀 Preview complete for {repo_type}s")
            continue

        if actually_delete:
            if not confirm_deletion(len(repos), repo_type, include_public):
                print(f"❌ Deletion cancelled for {repo_type}s")
                continue

        delete_repos(repos, repo_type, dry_run=dry_run)

    print("\n🎉 Cleanup complete!")

    if preview_only:
        print("💡 Run again with option 3 to actually delete repositories")
    elif dry_run:
        print("💡 Run again with option 3 to perform actual deletion")


if __name__ == "__main__":
    main()
