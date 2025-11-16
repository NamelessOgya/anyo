import subprocess
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def save_git_info(output_path: Path):
    """
    Retrieves the current git commit hash and saves it to a file.
    :param output_path: The path to save the git_info.txt file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    try:
        # Get the current commit hash
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )

        # Get the git status (to check for uncommitted changes)
        git_status = (
            subprocess.check_output(["git", "status", "--porcelain"])
            .strip()
            .decode("utf-8")
        )

        with open(output_path, "w") as f:
            f.write(f"Git Commit Hash: {commit_hash}\n")
            if git_status:
                f.write("Uncommitted changes detected:\n")
                f.write(git_status + "\n")
            else:
                f.write("No uncommitted changes.\n")
        log.info(f"Git information saved to {output_path}")
    except subprocess.CalledProcessError as e:
        log.warning(f"Could not retrieve git information: {e}")
        with open(output_path, "w") as f:
            f.write("Could not retrieve git information.\n")
    except FileNotFoundError:
        log.warning("Git command not found. Is Git installed and in PATH?")
        with open(output_path, "w") as f:
            f.write("Git command not found.\n")


if __name__ == "__main__":
    # Example usage
    test_output_dir = Path("temp_results/metrics")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    save_git_info(test_output_dir / "git_info.txt")

    # Clean up
    import shutil

    shutil.rmtree("temp_results")
