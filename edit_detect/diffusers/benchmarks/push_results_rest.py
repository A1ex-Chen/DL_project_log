import glob
import sys

import pandas as pd
from huggingface_hub import hf_hub_download, upload_file
from huggingface_hub.utils._errors import EntryNotFoundError


sys.path.append(".")
from utils import BASE_PATH, FINAL_CSV_FILE, GITHUB_SHA, REPO_ID, collate_csv  # noqa: E402








if __name__ == "__main__":
    push_to_hf_dataset()