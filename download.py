import os
import gdown
import argparse

# ── Link Google Drive ─────────────────────────────────────────────────────────
# Sostituisci gli ID con quelli reali dopo aver caricato i file su Drive
FILES = {
    "snapshots_train.npy":  "https://drive.google.com/file/d/1fTE-WZ4OWXyoWke3nlhtU3-CFhWbSxKn/view?usp=sharing",
    #"https://drive.google.com/file/d/16iag1bDzGUrzfbygCK1jyG6-61OmrSfN/view?usp=drive_link"
    "parameters_train.npy": "https://drive.google.com/file/d/13l4vz-LN57J4p-wGFOcgF1CfUjTI8BJ0/view?usp=sharing",
    #"https://drive.google.com/file/d/1DmJAcsbwwDd0SJBxMECJFCA5t8IsRjX0/view?usp=drive_link"
    "snapshots_test.npy":   "https://drive.google.com/file/d/1x5iW64GqTwe4MLjtbbSdcAylv3P3S0r9/view?usp=sharing",
    #"https://drive.google.com/file/d/1TucyKJJYN8Thq7HHiGQmr8i_juzho43A/view?usp=drive_link"
    "parameters_test.npy":  "https://drive.google.com/file/d/1JoXAs-MEmfCs-Xv-dgI1BC55wZaM4Ral/view?usp=sharing",
    #"https://drive.google.com/file/d/1NjYTmP23npkJmTsAi5JmdDZvHcX9z9lz/view?usp=drive_link"
}


def get_drive_id(url_or_id):
    if "drive.google.com" in url_or_id:
        return url_or_id.split("/d/")[1].split("/")[0]
    return url_or_id


def download_data(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)

    for fname, file_id in FILES.items():
        out_path = os.path.join(data_dir, fname)

        if os.path.exists(out_path):
            print(f"  Already exists, skipping: {fname}")
            continue

        print(f"  Downloading {fname}...")
        url = f"https://drive.google.com/uc?id={get_drive_id(file_id)}"
        gdown.download(url, out_path, quiet=False)

    print("\nDone. Files saved in:", data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory where to save the data")
    args = parser.parse_args()

    download_data(args.data_dir)
