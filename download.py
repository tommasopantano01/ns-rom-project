import os
import gdown
import argparse

FILES = {
    "snapshots_train.npy":       "https://drive.google.com/file/d/1lD7m7UGnuQMal1JEVLJZap_5twEH3haV/view?usp=sharing",
    "snapshots_train_enriched":  "https://drive.google.com/file/d/1oaVh8KNTfPoFYOPUkTGdhzJvhufEjnRj/view?usp=sharing",
    "parameters_train.npy":      "https://drive.google.com/file/d/1Rf74LJTKBv1C0H9XdkefZMk3RViNqZ-X/view?usp=sharing",
    "parameters_train_enriched": "https://drive.google.com/file/d/1hHB54xZW_CuW4Jdk7swTsoq2FQ5bkHyn/view?usp=sharing",
    "snapshots_test.npy":        "https://drive.google.com/file/d/142IwZF8k5r8paXtUjZ-wZ_7ZLK9Puyqt/view?usp=sharing",
    "parameters_test.npy":       "https://drive.google.com/file/d/1oZ1mwEc1XY0Wwcmi_dVvCb2WIfAu8sof/view?usp=sharing",
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
