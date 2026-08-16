import json
from pathlib import Path

import difPy
import hydra
from tqdm import tqdm


def run(dir_path: str, remove_dups: bool):
    dif = difPy.build(dir_path)
    search = difPy.search(dif)

    # save results to json file:
    with open("dups.json", "w") as f:
        json.dump(search.result, f)

    # read dups
    with open("dups.json", "r") as f:
        data = json.load(f)

    print(f"Found {len(data)} dups")

    if remove_dups and len(data):
        clean_data = data.copy()
        print("Deleting dups...")
        for case_idx, dup_case in tqdm(data.items()):
            for dup in dup_case:
                location = dup[0]
                mse = dup[1]

                if mse < 0.01:
                    Path(location).unlink()

                    if case_idx in clean_data:
                        clean_data.pop(case_idx)

        with open("dups_cleaned.json", "w") as f:
            json.dump(clean_data, f)


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg):
    remove_dups = True
    data_path = Path(cfg.train.data_path) / "images"

    run(str(data_path), remove_dups)


if __name__ == "__main__":
    main()
