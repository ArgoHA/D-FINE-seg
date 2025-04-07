"""
Swap old label with new one
"""

from pathlib import Path

from tqdm import tqdm


def swap_labels(labels_path, output_path, label_old, label_new):
    output_path.mkdir(exist_ok=True, parents=True)
    label_old_str = str(label_old)
    label_new_str = str(label_new)
    for label in tqdm(list(labels_path.iterdir()), total=len(list(labels_path.iterdir()))):
        if label.suffix != ".txt" or label.name == "labels.txt":
            continue

        with open(label, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts and parts[0] == label_old_str:
                parts[0] = label_new_str
                new_line = " ".join(parts) + "\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        with open(output_path / label.name, "w") as f:
            f.writelines(new_lines)


def main():
    labels_path = Path("")
    output_path = labels_path.parent / f"{labels_path.name}_swapped"
    label_old = 2222
    label_new = 13

    swap_labels(labels_path, output_path, label_old, label_new)


if __name__ == "__main__":
    main()
