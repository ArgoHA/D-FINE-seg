from pathlib import Path


def remove_empty_txt_files(folder_path):
    # Convert the input string to a Path object
    folder = Path(folder_path)

    # Check if the provided path is a directory
    if not folder.is_dir():
        print("The provided path is not a valid directory.")
        return

    # Iterate through all txt files in the directory
    for file in folder.glob("*.txt"):
        # Check if the file is empty
        if file.stat().st_size == 0:
            print(f"Removing empty file: {file}")
            file.unlink()  # Delete the file


# Example usage:
if __name__ == "__main__":
    folder_path = "/home/argo/Desktop/Projects/Veryfi/detector/data/dataset/labels"
    remove_empty_txt_files(folder_path)
