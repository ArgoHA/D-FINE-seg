import multiprocessing as mp
from pathlib import Path

import hydra
import pypdfium2 as pdfium
from loguru import logger
from omegaconf import DictConfig
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from tqdm import tqdm


def convert_pdf_to_jpg(pdf_path: Path, dpi: int = 200) -> None:
    """
    Convert pdf to .jpg format with specified DPI (default: 200).
    """
    pages = pdfium.PdfDocument(pdf_path)
    for i, page in enumerate(pages):
        pil_image = page.render(scale=dpi / 72.0).to_pil().convert("RGB")
        output_path = pdf_path.with_name(f"{pdf_path.stem}_{i}.jpg")
        pil_image.save(output_path)

    pdf_path.unlink()


def convert_image_to_jpg(filepath: Path) -> None:
    """
    Convert a single image file to .jpg format
    """
    if filepath.suffix.lower() in [".tif", ".jpeg", ".png", ".tiff", ".heic"]:
        try:
            image = Image.open(filepath).convert("RGB")
            image = ImageOps.exif_transpose(image)  # fix rotation
        except OSError:
            print("Can't open, deleting:", filepath.name)
            # filepath.unlink()
            return

        # if img_name.jpg and img_name.png (or other) exists, rename them
        f_idx = 1
        original_filepath = filepath
        while filepath.with_suffix(".jpg").exists() and f_idx <= 100:
            if f_idx > 1:
                filepath = (
                    filepath.parent / f"{str(filepath.stem).rsplit('_', 1)[0]}_{f_idx}"
                ).with_suffix(".jpg")
            else:
                filepath = (filepath.parent / f"{filepath.stem}_{f_idx}").with_suffix(".jpg")
            f_idx += 1

        if f_idx == 100:
            print("Can't save:", filepath.name)
            return

        image.save(filepath.with_suffix(".jpg"))
        original_filepath.unlink()

    elif filepath.suffix.lower() == ".pdf":
        convert_pdf_to_jpg(filepath)

    elif filepath.suffix.lower() != ".jpg":
        print("NOT converted:", filepath.stem)
        # filepath.unlink()


def convert_images_to_jpg(dir_path: Path, num_threads: int) -> None:
    """
    Convert all images in a directory to .jpg format
    """
    all_files = [f.stem for f in dir_path.iterdir() if not f.name.startswith(".")]

    with mp.Pool(processes=num_threads) as pool:
        filepaths = [
            filepath for filepath in dir_path.glob("*") if not filepath.name.startswith(".")
        ]

        for _ in tqdm(pool.imap_unordered(convert_image_to_jpg, filepaths)):
            pass

    jpg_files = [f.stem for f in dir_path.iterdir() if f.suffix.lower() == ".jpg"]
    lost_files = set(all_files) - set(jpg_files)

    if not lost_files:
        logger.info(
            f"In {dir_path}, All files were converted to .jpg, total amount: {len(jpg_files)}"
        )
    else:
        logger.warning(
            f"In {dir_path}, Not converted to .jpg, amount: {len(lost_files)}, files: {lost_files}"
        )


def remove_empty_labels(dir_path: Path) -> None:
    labels = [f.name for f in (dir_path.parent / "labels").iterdir() if not f.name.startswith(".")]
    counter = 0
    for label in labels:
        if not ((dir_path.parent / "labels") / label).stat().st_size:
            (dir_path.parent / "labels" / label).unlink()
            counter += 1
    if counter:
        logger.info(f"Removed {counter} empty labels")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    paths = {"root_path": Path(cfg.train.data_path)}
    if test_path := cfg.train.path_to_test_data:
        paths["test_path"] = Path(test_path)

    for _, data_path in paths.items():
        if (data_path / "images").exists():
            convert_images_to_jpg((data_path / "images"), cfg.train.num_workers)
            remove_empty_labels((data_path / "images"))


if __name__ == "__main__":
    register_heif_opener()
    main()
