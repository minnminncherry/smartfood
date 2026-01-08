import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


class DatasetDivider:
    """Divide image datasets into train/test splits and copy files.

    sources: mapping from class name -> source directory (string or Path)
    output_dir: directory where `train/` and `test/` folders will be created
    train_ratio: fraction of images to place in the training set
    extensions: iterable of accepted file extensions (lowercase, with dot)
    """

    def __init__(self, sources: Dict[str, str], output_dir: str, train_ratio: float = 0.8,
                 extensions: Iterable[str] = None):
        self.sources = {str(k): Path(v) for k, v in sources.items()}
        self.output_dir = Path(output_dir)
        self.train_ratio = float(train_ratio)
        if extensions is None:
            self.extensions = IMAGE_EXTENSIONS
        else:
            self.extensions = {e if e.startswith('.') else f'.{e}' for e in extensions}

        self.train_dir = self.output_dir / 'train'
        self.test_dir = self.output_dir / 'test'

    def _gather_images(self, folder: Path) -> List[Path]:
        if not folder.exists() or not folder.is_dir():
            return []
        return sorted([p for p in folder.iterdir() if p.suffix.lower() in self.extensions])

    def divide_and_copy(self, dry_run: bool = False) -> None:
        """Divide each class source into train/test and copy files.

        If `dry_run` is True, actions will be printed but files won't be copied.
        """
        for class_name, src_folder in self.sources.items():
            images = [f for f in os.listdir(src_folder)
          if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"))]
            total = len(images)
            train_count = int(self.train_ratio * total)
            test_count = total - train_count

            print(f"Class '{class_name}': total={total}, train={train_count}, test={test_count}")

            train_class_dir = self.train_dir / class_name
            test_class_dir = self.test_dir / class_name
            print(f"Creating directories: {train_class_dir}, {test_class_dir}")

            for p in images[:train_count]:
                src = os.path.join(src_folder, p)
                dst = os.path.join(train_class_dir, p)
                shutil.copy2(src, dst)

            for p in images[train_count:]:
                src = os.path.join(src_folder, p)
                dst = os.path.join(test_class_dir, p)
                shutil.copy2(src, dst)

        print("Copying complete.")

    def report_counts(self) -> None:
        """Print counts of images in train and test folders per class."""
        print("For the training dataset:\n")
        for class_folder in sorted(self.train_dir.iterdir()):
            if class_folder.is_dir():
                count = sum(1 for f in class_folder.iterdir() if f.suffix.lower() in self.extensions)
                print(f"| {class_folder.name:<15}: {count:6,} images")

        print("-----------------------------------------------------")
        print("\nFor the testing dataset\n")
        for class_folder in sorted(self.test_dir.iterdir()):
            if class_folder.is_dir():
                count = sum(1 for f in class_folder.iterdir() if f.suffix.lower() in self.extensions)
                print(f"| {class_folder.name:<15}: {count:6,} images")


    def delete_images_in_folder(self, folder_paths) -> None:
        for folder_path in folder_paths.values():

            # Image extensions to delete
            image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

            for filename in os.listdir(folder_path):
                if filename.lower().endswith(image_extensions):
                    file_path = os.path.join(folder_path, filename)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

            print("All images deleted successfully!")


if __name__ == '__main__':
    # Default configuration preserved from the original script
    sources = {
        'mangos': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/food_images/mangos/images',
        'wasabi peas': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/food_images/peas/images',
        'Mi Gorenge noodle': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/food_images/noodles/images',
        'banana': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/food_images/banana/images',
        'dragon_fruit': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/food_images/dragon_fruit/images',
        'guava': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/food_images/guava/images',
        'papaya': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/food_images/papaya/images',
        'pineapple': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/food_images/pineapple/images',
        'pomelo': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/food_images/pomelo/images',
    }

    test_dest = {
        'mangos': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/test/mangos',
        'wasabi peas': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/test/wasabi peas',
        'Mi Gorenge noodle': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/test/Mi Gorenge noodle',
        'banana': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/test/banana',
        'dragon_fruit': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/test/dragon_fruit',
        'guava': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/test/guava',
        'papaya': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/test/papaya',
        'pineapple': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/test/pineapple',
        'pomelo': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/test/pomelo'
    }

    train_dest = {
        'mangos': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/train/mangos',
        'wasabi peas': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/train/wasabi peas',
        'Mi Gorenge noodle': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/train/Mi Gorenge noodle',
        'banana': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/train/banana',
        'dragon_fruit': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/train/dragon_fruit',
        'guava': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/train/guava',
        'papaya': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/train/papaya',
        'pineapple': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/train/pineapple',
        'pomelo': '/home/licongcong/Desktop/learning/gse_project/scan2food/model/train/pomelo'
    }

    output_folder = '/home/licongcong/Desktop/learning/gse_project/scan2food/model/'
    divider = DatasetDivider(sources, output_folder, train_ratio=0.7)
    divider.delete_images_in_folder(train_dest)
    divider.delete_images_in_folder(test_dest)

    divider.divide_and_copy(dry_run=False)
    divider.report_counts()
