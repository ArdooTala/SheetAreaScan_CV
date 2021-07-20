import pathlib as pl
from src.cv_utils import *


def main():
    img_path = pl.Path(r"data/Images")
    exp_path = pl.Path(r"data/Exports")
    for file in img_path.glob("*.jpeg"):

        exports = exp_path / file.stem
        pl.Path.mkdir(exp_path / file.stem, parents=True, exist_ok=True)
        print("Image > ", file, " > ", exports)
        sheet_scanner = SheetScanner(file, exports)
        sheet_scanner.instructions()
        sheet_scanner.refine()


if __name__ == '__main__':
    main()
