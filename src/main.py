import argparse
import logging
import numpy as np
from pathlib import Path
from seg_siemens.segmentation import (
    read_dicom_segmentation,
    create_seg_object,
    load_dicom_series,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seg", help="segmentation dicom file")
    return parser.parse_args()


def main():
    args = argument()
    seg_file_path = args.seg
    logger.debug(seg_file_path)

    # Chemin vers le fichier DICOM de segmentation
    # seg_file_path = (
    #     Path(directory)
    #     / "GRIJEA.SEG.PET_ICO_TAP_CRA.2759.2.2026.01.26.16.42.19.829.20953859.dcm"
    # )

    # Lecture du fichier DICOM de segmentation
    segment_file = Path(seg_file_path)
    mask = read_dicom_segmentation(segment_file)

    print(f"Shape: {mask.shape}")
    print(f"Dtype: {mask.dtype}")
    print(f"Unique values: {np.unique(mask)}")
    print(f"volume cm3: {mask.sum()}")

    series_number = 100
    pt_datasets = load_dicom_series(segment_file.parent, "PT")
    seg_object = create_seg_object(pt_datasets, mask, series_number)
    seg_object.save_as(segment_file.parent / "seg_test.dcm")
    # # Création de l'objet Segmentation de highdicom
    # seg = hd.seg.Segmentation.from_dataset(seg_dataset)

    # # Accès aux masques de segmentation
    # for segment in seg.segments:
    #     mask = segment.pixel_array
    #     print(f"Segment {segment.segment_number}:")
    #     print(f"Shape: {mask.shape}")
    #     print(f"Dtype: {mask.dtype}")
    #     print(f"Unique values: {np.unique(mask)}")


if __name__ == "__main__":
    main()
