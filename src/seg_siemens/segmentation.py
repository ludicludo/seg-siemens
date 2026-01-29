from pathlib import Path
from typing import List, Optional
import numpy as np
import pydicom
from pydicom.misc import is_dicom
from pydicom import Dataset
from pydicom.sr.codedict import codes
from highdicom.seg import (
    SegmentDescription,
    SegmentAlgorithmTypeValues,
)
from highdicom import AlgorithmIdentificationSequence
from highdicom.seg import Segmentation
from highdicom.seg import SegmentationTypeValues
from highdicom.uid import UID

import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def validate_dicom_file(file_path: Path) -> bool:
    """Validate the input DICOM file path."""
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not file_path.is_file():
        raise ValueError(f"The path {file_path} is not a file.")
    if not is_dicom(file_path):
        raise ValueError(f"The file {file_path} is not a DICOM file.")
    return True


def validate_dicom_data(dicom_data: pydicom.Dataset) -> bool:
    """Validate the DICOM data."""
    if not hasattr(dicom_data, "pixel_array"):
        raise ValueError("The DICOM data does not contain pixel array.")
    if dicom_data.pixel_array.ndim not in (2, 3):
        raise ValueError("The DICOM pixel array should be 2 or 3-dimensional.")
    return True


def read_dicom_segmentation(file_path: Path) -> np.ndarray:
    """
    Read a DICOM segmentation file and return a binary mask.

    Args:
        file_path: Path to the DICOM segmentation file.

    Returns:
        Binary mask as a NumPy array with dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a DICOM file or the DICOM data is invalid.
    """
    validate_dicom_file(file_path)

    dicom_data = pydicom.dcmread(file_path)
    validate_dicom_data(dicom_data)

    pixel_array = dicom_data.pixel_array

    # Convert the pixel array to a binary mask
    binary_mask = (pixel_array > 0).astype(np.uint8)

    return binary_mask


def load_dicom_series(
    directory: Path, expected_modality: Optional[str] = None
) -> List[Dataset]:
    """
    Charge et traite les fichiers DICOM selon leur modalité.

    Args:
        directory (Path): Chemin vers le dossier contenant les fichiers DICOM.
        expected_modality (Optional[str], optional): Modalité attendue. Si None, aucune vérification.

    Returns:
        List[Dataset]: Liste des objets DICOM, triée si modalité PET/CT, non triée sinon.
    """
    dicom_files = []
    for f in directory.iterdir():
        if pydicom.misc.is_dicom(str(f)):
            try:
                dicom_file = pydicom.dcmread(str(f), force=True)
                if (
                    expected_modality
                    and hasattr(dicom_file, "Modality")
                    and dicom_file.Modality != expected_modality
                ):
                    logging.warning(
                        f"Modalité incorrecte pour {f}: {dicom_file.Modality}"
                    )
                    continue
                dicom_files.append(dicom_file)
            except Exception as e:
                logging.error(f"Erreur lecture {f}: {str(e)}")

    if (
        dicom_files
        and hasattr(dicom_files[0], "Modality")
        and dicom_files[0].Modality in ["PT", "CT"]
    ):
        dicom_files.sort(
            key=lambda x: (
                float(x.ImagePositionPatient[2])
                if hasattr(x, "ImagePositionPatient") and x.ImagePositionPatient
                else 0
            )
        )
    return dicom_files


def check_series_compatibility(dicom_series, mask_array):
    """
    Vérifie si une série DICOM est compatible avec un masque binaire.

    Args:
        dicom_series (list): Liste des objets DICOM.
        mask_array (numpy.ndarray): Masque binaire.

    Returns:
        bool: True si la série DICOM est compatible avec le masque binaire, False sinon.
    """
    if len(dicom_series) != mask_array.shape[0]:
        return False

    for dicom_file, mask_slice in zip(dicom_series, mask_array):
        if (dicom_file.Rows, dicom_file.Columns) != mask_slice.shape:
            return False
        if not np.allclose(
            [dicom_file.Rows, dicom_file.Columns], [mask_slice.shape[1], mask_slice.shape[0]]
        ):
            return False

    return True


def create_segment_description(segment_number: int = 1) -> SegmentDescription:
    """
    Create a segment description.

    Args:
        segment_number (int, optional): Number of the segment. Defaults to 1.

    Returns:
        SegmentDescription: Segment description.
    """
    algorithm_identification = AlgorithmIdentificationSequence(
        name="Tumor Segmentation",
        version="1.0",
        family=codes.cid7162.ArtificialIntelligence,
    )

    segment_description = SegmentDescription(
        segment_number=segment_number,
        segment_label="Tumor",
        segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
        segmented_property_type=codes.SCT.Tumor,
        algorithm_type=SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=algorithm_identification,
        tracking_uid=UID(),
        tracking_id="1",
    )

    return segment_description


def create_seg_object(
    source_images: List[Dataset],
    mask_array: np.ndarray,
    series_number: int,
    series_instance_uid: Optional[str] = None,
    segment_number: int = 1,
) -> Segmentation:
    """
    Create a DICOM SEG object from a binary mask and DICOM images.

    Args:
        source_images (List[Dataset]): List of DICOM datasets.
        mask_array (np.ndarray): Binary mask of the tumor.
        series_number (int): Number of the SEG series.
        series_instance_uid (str): UID of the SEG series.
        segment_number (int, optional): Number of the segment. Defaults to 1.

    Returns:
        Segmentation: DICOM SEG object.

    Raises:
        ValueError: If the source images are not compatible with the binary mask.
        TypeError: If any of the input arguments are of the wrong type.
    """
    # Generate series_instance_uid if not provided
    if series_instance_uid is None:
        series_instance_uid = str(UID())

    # Validate input arguments
    if not isinstance(source_images, list) or not all(
        isinstance(img, Dataset) for img in source_images
    ):
        raise TypeError("source_images must be a list of pydicom.dcmread objects.")
    if not isinstance(mask_array, np.ndarray):
        raise TypeError("mask_array must be a numpy.ndarray.")
    if not isinstance(series_instance_uid, str):
        raise TypeError("series_instance_uid must be a string.")
    if not isinstance(series_number, int) or not isinstance(segment_number, int):
        raise TypeError("series_number and segment_number must be integers.")

    # Check series compatibility
    if not check_series_compatibility(source_images, mask_array):
        raise ValueError("Source images are not compatible with the binary mask.")

    # Create segment description
    segment_description = create_segment_description(segment_number)

    # Create SEG object
    seg = Segmentation(
        source_images=source_images,
        pixel_array=mask_array,
        segmentation_type=SegmentationTypeValues.BINARY,
        segment_descriptions=[segment_description],
        series_instance_uid=series_instance_uid,
        series_number=series_number,
        instance_number=1,
        sop_instance_uid=UID(),
        manufacturer="Your Manufacturer",
        manufacturer_model_name="Your Model",
        software_versions="1.0",
        device_serial_number="Your Serial Number",
        series_description="Segmentation from ICO script"
    )
    return seg


# Exemple d'utilisation
if __name__ == "__main__":
    ct_dicom_dir = Path("path/to/ct_series")
    pt_dicom_dir = Path("path/to/pt_series")
    mask_array = np.random.randint(
        0, 2, size=(len(list(pt_dicom_dir.iterdir())), 512, 512), dtype=np.uint8
    )  # Exemple de masque binaire
    series_instance_uid = "1.2.3.4.5.6.7.8.9.10"
    series_number = 100

    seg_object = create_seg_object(
        ct_dicom_dir, pt_dicom_dir, mask_array, series_instance_uid, series_number
    )

    # Sauvegarder l'objet DICOM SEG
    seg_object.save_as("path/to/seg.dcm")
