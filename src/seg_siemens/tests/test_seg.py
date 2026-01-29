import os
import numpy as np
import pytest
from highdicom.seg import Segmentation
from seg_siemens import load_dicom_series, check_series_compatibility, create_seg_object


@pytest.fixture
def dicom_series():
    # Crée une série DICOM factice pour les tests
    dicom_files = []
    for i in range(10):
        dicom_file = pydicom.Dataset()
        dicom_file.Rows = 512
        dicom_file.Columns = 512
        dicom_file.PixelSpacing = [1.0, 1.0]
        dicom_file.ImagePositionPatient = [0.0, 0.0, float(i)]
        dicom_files.append(dicom_file)
    return dicom_files


@pytest.fixture
def mask_array():
    # Crée un masque binaire factice pour les tests
    return np.random.randint(0, 2, size=(10, 512, 512), dtype=np.uint8)


def test_load_dicom_series(tmp_path, dicom_series):
    # Crée des fichiers DICOM factices dans un dossier temporaire
    for i, dicom_file in enumerate(dicom_series):
        dicom_file.save_as(os.path.join(tmp_path, f"dicom_{i}.dcm"))

    # Charge les fichiers DICOM du dossier temporaire
    loaded_series = load_dicom_series(tmp_path)

    # Vérifie que les fichiers DICOM ont été correctement chargés et triés
    assert len(loaded_series) == len(dicom_series)
    for loaded_file, original_file in zip(loaded_series, dicom_series):
        assert loaded_file.Rows == original_file.Rows
        assert loaded_file.Columns == original_file.Columns
        assert loaded_file.PixelSpacing == original_file.PixelSpacing
        assert loaded_file.ImagePositionPatient == original_file.ImagePositionPatient


def test_check_series_compatibility(dicom_series, mask_array):
    # Vérifie que la série DICOM est compatible avec le masque binaire
    assert check_series_compatibility(dicom_series, mask_array)


def test_create_seg_object(tmp_path, dicom_series, mask_array):
    # Crée des fichiers DICOM factices dans un dossier temporaire
    for i, dicom_file in enumerate(dicom_series):
        dicom_file.save_as(os.path.join(tmp_path, f"dicom_{i}.dcm"))

    # Crée un objet DICOM SEG
    seg_object = create_seg_object(
        tmp_path, tmp_path, mask_array, "1.2.3.4.5.6.7.8.9.10", 100
    )

    # Vérifie que l'objet DICOM SEG a été correctement créé
    assert isinstance(seg_object, Segmentation)
    assert seg_object.segment_descriptions[0].segment_label == "Tumor"
