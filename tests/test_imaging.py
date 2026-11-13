"""Synthetic, non-patient fixtures. Run with unittest discover or pytest."""
from io import BytesIO
import gzip
import unittest
from unittest.mock import patch
import zipfile

import nibabel as nib
import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from nucleo_viewer.errors import AppError
from nucleo_viewer import imaging
from nucleo_viewer.imaging import CTVolume, Mask, load_dicom_zip, load_mask, render_slice


SERIES = generate_uid()


def dicom_bytes(z=0, instance=1, pixels=None, **overrides):
    """Create a real, tiny Part-10 DICOM, without external image files."""
    pixels = np.asarray([[0, 1], [2, 3]] if pixels is None else pixels, dtype="<i2")
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CT"
    dataset.SeriesInstanceUID = SERIES
    dataset.StudyInstanceUID = generate_uid()
    dataset.InstanceNumber = instance
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [10, 20, z]
    dataset.PixelSpacing = [2, 3]
    dataset.SliceThickness = 2
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleSlope = 2
    dataset.RescaleIntercept = -1000
    dataset.PixelData = pixels.tobytes()
    for name, value in overrides.items():
        if name == "TransferSyntaxUID":
            meta.TransferSyntaxUID = value
        elif value is None:
            delattr(dataset, name)
        else:
            setattr(dataset, name, value)
    stream = BytesIO()
    dataset.save_as(stream, write_like_original=False)
    return stream.getvalue()


def zipped(entries):
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return stream.getvalue()


def nifti_bytes(data, affine=None):
    image = nib.Nifti1Image(np.asarray(data), np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units("mm")
    return gzip.compress(image.to_bytes())


class DicomTests(unittest.TestCase):
    def test_sort_hu_shape_and_lps_to_ras(self):
        raw = zipped([("later.dcm", dicom_bytes(2, 1, [[4, 5], [6, 7]])),
                      ("earlier.dcm", dicom_bytes(0, 2))])
        ct = load_dicom_zip(raw)
        self.assertEqual(ct.slice_count, 2)
        self.assertEqual(ct.shape, (2, 2, 2))
        self.assertEqual(ct.pixels.dtype, np.float32)
        np.testing.assert_array_equal(ct.pixels[0], [[-1000, -998], [-996, -994]])
        np.testing.assert_array_equal(ct.pixels[1], [[-992, -990], [-988, -986]])
        np.testing.assert_allclose(ct.affine, [[-3, 0, 0, -10], [0, -2, 0, -20],
                                              [0, 0, 2, 0], [0, 0, 0, 1]])

    def test_ignore_metadata_files_and_directories(self):
        raw = zipped([("nested/a.dcm", dicom_bytes()), ("folder/", b""),
                      ("__MACOSX/junk", b"junk"), (".hidden/a", b"junk"),
                      (".DS_Store", b"junk"), ("DICOMDIR", b"junk")])
        self.assertEqual(load_dicom_zip(raw).slice_count, 1)

    def test_unsafe_paths_even_when_hidden(self):
        for path in ("../a.dcm", "/absolute.dcm", "C:\\a.dcm", "folder/../../a", ".hidden/../a"):
            with self.subTest(path=path), self.assertRaises(AppError):
                load_dicom_zip(zipped([(path, dicom_bytes())]))

    def test_bounds_are_checked_before_reading_members(self):
        raw = zipped([("a.dcm", dicom_bytes()), ("b.dcm", dicom_bytes(2, 2))])
        for constant in ("MAX_ZIP_ENTRIES", "MAX_ZIP_MEMBER_BYTES", "MAX_ZIP_EXPANDED_BYTES"):
            with self.subTest(constant=constant), patch.object(imaging, constant, 1):
                with patch.object(zipfile.ZipFile, "open", side_effect=AssertionError("decompressed too early")):
                    with self.assertRaises(AppError):
                        load_dicom_zip(raw)

    def test_invalid_or_empty_archive(self):
        for raw in (b"not a zip", zipped([]), zipped([("junk.txt", b"not DICOM")])):
            with self.subTest(raw=raw[:8]), self.assertRaises(AppError):
                load_dicom_zip(raw)

    def test_required_metadata_and_pixel_length(self):
        for changes in ({"ImagePositionPatient": None}, {"PixelSpacing": [0, 1]},
                        {"RescaleSlope": "nan"}, {"RescaleIntercept": "inf"},
                        {"PixelData": b"\0\0"}, {"Rows": 0}, {"BitsStored": 17},
                        {"NumberOfFrames": 2}, {"PhotometricInterpretation": "RGB"},
                        {"Modality": "MR"}, {"SliceThickness": -1}):
            with self.subTest(changes=changes), self.assertRaises(AppError):
                load_dicom_zip(zipped([("a.dcm", dicom_bytes(**changes))]))

    def test_series_and_geometry_consistency(self):
        for changes in ({"SeriesInstanceUID": generate_uid()}, {"PixelSpacing": [1, 1]},
                        {"ImageOrientationPatient": [-1, 0, 0, 0, -1, 0]}):
            with self.subTest(changes=changes), self.assertRaises(AppError):
                load_dicom_zip(zipped([("a", dicom_bytes()), ("b", dicom_bytes(2, 2, **changes))]))

    def test_instance_number_is_optional(self):
        ct = load_dicom_zip(zipped([("a", dicom_bytes(0, InstanceNumber=None)),
                                   ("b", dicom_bytes(2, InstanceNumber=None))]))
        self.assertEqual(ct.slice_count, 2)

    def test_duplicate_instance_and_position(self):
        for z, instance in ((2, 1), (0, 2)):
            with self.subTest(z=z, instance=instance), self.assertRaises(AppError):
                load_dicom_zip(zipped([("a", dicom_bytes()), ("b", dicom_bytes(z, instance))]))
        duplicate = dicom_bytes()
        with self.assertRaises(AppError):
            load_dicom_zip(zipped([("a", duplicate), ("b", duplicate)]))

    def test_nonuniform_spacing_and_drift(self):
        with self.assertRaisesRegex(AppError, "gaps between CT slices are uneven"):
            load_dicom_zip(zipped([(str(i), dicom_bytes(z, i + 1)) for i, z in enumerate([0, 2, 5])]))
        with self.assertRaisesRegex(AppError, "do not line up correctly"):
            load_dicom_zip(zipped([("a", dicom_bytes()),
                                  ("b", dicom_bytes(2, 2, ImagePositionPatient=[10.2, 20, 2]))]))

    def test_oblique_geometry_rejected(self):
        with self.assertRaisesRegex(AppError, "horizontal CT slices"):
            load_dicom_zip(zipped([("a", dicom_bytes(ImageOrientationPatient=[1, 0, 0, 0, 0.8, 0.6]))]))

    def test_unsupported_transfer_syntax(self):
        encoded = dicom_bytes()
        for suffix in (b"2", b"5"):  # Explicit big-endian and RLE compressed
            unsupported = encoded.replace(b"1.2.840.10008.1.2.1", b"1.2.840.10008.1.2." + suffix)
            with self.subTest(suffix=suffix), self.assertRaisesRegex(AppError, "storage format the viewer cannot read"):
                load_dicom_zip(zipped([("a", unsupported)]))

    def test_voxel_cap(self):
        with patch.object(imaging, "MAX_VOXELS", 4), self.assertRaises(AppError):
            load_dicom_zip(zipped([("a", dicom_bytes()), ("b", dicom_bytes(2, 2))]))

    def test_signed_twelve_bit_pixels(self):
        ct = load_dicom_zip(zipped([("a", dicom_bytes(pixels=[[-1, -2048], [0, 2047]],
                                                               BitsStored=12, HighBit=11,
                                                               RescaleSlope=1, RescaleIntercept=0))]))
        np.testing.assert_array_equal(ct.pixels[0], [[-1, -2048], [0, 2047]])


class MaskTests(unittest.TestCase):
    def test_binary_mask_and_affine(self):
        data = np.zeros((2, 3, 4), dtype=np.uint8)
        data[1, 2, 3] = 1
        affine = np.diag([-2., -3., 4., 1.])
        mask = load_mask(nifti_bytes(data, affine))
        np.testing.assert_array_equal(mask.data, data)
        np.testing.assert_allclose(mask.affine, affine)
        self.assertEqual(mask.data.dtype, np.uint8)

    def test_bad_dimensions_binary_and_nonfinite(self):
        for data in (np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2, 2, 1), dtype=np.uint8),
                     np.full((2, 2, 2), 2, dtype=np.uint8),
                     np.full((2, 2, 2), np.nan, dtype=np.float32),
                     np.full((2, 2, 2), np.inf, dtype=np.float32)):
            with self.subTest(shape=data.shape), self.assertRaises(AppError):
                load_mask(nifti_bytes(data))

    def test_explicit_geometry_is_required(self):
        image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))
        payload = bytearray(image.to_bytes())
        payload[252:256] = b"\0" * 4  # qform_code and sform_code are both zero
        with self.assertRaisesRegex(AppError, "missing valid positioning information"):
            load_mask(gzip.compress(payload))

    def test_singular_affine(self):
        image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))
        payload = bytearray(image.to_bytes())
        payload[280:296] = b"\0" * 16  # singular sform, without writer repairing it
        with self.assertRaises(AppError):
            load_mask(gzip.compress(payload))

    def test_expansion_limit(self):
        with patch.object(imaging, "MAX_NIFTI_BYTES", 100), self.assertRaises(AppError):
            load_mask(gzip.compress(b"x" * 100))

    def test_bad_gzip_and_truncated_nifti(self):
        raw = nifti_bytes(np.zeros((2, 2, 2), dtype=np.uint8))
        for bad in (b"bad", raw[:-5], gzip.compress(b"bad"), gzip.compress(gzip.decompress(raw)[:-1])):
            with self.subTest(raw=bad[:8]), self.assertRaises(AppError):
                load_mask(bad)

    def test_declared_shape_checked_before_allocation(self):
        image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))
        payload = bytearray(image.to_bytes())
        # NIfTI-1 dim[1:4] are int16 values at byte offsets 42, 44, 46.
        for offset in (42, 44, 46):
            payload[offset:offset + 2] = (30_000).to_bytes(2, "little", signed=True)
        with self.assertRaisesRegex(AppError, "too large for this viewer"):
            load_mask(gzip.compress(payload))

    def test_meter_units_converted_to_mm(self):
        image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.diag([0.001, 0.001, 0.001, 1]))
        image.header.set_xyzt_units("meter")
        np.testing.assert_allclose(load_mask(gzip.compress(image.to_bytes())).affine, np.eye(4), atol=1e-6)


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.affine = np.diag([-1., -1., 1., 1.])
        self.ct = CTVolume(np.zeros((1, 2, 3), dtype=np.float32), self.affine)

    def rendered(self, ct=None, overlays=None, **kwargs):
        raw = render_slice(self.ct if ct is None else ct, 0, [] if overlays is None else overlays, **kwargs)
        self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))
        return np.asarray(Image.open(BytesIO(raw)))

    def test_window_and_native_dimensions(self):
        ct = CTVolume(np.array([[[-200, 0, 200], [-400, 0, 400]]], dtype=np.float32), self.affine)
        rgb = self.rendered(ct, center=0, width=400)
        self.assertEqual(rgb.shape, (2, 3, 3))
        np.testing.assert_array_equal(rgb[0, :, 0], [0, 128, 255])

    def test_overlay_is_sampled_in_world_coordinates(self):
        # Mask voxel (0,0,0) is at CT column 1, NOT column 0.
        affine = self.affine.copy()
        affine[0, 3] = -1
        mask = Mask(np.ones((1, 1, 1), dtype=np.uint8), affine)
        base = self.rendered()
        rgb = self.rendered(overlays=[(mask, (255, 0, 0))])
        changed = np.any(rgb != base, axis=-1)
        np.testing.assert_array_equal(changed, [[False, True, False], [False, False, False]])
        self.assertGreater(rgb[0, 1, 0], rgb[0, 1, 1])

    def test_ct_and_overlay_share_radiological_flips(self):
        affine = np.eye(4)
        ct = CTVolume(np.array([[[0, 100, 200], [300, 400, 500]]], dtype=np.float32), affine)
        mask_data = np.zeros((3, 2, 1), dtype=np.uint8)
        mask_data[0, 0, 0] = 1
        base = self.rendered(ct, center=250, width=500)
        rgb = self.rendered(ct, [(Mask(mask_data, affine), (255, 0, 0))], center=250, width=500)
        self.assertEqual(base[0, 0, 0], 255)
        self.assertEqual(base[-1, -1, 0], 0)
        expected = np.zeros((2, 3), dtype=bool)
        expected[-1, -1] = True
        np.testing.assert_array_equal(np.any(rgb != base, axis=-1), expected)

    def test_in_plane_rotation_applies_to_overlays(self):
        affine = np.array([[0., -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ct = CTVolume(np.zeros((1, 2, 3), dtype=np.float32), affine)
        data = np.zeros((3, 2, 1), dtype=np.uint8)
        data[2, 1, 0] = 1
        base = self.rendered(ct)
        rgb = self.rendered(ct, [(Mask(data, affine), (0, 255, 0))])
        self.assertEqual(rgb.shape, (3, 2, 3))
        expected = np.zeros((3, 2), dtype=bool)
        expected[2, 1] = True
        np.testing.assert_array_equal(np.any(rgb != base, axis=-1), expected)

    def test_nearest_neighbor_and_out_of_bounds(self):
        affine = self.affine.copy()
        affine[0, 0] = -2
        data = np.zeros((2, 2, 1), dtype=np.uint8)
        data[1, 0, 0] = 1
        base = self.rendered()
        rgb = self.rendered(overlays=[(Mask(data, affine), (255, 0, 0))])
        expected = np.zeros((2, 3), dtype=bool)
        expected[0, 2] = True  # column 1 maps to 0.5, ties-to-even chooses zero
        np.testing.assert_array_equal(np.any(rgb != base, axis=-1), expected)

    def test_disjoint_field_of_view_rejected_but_empty_slice_allowed(self):
        distant = self.affine.copy()
        distant[0, 3] = 1000
        with self.assertRaisesRegex(AppError, "outside"):
            self.rendered(overlays=[(Mask(np.ones((2, 2, 1), dtype=np.uint8), distant), (255, 0, 0))])
        empty = Mask(np.zeros((3, 2, 1), dtype=np.uint8), self.affine)
        np.testing.assert_array_equal(self.rendered(overlays=[(empty, (255, 0, 0))]), self.rendered())
        # A mask may overlap a different CT slice, while the requested slice is empty.
        ct = CTVolume(np.zeros((3, 2, 3), dtype=np.float32), self.affine)
        later = self.affine.copy()
        later[2, 3] = 2
        np.testing.assert_array_equal(self.rendered(ct, [(Mask(np.ones((1, 1, 1), dtype=np.uint8), later),
                                                        (255, 0, 0))]), self.rendered(ct))

    def test_bad_index_and_window(self):
        for index in (-1, 1, 0.5, True):
            with self.subTest(index=index), self.assertRaises(AppError):
                render_slice(self.ct, index, [])
        for center, width in ((0, 0), (0, -1), (float("nan"), 1), (0, float("inf"))):
            with self.subTest(center=center, width=width), self.assertRaises(AppError):
                render_slice(self.ct, 0, [], center, width)


if __name__ == "__main__":
    unittest.main()
