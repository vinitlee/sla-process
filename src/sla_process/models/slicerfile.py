import UVtoolsBootstrap as uv
import numpy as np
import cv2
from tqdm import tqdm
from line_profiler import profile
from pathlib import Path
from PIL import Image
from IPython.display import display

from typing import cast

# Pythonnet imports
import clr
from System import Array, Byte  # type: ignore
from System.Runtime.InteropServices import Marshal  # type: ignore
from Emgu.CV import Mat  # type: ignore
from Emgu.CV.CvEnum import DepthType  # type: ignore

import ctypes


class SlicerFile:
    """
    A class to represent a 3D slicer file.
    """

    __valid_params = [
        "BottomLayerCount",
        "BottomLightOffDelay",
        "BottomWaitTimeBeforeCure",
        "BottomExposureTime",
        "BottomWaitTimeAfterCure",
        "BottomLiftHeight",
        "BottomLiftSpeed",
        "BottomLiftAcceleration",
        "BottomLiftHeight2",
        "BottomLiftSpeed2",
        "BottomLiftAcceleration2",
        "BottomWaitTimeAfterLift",
        "BottomRetractSpeed",
        "BottomRetractAcceleration",
        "BottomRetractHeight2",
        "BottomRetractSpeed2",
        "BottomRetractAcceleration2",
        "BottomLightPWM",
        "PositionZ",
        "TransitionLayerCount",
        "LightOffDelay",
        "WaitTimeBeforeCure",
        "ExposureTime",
        "WaitTimeAfterCure",
        "LiftHeight",
        "LiftSpeed",
        "LiftAcceleration",
        "LiftHeight2",
        "LiftSpeed2",
        "LiftAcceleration2",
        "WaitTimeAfterLift",
        "RetractSpeed",
        "RetractAcceleration",
        "RetractHeight2",
        "RetractSpeed2",
        "RetractAcceleration2",
        "LightPWM",
        "Pause",
        "ChangeResin",
        "MachineName",
        "ResinName",
    ]
    print_params: dict[str, float | int] = {}
    UVObj: "uv.FileFormats.FileFormat"
    layers: np.ndarray
    thumbnails: list[np.ndarray] = []

    def __init__(self, file_path: str | Path):
        """
        Initialize the SlicerFile with a file path.
        :param file_path: Path to the slicer file.
        """
        self.file_path = Path(file_path)
        try:
            self.UVObj = cast(
                "uv.FileFormats.FileFormat",
                uv.FileFormats.FileFormat.Open(str(file_path)),
            )
        except Exception as e:
            raise RuntimeError(e)

        # if self.UVObj is None:
        #     raise RuntimeError(f"Unable to find {file_path} or it's invalid file")

        self.unpack_print_params()
        self.unpack_thumbnails()
        self.unpack_layers()

    def save(self, file_path: str | Path | None = None):
        """
        Save the SlicerFile to the specified file path.
        :param file_path: Path to save the slicer file. If None, overwrite the original file.
        """
        if file_path is None:
            file_path = str(self.file_path)
        file_path = str(file_path)

        self.pack_print_params()
        self.pack_thumbnails()
        self.pack_layers()

        self.UVObj.SetNormalWaitTimeBeforeCureOrLightOffDelay(
            self.print_params.get("WaitTimeBeforeCure", 0.0)
        )

        self.UVObj.SaveAs(str(file_path))
        print(f"💾 Saved to {file_path}")

    def unpack_print_params(self):
        """
        Unpack the print parameters from the UVTools file.
        :return: None
        """
        for param in self.__valid_params:
            if hasattr(self.UVObj, param):
                self.print_params[param] = getattr(self.UVObj, param)

    def pack_print_params(self):
        """
        Pack the print parameters back to the UVTools file.
        :return: None
        """
        for param, value in self.print_params.items():
            if hasattr(self.UVObj, param):
                setattr(self.UVObj, param, value)

    def unpack_thumbnails(self):
        """
        Unpack the thumbnails from the UVTools file.
        :return: None
        """
        self.thumbnails = [
            SlicerFile.mat_to_np(thumbnail)[..., :3]
            for thumbnail in self.UVObj.Thumbnails
        ]

    def pack_thumbnails(self):
        """
        Pack the thumbnails back to the UVTools file.
        :return: None
        """
        for i, thumbnail in enumerate(self.thumbnails):
            self.UVObj.SetThumbnail(i, SlicerFile.np_to_mat(thumbnail))
        self.UVObj.SanitizeThumbnails()

    @property
    def bottom_layer_count(self):
        return self.UVObj.BottomLayerCount

    @staticmethod
    def match_orientation(im, im_to_match, rotation=cv2.ROTATE_90_CLOCKWISE):
        if (im.shape[0] > im.shape[1]) ^ (im_to_match.shape[0] > im_to_match.shape[1]):
            im = cv2.rotate(im, rotation)

        return im

    @staticmethod
    def resize_to_fit(im, shape):
        # Resize the image to fit within the dimensions of the target image, keeping the aspect ratio
        target_height, target_width = shape[:2]  # Width and height to fit in
        im_height, im_width = im.shape[:2]

        target_aspect_ratio = target_width / target_height
        im_aspect_ratio = im_width / im_height

        if im_aspect_ratio > target_aspect_ratio:
            # Image is wider than the target
            new_width = target_width
            new_height = int(target_width / im_aspect_ratio)
        else:
            # Image is taller than the target
            new_height = target_height
            new_width = int(target_height * im_aspect_ratio)

        im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # Create a new image with the target dimensions and fill it with black
        new_im = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        # Calculate the position to place the resized image in the center
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        slice = [y_offset, y_offset + new_height, x_offset, x_offset + new_width]
        # Place the resized image in the center of the new image
        new_im[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = im
        return new_im

    def set_thumbnails(self, im, index=None):
        """
        Set the image as a thumbnail in the UVTools file.
        :param im: The image to set as a thumbnail.
        :param index: The index of the thumbnail to set. If None, set the first thumbnail.
        :return: None
        """
        for i, thumbnail in enumerate(self.thumbnails):
            if index is None or i == index:
                temp_im = im.copy()
                temp_im = self.match_orientation(temp_im, thumbnail)
                temp_im = self.resize_to_fit(temp_im, thumbnail.shape)
                self.thumbnails[i] = temp_im

    @profile
    def unpack_layers(self):
        """
        Unpack the layers from the UVTools file into 3D numpy array.
        :return: None
        """
        self.layers = np.stack(
            [
                SlicerFile.mat_to_np(layer.LayerMat)
                for layer in self.UVObj.Layers
                if layer is not None
            ]
        )

        if self.layers.size == 0:
            raise RuntimeError("No layers found in the file")

    @profile
    def pack_layers(self):
        """
        Pack the layers back to the UVTools file.
        :return: None
        """
        self.UVObj.Layers = Array.CreateInstance(uv.Layers.Layer, len(self.layers))

        # Set the new layers from the array
        for i, layer in tqdm(
            enumerate(self.layers),
            total=len(self.layers),
            desc=f"Packing {len(self.layers)} Layers",
            unit="layers",
        ):
            layer_mat = SlicerFile.np_to_mat(layer)
            new_layer = self.make_layer(layer_mat=layer_mat)
            self.UVObj.SetLayer(i, new_layer)

        self.UVObj.RebuildLayersProperties()

    @profile
    def make_layer(self, layer_mat=None):
        """
        Create a new layer object.
        :return: A new Layer object.
        """
        # Create a new Layer object
        new_layer = uv.Layers.Layer(self.UVObj)
        if layer_mat is not None:
            new_layer.LayerMat = layer_mat
        return new_layer

    @profile
    @staticmethod
    def mat_to_np(mat, dtype=np.int16):
        """
        Convert an Emgu CV Mat object to a NumPy array.

        :param mat: The Emgu CV Mat object.
        :return: A NumPy array containing the image data.
        """
        # Get the dimensions of the Mat
        height, width = mat.Rows, mat.Cols
        channels = mat.NumberOfChannels
        element_size = mat.ElementSize  # Size of each element in bytes

        # Calculate the total number of bytes
        total_bytes = height * width * element_size

        # Get a pointer to the Mat's data
        mat_ptr = int(mat.DataPointer.ToInt64())

        # Create a NumPy array from the raw bytes
        raw_bytes = np.zeros(total_bytes, dtype=np.uint8)
        ctypes.memmove(raw_bytes.ctypes.data, mat_ptr, total_bytes)

        # Reshape the array to match the Mat's dimensions
        raw_image = (
            raw_bytes.reshape((height, width, channels))
            if channels > 1
            else raw_bytes.reshape((height, width))
        )

        # Cast to more forgiving type
        raw_image = raw_image.astype(dtype)

        return raw_image

    @profile
    @staticmethod
    def np_to_mat(numpy_array: np.typing.NDArray):
        """
        Convert a NumPy array (grayscale or RGB) to an Emgu.CV.Mat object.

        :param numpy_array: The NumPy array containing the image data.
        :return: An Emgu.CV.Mat object.
        """
        if numpy_array.dtype not in [np.uint8, np.int16]:
            raise ValueError(f"NumPy array <{numpy_array.dtype}> not of valid type")
        numpy_array = np.clip(numpy_array, 0, 255).astype(np.uint8)

        # Handle grayscale (2D) and RGB (3D) images
        if len(numpy_array.shape) == 2:  # Grayscale
            height, width = numpy_array.shape
            channels = 1
        elif len(numpy_array.shape) == 3 and numpy_array.shape[2] == 3:  # RGB
            height, width, channels = numpy_array.shape
            # Note: OpenCV uses BGR format, but Emgu CV uses RGB format. If needed, you can convert it here.
            # numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("NumPy array must be a 2D (grayscale) or 3D (RGB) array.")

        mat = Mat(height, width, DepthType.Cv8U, channels)
        total_bytes = numpy_array.nbytes

        # Pointer to the Mat's buffer (IntPtr -> convert to a Python int)
        mat_ptr = int(mat.DataPointer.ToInt64())

        # Pointer to the NumPy array’s data
        np_ptr = numpy_array.ctypes.data

        # Do a raw memory copy
        ctypes.memmove(mat_ptr, np_ptr, total_bytes)

        return mat
