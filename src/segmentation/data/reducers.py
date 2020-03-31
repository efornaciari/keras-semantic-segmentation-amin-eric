import numpy as np

PIXEL_TYPE = 'pixel_type'
PIXEL_VALUE = 'pixel_value'

PATIENT_ID = 'patient_id'
TISSUE_BLOCK = 'tissue_block'
SEGMENTATION = 'segmentation'


class AveragePixelTypePerPatientReducer:
    def __init__(self, encodings):
        """
        [
            {
                "pixel_type": "Background",
                "pixel_value": (255, 255, 255)
            },
            {
                "pixel_type": "Leading Edge",
                "pixel_value": (33, 143, 166)
            },
            {
                "pixel_type": "Infiltrating Tumor",
                "pixel_value": (1, 2, 3)
            },
            {
                "pixel_type": "Cellular Tumor",
                "pixel_value": (1, 2, 3)
            },
            {
                "pixel_type": "Perinecrotic Zone",
                "pixel_value": (1, 2, 3)
            },
            {
                "pixel_type": "Necrosis",
                "pixel_value": (1, 2, 3)
            }
        ]
        """
        self.pixel_value_to_pixel_type = self._build_map(encodings, PIXEL_VALUE, PIXEL_TYPE)
        self.pixel_type_to_pixel_value = self._build_map(encodings, PIXEL_TYPE, PIXEL_VALUE)

    @staticmethod
    def _build_map(encodings, key_key, value_key):
        map = {}
        for encoding in encodings:
            map[encoding[key_key]] = encoding[value_key]
        return map

    def calculate_average_pixel_type_per_patient(self, df):
        pixel_type_to_pixel_count = df[SEGMENTATION].apply(self._build_reduce_function())
        for pixel_type in self.pixel_type_to_pixel_value.keys():
            df[pixel_type] = pixel_type_to_pixel_count.apply(lambda x: x[pixel_type])
        return df[list(self.pixel_type_to_pixel_value.keys()) + [PATIENT_ID]].groupby(PATIENT_ID).mean().reset_index()

    def _build_reduce_function(self):
        def _curried_reduce_function(segmentation):
            pixels, counts = np.unique(segmentation.reshape(-1, 3), return_counts=True, axis=0)
            pixel_value_to_pixel_count = dict(zip([tuple(pixel) for pixel in pixels], list(counts)))
            return {self.pixel_value_to_pixel_type[pixel_value]: pixel_count for pixel_value, pixel_count in pixel_value_to_pixel_count.items()}
        return _curried_reduce_function


