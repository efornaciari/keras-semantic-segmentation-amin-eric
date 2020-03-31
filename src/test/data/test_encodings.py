import unittest
import numpy as np
import pandas as pd

import segmentation.data.reducers as subject


class TestAveragePixelTypePerPatientReducer(unittest.TestCase):
    def test_calculate_average_pixel_type_per_patient(self):
        # GIVEN
        encodings = [
            {
                subject.PIXEL_TYPE: 'A',
                subject.PIXEL_VALUE: (1, 2, 3),
            },
            {
                subject.PIXEL_TYPE: 'B',
                subject.PIXEL_VALUE: (3, 2, 1),
            }
        ]

        data = pd.DataFrame([
            {
                "patient_id": "TCGA-06-0145-01Z-00-DX4.52da9a91-d1df-4e6b-8a80-9f2a2749cdeb",
                "tissue_block": "A",
                "segmentation": np.array([
                    [
                        [
                            [1, 2, 3],
                            [3, 2, 1],
                        ],
                        [
                            [1, 2, 3],
                            [3, 2, 1],
                        ],
                    ],
                ])
            },
            {
                "patient_id": "TCGA-06-0145-01Z-00-DX4.52da9a91-d1df-4e6b-8a80-9f2a2749cdeb",
                "tissue_block": "A",
                "segmentation": np.array([
                    [
                        [
                            [1, 2, 3],
                            [3, 2, 1],
                        ],
                        [
                            [1, 2, 3],
                            [3, 2, 1],
                        ],
                    ],
                ])
            },
            {
                "patient_id": "TCGA-06-0145-01Z-00-DX3.319351f0-313a-4c99-b12f-a67b52c6785c",
                "tissue_block": "A",
                "segmentation": np.array([
                    [
                        [
                            [1, 2, 3],
                            [1, 2, 3],
                        ],
                        [
                            [1, 2, 3],
                            [3, 2, 1],
                        ],
                    ],
                ])
            }
        ])
        expected_average_pixel_type_per_patient = pd.DataFrame([
            {
                "patient_id": "TCGA-06-0145-01Z-00-DX3.319351f0-313a-4c99-b12f-a67b52c6785c",
                'A': 3,
                'B': 1,
            },
            {
                "patient_id": "TCGA-06-0145-01Z-00-DX4.52da9a91-d1df-4e6b-8a80-9f2a2749cdeb",
                'A': 2,
                'B': 2,
            },
        ])
        reducer = subject.AveragePixelTypePerPatientReducer(encodings)

        # WHEN
        average_pixel_type_per_patient = reducer.calculate_average_pixel_type_per_patient(data)

        # THEN
        assert average_pixel_type_per_patient.equals(expected_average_pixel_type_per_patient)
