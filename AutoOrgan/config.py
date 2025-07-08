config_dict_low = {
    "dataset_name": "Dataset106_TotalBone3",
    "plans_name": "nnUNetPlans",
    "original_median_spacing_after_transp": [
        3.0,
        3.0,
        3.0
    ],
    "original_median_shape_after_transp": [
        662,
        167,
        167
    ],
    "image_reader_writer": "SimpleITKIO",
    "transpose_forward": [
        0,
        1,
        2
    ],
    "transpose_backward": [
        0,
        1,
        2
    ],
    "configurations": {
        "3d_fullres": {
            "data_identifier": "nnUNetPlans_3d_fullres",
            "preprocessor_name": "DefaultPreprocessor",
            "batch_size": 2,
            "patch_size": [
                256,
                80,
                80
            ],
            "median_image_size_in_voxels": [
                662.0,
                167.0,
                167.0
            ],
            "spacing": [
                3.0,
                3.0,
                3.0
            ],
            "normalization_schemes": [
                "CTNormalization"
            ],
            "use_mask_for_norm": [
                False
            ],
            "resampling_fn_data": "resample_data_or_seg_to_shape",
            "resampling_fn_seg": "resample_data_or_seg_to_shape",
            "resampling_fn_data_kwargs": {
                "is_seg": False,
                "order": 3,
                "order_z": 0,
                "force_separate_z": None
            },
            "resampling_fn_seg_kwargs": {
                "is_seg": True,
                "order": 1,
                "order_z": 0,
                "force_separate_z": None
            },
            "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
            "resampling_fn_probabilities_kwargs": {
                "is_seg": False,
                "order": 1,
                "order_z": 0,
                "force_separate_z": None
            },
            "architecture": {
                "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
                "arch_kwargs": {
                    "n_stages": 6,
                    "features_per_stage": [
                        32,
                        64,
                        128,
                        256,
                        320,
                        320
                    ],
                    "conv_op": "torch.nn.modules.conv.Conv3d",
                    "kernel_sizes": [
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ]
                    ],
                    "strides": [
                        [
                            1,
                            1,
                            1
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            1,
                            1
                        ]
                    ],
                    "n_conv_per_stage": [
                        2,
                        2,
                        2,
                        2,
                        2,
                        2
                    ],
                    "n_conv_per_stage_decoder": [
                        2,
                        2,
                        2,
                        2,
                        2
                    ],
                    "conv_bias": True,
                    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                    "norm_op_kwargs": {
                        "eps": 1e-05,
                        "affine": True
                    },
                    "dropout_op": None,
                    "dropout_op_kwargs": None,
                    "nonlin": "torch.nn.LeakyReLU",
                    "nonlin_kwargs": {
                        "inplace": True
                    }
                },
                "_kw_requires_import": [
                    "conv_op",
                    "norm_op",
                    "dropout_op",
                    "nonlin"
                ]
            },
            "batch_dice": True
        },
    },
    "experiment_planner_used": "ExperimentPlanner",
    "label_manager": "LabelManager",
    "foreground_intensity_properties_per_channel": {
        "0": {
            "max": 34867.0,
            "mean": 435.54876708984375,
            "median": 306.0,
            "min": -1023.0,
            "percentile_00_5": -80.0,
            "percentile_99_5": 1629.0,
            "std": 422.6369323730469
        }
    }
}

config_dict = {
    "dataset_name": "Dataset105_TotalBone",
    "plans_name": "nnUNetPlans",
    "original_median_spacing_after_transp": [
        1.5,
        0.9765625,
        0.9765625
    ],
    "original_median_shape_after_transp": [
        1323,
        512,
        512
    ],
    "image_reader_writer": "SimpleITKIO",
    "transpose_forward": [
        0,
        1,
        2
    ],
    "transpose_backward": [
        0,
        1,
        2
    ],
    "configurations": {
        "2d": {
            "data_identifier": "nnUNetPlans_2d",
            "preprocessor_name": "DefaultPreprocessor",
            "batch_size": 13,
            "patch_size": [
                448,
                448
            ],
            "median_image_size_in_voxels": [
                512.0,
                512.0
            ],
            "spacing": [
                0.9765625,
                0.9765625
            ],
            "normalization_schemes": [
                "CTNormalization"
            ],
            "use_mask_for_norm": [
                False
            ],
            "UNet_class_name": "PlainConvUNet",
            "UNet_base_num_features": 32,
            "n_conv_per_stage_encoder": [
                2,
                2,
                2,
                2,
                2,
                2,
                2
            ],
            "n_conv_per_stage_decoder": [
                2,
                2,
                2,
                2,
                2,
                2
            ],
            "num_pool_per_axis": [
                6,
                6
            ],
            "pool_op_kernel_sizes": [
                [
                    1,
                    1
                ],
                [
                    2,
                    2
                ],
                [
                    2,
                    2
                ],
                [
                    2,
                    2
                ],
                [
                    2,
                    2
                ],
                [
                    2,
                    2
                ],
                [
                    2,
                    2
                ]
            ],
            "conv_kernel_sizes": [
                [
                    3,
                    3
                ],
                [
                    3,
                    3
                ],
                [
                    3,
                    3
                ],
                [
                    3,
                    3
                ],
                [
                    3,
                    3
                ],
                [
                    3,
                    3
                ],
                [
                    3,
                    3
                ]
            ],
            "unet_max_num_features": 512,
            "resampling_fn_data": "resample_data_or_seg_to_shape",
            "resampling_fn_seg": "resample_data_or_seg_to_shape",
            "resampling_fn_data_kwargs": {
                "is_seg": False,
                "order": 3,
                "order_z": 0,
                "force_separate_z": None
            },
            "resampling_fn_seg_kwargs": {
                "is_seg": True,
                "order": 1,
                "order_z": 0,
                "force_separate_z": None
            },
            "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
            "resampling_fn_probabilities_kwargs": {
                "is_seg": False,
                "order": 1,
                "order_z": 0,
                "force_separate_z": None
            },
            "batch_dice": True
        },
        "3d_lowres": {
            "data_identifier": "nnUNetPlans_3d_lowres",
            "preprocessor_name": "DefaultPreprocessor",
            "batch_size": 2,
            "patch_size": [
                192,
                96,
                96
            ],
            "median_image_size_in_voxels": [
                360,
                139,
                139
            ],
            "spacing": [
                5.507178410171863,
                3.5854026107889747,
                3.5854026107889747
            ],
            "normalization_schemes": [
                "CTNormalization"
            ],
            "use_mask_for_norm": [
                False
            ],
            "UNet_class_name": "PlainConvUNet",
            "UNet_base_num_features": 32,
            "n_conv_per_stage_encoder": [
                2,
                2,
                2,
                2,
                2,
                2
            ],
            "n_conv_per_stage_decoder": [
                2,
                2,
                2,
                2,
                2
            ],
            "num_pool_per_axis": [
                5,
                4,
                4
            ],
            "pool_op_kernel_sizes": [
                [
                    1,
                    1,
                    1
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    1,
                    1
                ]
            ],
            "conv_kernel_sizes": [
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ]
            ],
            "unet_max_num_features": 320,
            "resampling_fn_data": "resample_data_or_seg_to_shape",
            "resampling_fn_seg": "resample_data_or_seg_to_shape",
            "resampling_fn_data_kwargs": {
                "is_seg": False,
                "order": 3,
                "order_z": 0,
                "force_separate_z": None
            },
            "resampling_fn_seg_kwargs": {
                "is_seg": True,
                "order": 1,
                "order_z": 0,
                "force_separate_z": None
            },
            "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
            "resampling_fn_probabilities_kwargs": {
                "is_seg": False,
                "order": 1,
                "order_z": 0,
                "force_separate_z": None
            },
            "batch_dice": False,
            "next_stage": "3d_cascade_fullres"
        },
        "3d_fullres": {
            "data_identifier": "nnUNetPlans_3d_fullres",
            "preprocessor_name": "DefaultPreprocessor",
            "batch_size": 2,
            "patch_size": [
                192,
                96,
                96
            ],
            "median_image_size_in_voxels": [
                1323.0,
                512.0,
                512.0
            ],
            "spacing": [
                1.5,
                0.9765625,
                0.9765625
            ],
            "normalization_schemes": [
                "CTNormalization"
            ],
            "use_mask_for_norm": [
                False
            ],
            "UNet_class_name": "PlainConvUNet",
            "UNet_base_num_features": 32,
            "n_conv_per_stage_encoder": [
                2,
                2,
                2,
                2,
                2,
                2
            ],
            "n_conv_per_stage_decoder": [
                2,
                2,
                2,
                2,
                2
            ],
            "num_pool_per_axis": [
                5,
                4,
                4
            ],
            "pool_op_kernel_sizes": [
                [
                    1,
                    1,
                    1
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    2,
                    2
                ],
                [
                    2,
                    1,
                    1
                ]
            ],
            "conv_kernel_sizes": [
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ],
                [
                    3,
                    3,
                    3
                ]
            ],
            "unet_max_num_features": 320,
            "resampling_fn_data": "resample_data_or_seg_to_shape",
            "resampling_fn_seg": "resample_data_or_seg_to_shape",
            "resampling_fn_data_kwargs": {
                "is_seg": False,
                "order": 3,
                "order_z": 0,
                "force_separate_z": None
            },
            "resampling_fn_seg_kwargs": {
                "is_seg": True,
                "order": 1,
                "order_z": 0,
                "force_separate_z": None
            },
            "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
            "resampling_fn_probabilities_kwargs": {
                "is_seg": False,
                "order": 1,
                "order_z": 0,
                "force_separate_z": None
            },
            "batch_dice": True
        },
        "3d_cascade_fullres": {
            "inherits_from": "3d_fullres",
            "previous_stage": "3d_lowres"
        }
    },
    "experiment_planner_used": "ExperimentPlanner",
    "label_manager": "LabelManager",
    "foreground_intensity_properties_per_channel": {
        "0": {
            "max": 46852.0,
            "mean": 445.6136169433594,
            "median": 311.0,
            "min": -1023.0,
            "percentile_00_5": -82.0,
            "percentile_99_5": 1658.0,
            "std": 436.04522705078125
        }
    }
}

labels_dict = {'labels':
    {
        "backgrou`nd": 0,
        "left_kneecap": 1,
        "right_kneecap": 2,
        "left_tibia": 3,
        "left_fibula": 4,
        "left_heel_root": 5,
        "left_heel_mid": 6,
        "left_toe_bone": 7,
        "sternum": 8,
        "skull": 9,
        "right_tibia": 10,
        "right_fibula": 11,
        "vertebrae_C1": 12,
        "vertebrae_C2": 13,
        "vertebrae_C3": 14,
        "vertebrae_C4": 15,
        "vertebrae_C5": 16,
        "vertebrae_C6": 17,
        "vertebrae_C7": 18,
        "vertebrae_T1": 19,
        "vertebrae_T2": 20,
        "vertebrae_T3": 21,
        "vertebrae_T4": 22,
        "vertebrae_T5": 23,
        "vertebrae_T6": 24,
        "vertebrae_T7": 25,
        "vertebrae_T8": 26,
        "vertebrae_T9": 27,
        "vertebrae_T10": 28,
        "vertebrae_T11": 29,
        "vertebrae_T12": 30,
        "vertebrae_L1": 31,
        "vertebrae_L2": 32,
        "vertebrae_L3": 33,
        "vertebrae_L4": 34,
        "vertebrae_L5": 35,
        "right_heel_root": 36,
        "right_heel_mid": 37,
        "right_toe_bone": 38,
        "rib_left_1": 39,
        "rib_left_2": 40,
        "rib_left_3": 41,
        "rib_left_4": 42,
        "rib_left_5": 43,
        "rib_left_6": 44,
        "rib_left_7": 45,
        "rib_left_8": 46,
        "rib_left_9": 47,
        "rib_left_10": 48,
        "rib_left_11": 49,
        "rib_left_12": 50,
        "rib_right_1": 51,
        "rib_right_2": 52,
        "rib_right_3": 53,
        "rib_right_4": 54,
        "rib_right_5": 55,
        "rib_right_6": 56,
        "rib_right_7": 57,
        "rib_right_8": 58,
        "rib_right_9": 59,
        "rib_right_10": 60,
        "rib_right_11": 61,
        "rib_right_12": 62,
        "left_scapula": 63,
        "right_scapula": 64,
        "left_clavicle": 65,
        "right_clavicle": 66,
        "left_thighbone": 67,
        "right_thighbone": 68,
        "left_hip": 69,
        "right_hip": 70,
        "sacrum": 71,
        "humerus____ulna____radius____carpal____metacarpal_bone____phalanx": 72
        }
    }