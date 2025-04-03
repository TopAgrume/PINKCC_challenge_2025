import albumentations

albumentations.RandomSizedCrop: [
    RandomSizedCropLowerBound,
    RandomSizedCropUpperBound,
],
albumentations.HorizontalFlip: [HorizontalFlipBound],
albumentations.VerticalFlip: [VerticalFlipBound],
albumentations.RandomRotate90: [
    RandomRotate90Bound,
    RandomRotate180Bound,
    RandomRotate270Bound,
],
albumentations.MultiplicativeNoise: [
    MultiplicativeNoiseLowerBound,
    MultiplicativeNoiseUpperBound,
],
albumentations.GaussianBlur: [
    GaussianBlurLowerBound,
    GaussianBlurUpperBound,
],
ElasticTransform: [
    ElasticTransformLowerBound,
    ElasticTransformUpperBound,
],
RandomBrightness: [
    RandomBrightnessLowerBound,
    RandomBrightnessUpperBound,
],
RandomContrast: [
    RandomContrastLowerBound,
    RandomContrastUpperBound,
],
RandomGammaMultiChannel: [
    RandomGammaLowerBound,
    RandomGammaUpperBound,
]