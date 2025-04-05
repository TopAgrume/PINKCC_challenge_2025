import albumentations as A


class DataAugmentation:
    @classmethod
    def get_augmentation_pipeline(cls,
                                    p=0.5,
                                    random_crop_scale=(0.85, 1.0),
                                    h_flip_prob=0.5,
                                    v_flip_prob=0.5,
                                    rotate_prob=0.5,
                                    mult_noise_range=(0.9, 1.1),
                                    blur_range=(0.5, 1.5),
                                    elastic_alpha_range=(1, 2),
                                    brightness_range=(-0.1, 0.1),
                                    contrast_range=(0.9, 1.1),
                                    gamma_range=(0.9, 1.1)):
        """
        Create an albumentations augmentation pipeline for medical images

        Args:
            p: Overall probability of applying augmentations
            random_crop_scale: Scale range for random crops
            h_flip_prob: Probability of horizontal flip
            v_flip_prob: Probability of vertical flip
            rotate_prob: Probability of 90° rotation
            mult_noise_range: Range for multiplicative noise
            blur_range: Range for Gaussian blur sigma
            elastic_alpha_range: Range for elastic transforms alpha
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            gamma_range: Range for gamma adjustment

        Returns:
            albumentations.Compose pipeline
        """
        return A.Compose([ # TODO: find good values
            #A.RandomSizedCrop(
            #    p=p
            #),
            A.HorizontalFlip(p=h_flip_prob),
            A.VerticalFlip(p=v_flip_prob),
            A.RandomRotate90(p=rotate_prob),
            #A.MultiplicativeNoise(
            #    p=p
            #),
            #A.GaussianBlur(
            #    p=p
            #),
            #A.ElasticTransform(
            #    p=p
            #),
            #A.RandomBrightnessContrast(
            #    p=p
            #),
            #A.RandomGamma(
            #    p=p
            #)
        ])#, additional_targets={'mask': 'image'})
