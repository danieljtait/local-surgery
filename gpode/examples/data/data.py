import bessel


class DataLoader:

    @classmethod
    def load(example, seed, times, noise_scales, *args, **kwargs):
        if example == "bessel jn":
            return bessel.bessel_jn_data(seed, times, **kwargs)
