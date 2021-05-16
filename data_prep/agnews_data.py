from data_prep.data import HuggingFaceData


class AGNewsData(HuggingFaceData):
    """
    Wrapper for the IMDb dataset.
    """

    def __init__(self, val_size=0.1):
        super().__init__(dataset_name="ag_news", val_size=val_size)
