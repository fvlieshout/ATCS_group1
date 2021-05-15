from data_prep.data import HuggingFaceData

class IMDbData(HuggingFaceData):
    """
    Wrapper for the IMDb dataset.
    """
    def __init__(self, val_size=0.1):
        super().__init__(dataset_name="imdb", val_size=val_size)
