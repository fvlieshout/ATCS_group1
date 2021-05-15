import random
from collections import defaultdict
import nltk
from nltk.corpus import reuters
from data_prep.data import Data

class ReutersData(Data):
    def __init__(self, r8=False, val_size=0.1):
        docs, unique_cls = self.prepare_reuters(r8=r8, val_size=val_size)
        train_docs, test_docs, val_docs = docs

        self.train = self._prepare_split(train_docs, unique_cls)
        self.test = self._prepare_split(test_docs, unique_cls)
        self.val = self._prepare_split(val_docs, unique_cls)

        self.classes = unique_cls

    @staticmethod
    def prepare_reuters(r8=False, val_size=0.1):
        """
        Download the dataset and filters out all documents which have more or less than 1 class.
        Then filters out all classes which have no remaining documents.
        Args:
            r8 (bool, optional): R8 is constructed by taking only the top 10 (original) classes. Defaults to False.
            val_size (float, optional): Proportion of training documents to include in the validation set.
        Returns:
            doc_splits (tuple): Tuple containing 3 List of training, test, and validation documents.
            unique_classes (List): List of Strings containing the class names sorted in alphabetical order.
        """
        nltk.download('reuters')
        # Filter out docs which don't have exactly 1 class
        data = defaultdict(lambda: {'train': [], 'test': []})

        for doc in reuters.fileids():
            categories = reuters.categories(doc)
            if len(categories) == 1:
                if doc.startswith('training'):
                    data[categories[0]]['train'].append(doc)
                if doc.startswith('test'):
                    data[categories[0]]['test'].append(doc)

        # Filter out classes which have no remaining docs
        for cat in reuters.categories():
            if len(data[cat]['train']) < 1 or len(data[cat]['test']) < 1:
                data.pop(cat, None)

        if r8:
            # Choose top 10 classes and then select the ones which still remain after filtering
            popular = sorted(reuters.categories(), key=lambda clz: len(reuters.fileids(clz)), reverse=True)[:10]
            data = dict([(cls, splits) for (cls, splits) in data.items() if cls in popular])

        # Create splits
        train_docs = [doc for cls, splits in data.items() for doc in splits['train']]
        test_docs = [doc for cls, splits in data.items() for doc in splits['test']]

        # Select the validation documents out of the training documents
        val_size = int(len(train_docs) * val_size)
        random.shuffle(train_docs)
        val_docs = train_docs[:val_size]
        train_docs = train_docs[val_size:]

        # sort the unique classes to ensure constant order
        unique_classes = sorted(data.keys())
        return (train_docs, test_docs, val_docs), unique_classes
    
    @staticmethod
    def _prepare_split(docs, classes):
        """
        Extract the text and labels of each documents.
        Args:
            docs (List): List of document names from which the texts and labels will be extracted.
            classes (List): List of unique class names sorted in alphabetical order.
        Returns:
            texts (List): List of Strings containing the text of the documents.
            labels (List): List of int containing the labels of the documents.
        """
        texts = []
        labels = []
        for doc in docs:
            text = ' '.join(reuters.words(doc))
            clz = reuters.categories(doc)[0]
            texts.append(text)
            labels.append(classes.index(clz))

        return texts, labels
    
    @property
    def train_data(self):
        """
        Get the training data as a tuple containing 2 elements:
            texts (List): List of training document contents as string.
            labels (List): List of training label ids as integer.
        """
        return self.train
    
    @property
    def test_data(self):
        """
        Get the test data as a tuple containing 2 elements:
            texts (List): List of test document contents as string.
            labels (List): List of test label ids as integer.
        """
        return self.test
    
    @property
    def val_data(self):
        """
        Get the val data as a tuple containing 2 elements:
            texts (List): List of val document contents as string.
            labels (List): List of val label ids as integer.
        """
        return self.val
    
    @property
    def num_classes(self):
        """
        Get the number of unique classes in the dataset.
        """
        return len(self.classes)

class R52Data(ReutersData):
    """
    Wrapper for the R52 dataset.
    """
    def __init__(self, val_size=0.1):
        super().__init__(r8=False, val_size=val_size)

class R8Data(ReutersData):
    """
    Wrapper for the R8 dataset.
    """
    def __init__(self, val_size=0.1):
        super().__init__(r8=True, val_size=val_size)
