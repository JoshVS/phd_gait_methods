from dataset_loaders.casia_dataset import CASIADataset
from classifiers.gcn_classifier import GCNClassifier

ds = CASIADataset(generate_test_video=None, max_samples=None, test_split=0.05)

classifier = GCNClassifier(ds)
classifier.generate_test_set_results()