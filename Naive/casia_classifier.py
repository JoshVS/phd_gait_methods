from dataset_loaders.casia_dataset import CASIADataset
from classifiers.gcn_classifier import GCNClassifier

ds = CASIADataset(generate_test_video=None, max_samples=4)

classifier = GCNClassifier(ds)
classifier.generate_test_set_results()