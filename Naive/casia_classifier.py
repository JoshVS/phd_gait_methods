from dataset_loaders.casia_dataset import CASIADataset
from classifiers.linear_assignment import LinearAssignmentClassifier

ds = CASIADataset(generate_test_video=2, max_samples=255)
classifier = LinearAssignmentClassifier(ds, num_dims=2, num_phases=1)
classifier.generate_test_set_results()