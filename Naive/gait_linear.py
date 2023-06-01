from dataset import NaiveKinectDataset
from linear_assignment import LinearAssignmentClassifier


ds = NaiveKinectDataset(max_samples=20, num_dims=2)
classifier = LinearAssignmentClassifier(ds, num_dims=2)
classifier.generate_test_set_results()