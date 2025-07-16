from dataset_loaders.naive_kinect_dataset import HARDetection
from classifiers.linear_assignment import LinearAssignmentClassifier

ds = HARDetection(generate_test_video=None)
classifier = LinearAssignmentClassifier(ds, num_dims=2, num_phases=1)
classifier.generate_test_set_results()