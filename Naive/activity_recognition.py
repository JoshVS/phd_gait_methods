from dataset import HARDetection
from linear_assignment import LinearAssignmentClassifier

ds = HARDetection(generate_test_video=None)
classifier = LinearAssignmentClassifier(ds, num_dims=2, num_phases=1)
classifier.generate_test_set_results()