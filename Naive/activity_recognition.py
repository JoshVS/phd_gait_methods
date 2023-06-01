from dataset import HARDetection
from linear_assignment import LinearAssignmentClassifier

ds = HARDetection(generate_test_video=1)
classifier = LinearAssignmentClassifier(ds, num_dims=2)
classifier.generate_test_set_results()