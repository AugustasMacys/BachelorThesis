import pandas as pd

from utilities import VALIDATION_LABELS


labels_validation = pd.read_csv(VALIDATION_LABELS)
real_len = len(labels_validation[labels_validation["label"] == 1])
fake_len = len(labels_validation) - real_len

print("Real Validation Videos: {}\nFake Validation Videos: {}".format(real_len, fake_len))
