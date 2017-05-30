import models.sda_classifier as sda
import numpy as np
import timeit
import json

model = sda.Classifier(num_input=12, num_units=4, num_steps=3, display=True,
                       batch_size=100, normalize=True)

start = timeit.default_timer()
model.train('./data/keyleak/train.csv')
train_time = timeit.default_timer() - start
start = timeit.default_timer()
print(json.dumps(model.test('./data/keyleak/test.csv'), indent=4))
test_time = timeit.default_timer() - start

print('Train: {:.5f}(s) | Test: {:.5f}(s) | Total: {:.5f}(s)'.format(
    train_time, test_time, train_time + test_time
))
