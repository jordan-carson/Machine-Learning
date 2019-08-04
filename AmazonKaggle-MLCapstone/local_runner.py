"""

    Local Runner:

        pass in parameters for the model, it will assign them to a data-type based on its contents.

"""
from amazon_feed import AmazonProcessor


params = {}
params['main'] = '/Users/jordancarson/PyCharmProjects/AmazonKaggle-MLCapstone'
params['train_path'] = 'resources/train-jpg/'
params['test_path'] = 'resources/test-jpg/'
params['train_labels'] = 'resources/train_v2.csv'
params['submission_file'] = 'resources/sample_submission_v2.csv'

params['write_submission_file'] = True
params['resize_image'] = (64, 64)
params['nfolds'] = 3
params['batch_size'] = 128

obj = AmazonProcessor(**params)
obj.run()