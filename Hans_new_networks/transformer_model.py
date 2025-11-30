import numpy as np

def data_input(npz_file_path="training_data_1.npz"):
	"""
	Load training data from training_data_1.npz and return x_train, y_train
	"""
	data = np.load(npz_file_path)
	x_train = data["x_train"]
	y_train = data["y_train"]
	return x_train, y_train


class TransformerModel:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def get_data_shapes(self):
        return self.x_train.shape, self.y_train.shape