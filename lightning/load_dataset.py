import pickle

from dataset import apply_transform, apply_padding

with open("lightning/AA_sample_dataset.pkl", "rb") as file:
    dataset = pickle.load(file)

transformed_data = apply_transform(dataset)
padded_data = apply_padding(transformed_data)


print(type(padded_data))