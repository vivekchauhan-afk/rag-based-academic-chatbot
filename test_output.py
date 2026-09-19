import pickle

with open("vector_db/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(metadata[0])