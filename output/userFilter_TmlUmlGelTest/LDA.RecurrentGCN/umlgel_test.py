import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

# read uml files
doc = pd.read_csv('uml/documents.csv')
graphs = torch.load('uml/graphs/graphs.pt')
features = np.load('uml/user_interests/features.npy')

# read gel files
pred_features = pd.read_csv('gel/userFeatures.csv')
embs = torch.load('gel/embeddings.pt')

# First Evaluation: comparing argmax values between embeddings and features
embargs, f1args, f2args, f3args = [], [], [], []
for e in embs: embargs.append(e.argmax().item())
for f1 in features[0]: f1args.append(f1.argmax())
for f2 in features[1]: f2args.append(f2.argmax())
for f3 in features[2]: f3args.append(f3.argmax())

equal_count = sum(x == y for x, y in zip(embargs, f1args))
print(f'Number of equal argmax values between embeddings and features[0]: {equal_count}')
equal_count = sum(x == y for x, y in zip(embargs, f2args))
print(f'Number of equal argmax values between embeddings and features[1]: {equal_count}')
equal_count = sum(x == y for x, y in zip(embargs, f3args))
print(f'Number of equal argmax values between embeddings and features[2]: {equal_count}')

# Second Evaluation: comparing user similarities for each feature set and embedding
similarity_feature_prediction = cosine_similarity(embs, embs)
similarity_training1 = cosine_similarity(features[0], features[0])
similarity_training2 = cosine_similarity(features[1], features[1])
similarity_training3 = cosine_similarity(features[2], features[2])
mae_comparison_1 = np.mean(np.abs(similarity_feature_prediction - similarity_training1))
mae_comparison_2 = np.mean(np.abs(similarity_feature_prediction - similarity_training2))
mae_comparison_3 = np.mean(np.abs(similarity_feature_prediction - similarity_training3))
mae_comparison_4 = np.mean(np.abs(similarity_training1 - similarity_training2))
mae_comparison_5 = np.mean(np.abs(similarity_training1 - similarity_training3))
mae_comparison_6 = np.mean(np.abs(similarity_training2 - similarity_training3))

print(f'MAE Comparison with P - F1: {mae_comparison_1.item()}')
print(f'MAE Comparison with P - F2: {mae_comparison_2.item()}')
print(f'MAE Comparison with P - F3: {mae_comparison_3.item()}')
print(f'MAE Comparison with F1 - F2: {mae_comparison_4.item()}')
print(f'MAE Comparison with F1 - F3: {mae_comparison_5.item()}')
print(f'MAE Comparison with F2 - F3: {mae_comparison_6.item()}')