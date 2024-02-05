import pickle, os
import pandas as pd
import torch
import torch.optim as optim

import params


def modelTrain(dataset):
    if params.gel['method'].lower()=='gconvgru':
        import GConvGRU
        model = GConvGRU.model(node_features=params.tml['numTopics'], filters=params.gel['embeddingDim'])
    elif params.gel['method'].lower()=='a3tgcn':
        import A3TGCN
        model = A3TGCN.model(node_features=params.tml['numTopics'], periods=12)#filters=params.gel['embeddingDim'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    model.train()
    from tqdm import tqdm
    for epoch in tqdm(range(params.gel['epoch'])):
        loss = 0
        step = 0
        for time, snapshot in enumerate(dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            # cost = torch.mean((y_hat-snapshot.y)**2)
            loss = loss + torch.mean((y_hat - snapshot.y) ** 2)
            step += 1
        loss = loss / (step + 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))
    model.eval()
    predictions = []
    labels = []
    cost = 0
    for time, snapshot in enumerate(dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
        print(cost)
        labels.append(snapshot.y)
        predictions.append(y_hat)
    cost = cost / (time+1)
    cost = cost.item()
    print("MSE: {:.4f}".format(cost))
    return model

def main(documents, dataset):
    if not os.path.isdir(params.gel["path2save"]): os.makedirs(params.gel["path2save"])
    if params.gel['method'].lower() == 'random':
        predicted_features = torch.rand(dataset.features[0].shape)
    elif params.gel['method'].lower() in ['GConvGRU', 'A3TGCN']:
        model = modelTrain(dataset)
        model.eval()
        with torch.no_grad():
            predicted_features = model(dataset[-1].y, dataset[-1].edge_index, dataset[-1].edge_weight)
        # print(predicted_features)
        # Thresholding
        predicted_features = torch.where(predicted_features < 0.05, torch.tensor(0), predicted_features)

    else:
        predicted_features = torch.rand(dataset.features[0].shape)

    with open(f'{params.gel["path2save"]}/embeddings.pkl', 'wb') as f:
        pickle.dump(predicted_features, f)
    torch.save(predicted_features, f'{params.gel["path2save"]}/embeddings.pt')
    user_features = pd.DataFrame({'UserId': documents['UserId'].unique(), 'FinalInterests': predicted_features.tolist()})
    user_features.to_csv(f'{params.gel["path2save"]}/userFeatures.csv')
    return user_features, predicted_features
