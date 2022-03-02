import numpy as np
import torch
import torchvision
from tqdm import tqdm

from model import PrototypeNetwork
from utils import ElasticDeform

config = PrototypeNetwork.get_default_config()

main_dataset = torchvision.datasets.MNIST(root='./data', transform=torchvision.transforms.Compose(
    [torchvision.transforms.RandomAffine(degrees=12, translate=(0.05, 0.05), shear=5), torchvision.transforms.ToTensor()]), download=False,
                                          train=True)

train_dataset, valid_dataset = torch.utils.data.random_split(main_dataset, [55_000, 5_000])

test_dataset = torchvision.datasets.MNIST(root='./data', transform=torchvision.transforms.ToTensor(), download=False,
                                          train=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = PrototypeNetwork(config).to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=config['lr'])
num_epochs = 1500
pbar = tqdm(range(num_epochs))
for epoch in pbar:
    total_loss = []
    acc = {'correct': 0,
           'total': 0}
    for batch in train_dataloader:
        network.train()
        optimizer.zero_grad()
        images, labels = batch[0].to(device), batch[1].to(device)
        images = images.float()
        logits, latent_state, reconstruction, prototype_out = network(images)

        loss1 = torch.nn.CrossEntropyLoss()(logits, labels)
        loss2 = torch.nn.MSELoss()(reconstruction, images)
        loss3 = torch.mean(torch.min(prototype_out, 0)[0])
        loss4 = torch.mean(torch.min(prototype_out, 1)[0])
        loss = loss1 + config['lambda'] * loss2 + config['lambda1'] * loss3 + config['lambda2'] * loss4
        loss.backward()
        optimizer.step()
        # print(loss)
        total_loss.append(float(loss.detach()))
        acc['correct'] += torch.sum(torch.max(logits, 1)[1] == labels)
        acc['total'] += len(batch[1])
        # pbar.set_postfix({'total_loss': np.mean(total_loss), 'train_accuracy': float(acc['correct'] / acc['total'])})

    total_valid_loss = []
    valid_acc = {'correct': 0,
                 'total': 0}
    with torch.no_grad():
        for batch in valid_dataloader:
            network.eval()
            images, labels = batch[0].to(device).float(), batch[1].to(device)

            logits, latent_state, reconstruction, prototype_out = network(images)
            # print(reconstruction.shape)

            loss1 = torch.nn.CrossEntropyLoss()(logits, labels)
            loss2 = torch.nn.MSELoss()(reconstruction, images)
            loss3 = torch.mean(torch.min(prototype_out, 0)[0])
            loss4 = torch.mean(torch.min(prototype_out, 1)[0])
            loss = loss1 + config['lambda'] * loss2 + config['lambda1'] * loss3 + config['lambda2'] * loss4
            total_valid_loss.append(float(loss.detach()))
            valid_acc['correct'] += torch.sum(torch.max(logits, 1)[1] == labels)
            valid_acc['total'] += len(batch[1])
        pbar.set_postfix({'total_loss': np.mean(total_loss), 'train_accuracy': float(acc['correct'] / acc['total']),
                          'total_valid_loss': np.mean(total_valid_loss),
                          'valid_accuracy': float(valid_acc['correct'] / valid_acc['total'])})

    if epoch % 30 == 0:
        torch.save(network.state_dict(), './checkpoints/model_{}.pt'.format(epoch))

with torch.no_grad():
    network.eval()
    acc = {'correct': 0,
           'total': 0}
    for batch in test_dataloader:
        images, labels = batch[0].to(device).float(), batch[1].to(device)
        logits, latent_state, reconstruction, prototype_out = network(images)

        acc['correct'] += torch.sum(torch.max(logits, 1)[1] == labels)
        acc['total'] += len(batch[1])

print("Final Test Accuracy", float(acc['correct'] / acc['total']))

torch.save(network.state_dict(), './checkpoints/model_final.pt')
