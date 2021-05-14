import torch
from HDGCN import HDGCN
from utils import DatasetLoader, accuracy
from torch.utils.data import DataLoader
from adabelief_pytorch import AdaBelief
import torch.nn.functional as F

#
# Settings.
#

torch.cuda.set_device(4)
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
# load datasets
#
batch_size = 32
bert_dim = 300
train_data = DatasetLoader('mr', set_name="train")
vocab = train_data.vocab
test_data = DatasetLoader('mr', set_name="test")
max_seq_len = train_data.nnodes
train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
nhid = 300
vote_dim = 100
nclass = train_data.nclass()
input_cols = ['node_embeddings', 'dependency_graph', 'polarity']

#
# Create capsule network.
#

network = HDGCN(nnodes=max_seq_len,
                nfeat=bert_dim,
                nhid=nhid,
                nclass=nclass,
                max_seq_len=max_seq_len,
                device=device,
                batch_size=batch_size,
                vocab=vocab).to(device)

optimizer = AdaBelief(network.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9, 0.999),
                      weight_decouple=True, rectify=False)


# Converts batches of class indices to classes of one-hot vectors.
def to_one_hot(x, length):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = 1.0
    return x_one_hot


def test():
    network.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_data_loader):
            inputs = [sample_batched[col].to(device) for col in input_cols]
            sentences = sample_batched['sentence']
            output = network(inputs[0], inputs[1], sentences, batch=i_batch, vis=False)
            test_loss += float(F.nll_loss(output, inputs[2].max(1, keepdim=False)[1]).item())
            total_samples += len(output)
            correct += accuracy(output, inputs[2].max(1, keepdim=False)[1])

        test_loss /= (i_batch + 1)
        correct = correct.item()
        correct /= (i_batch + 1)
    return correct


def train(epoch):
    train_loss = 0.
    correct = 0.
    total_samples = 0
    network.train()
    for i_batch, sample_batched in enumerate(train_data_loader):
        inputs = [sample_batched[col].to(device) for col in input_cols]
        sentences = sample_batched['sentence']
        optimizer.zero_grad()
        if epoch % 5 == 0:
            visualize = True
        else:
            visualize = False
        output = network(inputs[0], inputs[1], sentences, batch=i_batch, vis=visualize)
        # loss = network.loss(output, inputs[2])
        loss = F.nll_loss(output, inputs[2].max(1, keepdim=False)[1])
        loss.backward()
        train_loss += float(loss.item())
        optimizer.step()
        total_samples += len(output)
        correct += accuracy(output, inputs[2].max(1, keepdim=False)[1])

    correct = correct.item()
    correct /= (i_batch + 1)

    train_loss /= (i_batch + 1)
    print('Train Epoch: {} \t Loss: {:.6f}, \t Accuracy: {:.6f}'.format(epoch, train_loss, correct))
    return train_loss


num_epochs = 1000
best_correct = 0.0
corrects = []
for epoch in range(1, num_epochs + 1):
    print('training epochs: {}'.format(epoch))
    train_loss = train(epoch)
    correct = test()
    corrects.append(correct)
    if correct > best_correct:
        best_correct = correct
    print("best correct: {:.6f} \n".format(best_correct))
    print('>' * 100)
