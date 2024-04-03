import torch
import torchvision
from torch.utils.data import DataLoader
from model import SimpleModel

### --- YOUR CODE SHOULD WORK WITH THE UNMODIFIED VERSION OF THE FOLLOWING CODE --- ###
### --- Coded with assistnance from Github co-pilot -- ###

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

def main():

    # load dataset
    train_dataset = torchvision.datasets.FashionMNIST(root='./data/', train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data/', train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # train using warmup model
    model = SimpleModel().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Start training using {DEVICE} ...")    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch in train_loader:
            img, label = batch
            out = model(img.to(DEVICE))
            loss = torch.nn.functional.cross_entropy(out, label.to(DEVICE))
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: total loss - {total_loss:.4f}")


    model.eval()  
    num_correct = 0
    num_total = 0
    for batch in test_loader:
        img, label = batch
        out = model(img.to(DEVICE))
        pred_label = torch.argmax(out, dim=1)
        num_correct += torch.sum(pred_label == label.to(DEVICE)).item()
        num_total += len(label)

    assert num_total == 10000, f"number of test samples is incorrect got {num_total} expected 10000"
    accuracy = num_correct / num_total

    print(f"final accuracy: {accuracy}")
    assert accuracy > 0.85, f"accuracy: {accuracy}, This is lower than expect. The model is likely incorrect." 


if __name__ == "__main__":
    main()
