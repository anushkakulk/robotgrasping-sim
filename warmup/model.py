import torch

class SimpleModel(torch.nn.Module):
    def __init__(self, img_size=(28, 28), label_num=10):
        super().__init__()

        self.label_num = label_num
        
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(), # default is not true for relu  
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # max pool
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # i added another conv and relu layer to the recommneded model
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            torch.nn.ReLU(),  
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * (img_size[0] // 8) * (img_size[1] // 8), label_num),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
