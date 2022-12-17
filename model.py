import torch

class LeNet5(torch.nn.Module):
    # constructor
    def __init__(self, numCategory):
        super().__init__()
        #                           input channel, no. filters
        self.conv1 = torch.nn.Conv2d(3, 20, kernel_size=(3,3), padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d((2,2))
        self.conv2 = torch.nn.Conv2d(20, 50, kernel_size=(3,3), padding=1)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(156800, 100)
        self.linear2 = torch.nn.Linear(100, numCategory)

    # take x (input image) and output score vector
    def forward(self, x):
        feature_map = self.maxpool(self.relu(self.conv1(x)))
        feature_map = self.maxpool(self.relu(self.conv2(feature_map)))

        flattened = self.flatten(feature_map)
        h = self.relu(self.linear1(flattened))
        score = self.linear2(h)

        return score

if __name__ == '__main__':
    model = LeNet5(11)
    # one sample, RGB, height, width
    random_noise = torch.randn(2, 3, 224, 224)
    output = model(random_noise)
    print(output.shape)
