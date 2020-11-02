from torch import nn

class ClsHead(nn.Module):
    def __init__(self, represent_dim=None, num_class=None):
        super(ClsHead, self).__init__()
        self.hidden_dim = represent_dim
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.drop = nn.Dropout(0.2)
        self.cls = nn.Linear(hidden_dim, num_class)
        
    def forward(self, x):
        x = self.gap(x)
        x = x.view(-1, self.hidden_dim)
        # x = torch.flatten(x, 1)
        x = self.fc(x)
        return x