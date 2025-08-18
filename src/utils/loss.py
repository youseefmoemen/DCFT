import torch.nn as nn



class DCFTLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.basic_loss = nn.CrossEntropyLoss()


    def forward(self, model, logits, labels):
        vanilla_loss = self.basic_loss(logits, labels)
        total_orthogonal_loss = 0.0
        for name, param in model.named_parameters():
            if 'DCFT' in name:
                total_orthogonal_loss += param.get_orthogonal_loss()

        return vanilla_loss + self.alpha * total_orthogonal_loss
    