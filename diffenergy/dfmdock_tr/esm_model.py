import torch
import torch.nn as nn
import esm


class ESMLanguageModel(nn.Module):
    def __init__(
        self, 
        version: str = 'ESM2',
        rep_layer: int = 33,
        ):
        super().__init__()
        self.rep_layer = rep_layer
        self.model, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
        self.model.eval()
        
    def forward(self, x):
        for param in self.model.parameters():
            param.requires_grad = False

        results = self.model(x, repr_layers = [self.rep_layer])
        token_repr = results["representations"][self.rep_layer]

        return token_repr[:, 1:-1, :]

if __name__ == '__main__':
    model = ESMLanguageModel()
    inputs = torch.randint(0, 21, (1, 10))
    output = model(inputs)
    print(output)
