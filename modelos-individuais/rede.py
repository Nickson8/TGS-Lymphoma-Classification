"""
Rede do Modelo
"""
class Medic_Model(nn.Module):
    def __init__(self, modelo):
        super(Medic_Model, self).__init__()

        #Apenas extrai as features
        self.model = timm.create_model(modelo,pretrained=True,num_classes=1)
        self.name = modelo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        _, self.H, self.W = self.model.default_cfg["input_size"]

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.device)
    
    def forward(self, x):
        return self.model(x)