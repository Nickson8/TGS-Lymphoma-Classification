"""
Rede do Ensemble
"""
class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__()


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.backbone_names = ["Swin-B", "ViT-B", "Densenet", "Resnet"]

        #Feature extractors
        self.feature_extractors = [timm.create_model("swin_base_patch4_window7_224",pretrained=True,num_classes=0),
        timm.create_model("timm/vit_base_patch32_clip_448.laion2b_ft_in12k_in1k",pretrained=True,num_classes=0),
        timm.create_model("timm/densenet121.ra_in1k",pretrained=True,num_classes=0),
        timm.create_model("timm/resnet50.a1_in1k",pretrained=True,num_classes=0)]

        if torch.cuda.device_count() > 1:
            self.feature_extractors = [nn.DataParallel(modelo) for modelo in self.feature_extractors]
        
        self.feature_extractors = [x.to(self.device) for x in self.feature_extractors]

        #Classification Head
        self.layers = nn.Sequential(
            # First block
            nn.Linear(4864, 512),
            nn.BatchNorm1d(512),
            nn.GELU(), # A smooth, modern activation function
            nn.Dropout(0.5),
            
            # Output layer
            nn.Linear(512, 1)
        ).to(self.device)



        self.transforms = [ transforms.Resize( (m.module.default_cfg["input_size"][1], m.module.default_cfg["input_size"][2]) ) for m in self.feature_extractors]
        

    def forward(self, x):
        #Outputs dos feature extractors
        output_vit = self.feature_extractors[1](self.transforms[1](x))
        with torch.no_grad():
            output_swin = self.feature_extractors[0](self.transforms[0](x))
            output_desenet = self.feature_extractors[2](self.transforms[2](x))
            output_resnet = self.feature_extractors[3](self.transforms[3](x))

        meta_features = torch.cat([output_swin, output_vit, output_desenet, output_resnet], dim=1)

        return self.layers(meta_features) #Retorna logits

    @classmethod
    def load_ensemble(cls, path: str, metricas: dict = None):
        """
        Loads a model, reconstructing it from the saved configuration in the checkpoint file.
        """
        print(f"➡️ Loading model from {path}...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        
        # 1. Recreate the model from the saved configuration
        config = checkpoint['config']

        test_idxs = checkpoint['test_idxs']
        
        if metricas is None:
            print("Warning: No 'metricas' dictionary provided. Using an empty one.")
            metricas = {}

        # Create an instance of the class with the correct architecture
        instance = cls()
        
        # 2. Load the state dictionaries
        instance.layers.load_state_dict(checkpoint['ensemble_head_state_dict'])
        
        for i, name in enumerate(instance.backbone_names):
            # The model inside the instance might be wrapped in DataParallel
            model_to_load = instance.feature_extractors[i]
            if isinstance(model_to_load, nn.DataParallel):
                model_to_load.module.load_state_dict(checkpoint['backbones_state_dict'][name])
            else:
                model_to_load.load_state_dict(checkpoint['backbones_state_dict'][name])
            
        # 3. Move to device and set to evaluation mode
        instance.to(device)
        instance.eval()
        
        print("Model loaded successfully.")
        
        # Optionally load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            # You would need to re-create the optimizer first, then load its state
            print("Optimizer state found in checkpoint. Re-create optimizer and call .load_state_dict() to resume training.")
        
        return instance, test_idxs