

"""
Classe que treina e escreve todo o relatorio dos resultados de um modelo
dado o dataset

@param modelo -> modelo baseado em pytorch
"""
class Total_Writer_Ensemble:
    def __init__(self, modelo, batch_size, lr):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelo = modelo
        self.batch_size = batch_size
        self.lr = lr





    """
    =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    """





    def treina(self, X, y, epochs):
        """
        Funcao que treina 5 modelos baseados em um 5-Fold Cross Validation
        e salva eles na memoria
        """
        kf = StratifiedKFold(n_splits=5, shuffle=True)


        for idk, (train_idxs, test_idxs) in enumerate(kf.split(X, y)):
            train_idxs = train_idxs.tolist()
            test_idxs = test_idxs.tolist()
            self.test_idxs = test_idxs

            
            print(f"\n\n\n\n\n\n\n\n\n\n\n\n***\n***\n{idk+1}° Fold\n***\n***\n")
            #Treino é separado entre treino e val
            random.shuffle(train_idxs)
            train_idxs_2 = train_idxs[: int(0.9*len(train_idxs))]
            val_idxs = train_idxs[int(0.9*len(train_idxs)) :]


            train_loader = DataLoader(Modelo_Ensemble_Dataset([X[i] for i in train_idxs_2], [y[i] for i in train_idxs_2], is_training=True),
                                           batch_size=self.batch_size, shuffle=True,
                                           pin_memory=True, num_workers=2)
            
            val_loader = DataLoader(Modelo_Ensemble_Dataset([X[i] for i in val_idxs], [y[i] for i in val_idxs]),
                                           batch_size=self.batch_size, shuffle=False,
                                           pin_memory=True, num_workers=2)


            self.fit(epochs=epochs, lr=self.lr,
                       train_loader = train_loader,
                       val_loader = val_loader)

            self.save_ensemble(f"ensemble_fold_{idk+1}.pth")
    












    def fit(self, epochs, lr, train_loader, val_loader):
        # --- 3. Create the parameter groups for the optimizer ---
        param_groups = [
            # Group 1: The ensemble head with a high learning rate
            {'params': self.modelo.layers.parameters(), 'lr': lr},
            
            # Group 2: The ViT backbone with a very low learning rate
            {'params': self.modelo.feature_extractors[1].parameters(), 'lr': 2e-6}
        ]
        optimizer = AdamW(param_groups, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=4,
        )


        #Loss function
        criterion = nn.BCEWithLogitsLoss()

        
        cont = 0
        s = nn.Sigmoid()
        val_losses = [999999.9]

        #Training loop
        for epoch in range(epochs):
            
            #Treinando com os batchs
            train_loss = 0.0
            for inputs, labels in train_loader:
                # Move data to GPU
                inputs = inputs.to(self.device)
                # print("Input batch shape:", inputs.shape)
                labels = labels.float().to(self.device)

                
                optimizer.zero_grad()
                
                # Forward pass (modelo tem logits como output)
                outputs = self.modelo(inputs) #[batch_size, 1]
                outputs = outputs.view(-1) #[batch_size]

                loss = criterion(outputs, labels)
                train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()

            
            #Validaçao
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.float().to(self.device)
                    
                    outputs = self.modelo(inputs)
                    outputs = outputs.view(-1)

                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    outputs = s(outputs)
                    predicted = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            if(epoch%5==0):
                print(f'---Epoch {epoch+1}\nLoss do Treino: {train_loss/len(train_loader):.4f}\n' +
                  f'Loss da Validaçao: {val_loss/len(val_loader):.4f}, Accuracy: {100*correct/total:.2f}%, lr: {optimizer.param_groups[0]["lr"]:.9f}')

            scheduler.step(val_loss)
            
            val_losses.append(val_loss/len(val_loader))

            if(val_losses[-1] > val_losses[-2]):
                cont += 1
            else:
                cont = 0
            if(cont >= 8):
                print("####### Validação estagnada, parando o treinamento #######")












    def save_ensemble(self, path: str, epoch: int = None):
        """
        Saves the complete model state and configuration to a single file.
        """
        print(f"✅ Saving model to {path}...")
        
        # Handle DataParallel by accessing the .module attribute
        backbone_state_dicts = {}
        for name, model in zip(self.modelo.backbone_names, self.modelo.feature_extractors):
            if isinstance(model, nn.DataParallel):
                backbone_state_dicts[name] = model.module.state_dict()
            else:
                backbone_state_dicts[name] = model.state_dict()

        checkpoint = {
            'config': {
                'backbone_names': self.modelo.backbone_names,
                'head_input_features': self.modelo.layers[0].in_features,
                'head_hidden_dim': self.modelo.layers[0].out_features,
                'head_num_classes': self.modelo.layers[-1].out_features,
            },
            'ensemble_head_state_dict': self.modelo.layers.state_dict(),
            'backbones_state_dict': backbone_state_dicts,
            'test_idxs': self.test_idxs
        }

        # Optionally save optimizer and epoch for resuming training
        if hasattr(self.modelo, 'optimizer') and self.modelo.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.modelo.optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
            
        torch.save(checkpoint, path)
        print("Model saved successfully.")











    """
    =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    """











    def gera_relatorios(self, X, y):
        ensemble_metricas = {
            'TGS': {'precision': [], 'recall': [], 'specificity': [], 'f1-score': [], 'support': []},
            'Linfoma': {'precision': [], 'recall': [], 'specificity': [], 'f1-score': [], 'support': []},
            'accuracy': [],
            'AUC': [],
            'AUPRC': []
        }

        modelos = [x for x in os.listdir("working") if "fold" in x]

        for j, modelo in enumerate(modelos):
            rede, test_idxs = MetaLearner.load_ensemble(modelo)

            test_loader = DataLoader(Modelo_Ensemble_Dataset([X[i] for i in test_idxs], [y[i] for i in test_idxs]),
                                       batch_size=self.batch_size, shuffle=False,
                                       pin_memory=True, num_workers=2)

            self.teste(rede, ensemble_metricas, test_loader)

            self.gera_heatmaps(rede, X, test_idxs, j)

            self.gera_matrizes_de_conf(rede ,test_loader, j)
        

        create_results_document("Ensemble Model", ensemble_metricas)
    








    def teste(self, rede, ensemble_metricas, test_loader):
        # Fazendo predicoes
        all_preds = []
        all_probs = []
        all_labels = []
        
        s = nn.Sigmoid()
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.float().to(self.device)
                
                # Forward pass (modelo tem logits como output)
                outputs = rede(inputs) #[batch_size, 1]
                outputs = outputs.view(-1) #[batch_size]
                outputs = s(outputs)
                
                predicted = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calcula métricas detalhadas
        report_ensemble = classification_report(
            all_labels, 
            all_preds,
            target_names=["TGS", "Linfoma"],
            output_dict=True
        )
        
        auc = binary_auroc(torch.tensor(all_probs), torch.tensor(all_labels))
        auprc = binary_auprc(torch.tensor(all_probs), torch.tensor(all_labels))
        
        print(f"Accuracy: {100*report_ensemble['accuracy']:.2f}%")
        print("\nClassification Report:")
        print(classification_report(
            all_labels, 
            all_preds,
            target_names=["TGS", "Linfoma"]
        ))
        
        print(f"AUC: {auc:.2f}, AUPRC: {auprc:.2f}")

        report_ensemble["AUC"] = auc
        report_ensemble["AUPRC"] = auprc

        report_ensemble['TGS']['specificity'] = report_ensemble['Linfoma']['recall']
        report_ensemble['Linfoma']['specificity'] = report_ensemble['TGS']['recall']

        #-----Adicionando as metricas do fold atual-----
        for key in ensemble_metricas.keys():
            if key in report_ensemble:
                if isinstance(report_ensemble[key], dict):
                    for sub_key in report_ensemble[key].keys():
                        ensemble_metricas[key][sub_key].append(report_ensemble[key][sub_key])
                else:
                    ensemble_metricas[key].append(report_ensemble[key])






    def gera_heatmaps(self, ensemble, X, test_idxs, fold):

        swin = ensemble.feature_extractors[0]
        vit = ensemble.feature_extractors[1]
        densenet = ensemble.feature_extractors[2]
        resnet = ensemble.feature_extractors[3]
        
        # --- Define correct target layers for each model ---
        # For Transformers, we target the normalization layer of the last block.
        # For CNNs, we target the final convolutional layer or block.
        swin_target_layers = [swin.module.layers[-1].blocks[-1].norm2]
        vit_target_layers = [vit.module.blocks[-1].norm1]
        densenet_target_layers = [densenet.module.features[-1]] # Or densenet.features[-1] for older timm versions
        resnet_target_layers = [resnet.module.layer4[-1]]

        imagens_brutas = [X[i] for i in test_idxs]
        images_heat = []
        cont = 1
        
        for img in imagens_brutas:
            
            # --- 1. Generate individual heatmaps ---
            cam_swin = generate_grayscale_cam(swin, swin_target_layers, img, reshape_transform=reshape_transform_swin)
            
            cam_vit = generate_grayscale_cam(vit, vit_target_layers, img, reshape_transform=reshape_transform_vit, size=(448,448))
            cam_vit = cv2.resize(cam_vit, (224,224), interpolation=cv2.INTER_AREA)
            
            cam_densenet = generate_grayscale_cam(densenet, densenet_target_layers, img)
            
            cam_resnet = generate_grayscale_cam(resnet, resnet_target_layers, img)
            
            # --- 2. Average the heatmaps ---
            # Now we include all four models in the average calculation.
            all_cams = [cam_swin, cam_vit, cam_densenet, cam_resnet]
            mean_cam = np.mean(all_cams, axis=0)
            
            # --- 3. Prepare the original image for visualization ---
            img_size = (224, 224) # Ensure consistent size
            pil_img = img.copy()
            rgb_img_float = np.array(pil_img.resize(img_size), dtype=np.float32) / 255.0
            
            # --- 4. Overlay the averaged heatmap and plot ---
            mean_cam_image = show_cam_on_image(rgb_img_float, mean_cam, use_rgb=True, image_weight=0.6)

            images_heat.append(mean_cam_image)

            print(cont, end=' ')
            cont += 1

        #Fazendo o zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for i, heatmap_array in enumerate(images_heat):
                # Convert NumPy array to a PIL Image
                image = Image.fromarray(heatmap_array)
                
                # Create an in-memory buffer for the PNG image
                image_buffer = io.BytesIO()
                image.save(image_buffer, format='PNG')
                
                # Write the image buffer's content to the zip file
                file_name = f"heatmap_{i+1}.png"
                zip_file.writestr(file_name, image_buffer.getvalue())
        
        # --- 3. Save the Zip File to Disk ---
        zip_file_name = f"working/heatmaps/heatmaps_fold_{fold}.zip"
        with open(zip_file_name, "wb") as f:
            f.write(zip_buffer.getvalue())
        
        print(f"'\n{zip_file_name}' created successfully!")




    def gera_matrizes_de_conf(self, rede ,test_loader, fold):

        # Fazendo predicoes
        all_preds = []
        all_labels = []
        
        s = nn.Sigmoid()
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.float().to(self.device)
                
                # Forward pass (modelo tem logits como output)
                outputs = rede(inputs) #[batch_size, 1]
                outputs = outputs.view(-1) #[batch_size]
                outputs = s(outputs)
                
                predicted = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())


        #Matriz de confusão
        cm = confusion_matrix(all_labels, all_preds)
        class_names = ["TGS", "Linfoma"] # Seus nomes de classe
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, annot_kws={'size': 20})
        plt.title('Ensemble')
        plt.ylabel('Classe Verdadeira')
        plt.xlabel('Classe Predita')
        plt.savefig(f'/working/conf_matriz/matriz_fold_{fold}.tiff', dpi=600)