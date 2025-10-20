

"""
Classe que treina e escreve todo o relatorio dos resultados de um modelo
dado o dataset

"""
class Total_Writer_Ind:
    def __init__(self, modelo, nome, batch_size, lr):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelo = modelo
        self.batch_size = batch_size
        self.lr = lr
        self.H = self.modelo.H
        self.W = self.modelo.W
        self.nome = self.modelo.name
        self.nome_bonito = nome





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


            train_loader = DataLoader(Model_Dataset([X[i] for i in train_idxs_2], [y[i] for i in train_idxs_2], is_training=True, input_size=(self.H, self.W)),
                                           batch_size=self.batch_size, shuffle=True,
                                           pin_memory=True, num_workers=2)
            
            val_loader = DataLoader(Model_Dataset([X[i] for i in val_idxs], [y[i] for i in val_idxs], input_size=(self.H, self.W)),
                                           batch_size=self.batch_size, shuffle=False,
                                           pin_memory=True, num_workers=2)


            self.fit(epochs=epochs,
                       train_loader = train_loader,
                       val_loader = val_loader)

            self.save_model(f"modelo_fold_{idk+1}.pth")
    






    def get_optimizer(self):
        if "vit" in self.nome.lower():
            # Case 1: ViT model → all parameters trainable
            optimizer = AdamW(self.modelo.model.parameters(), lr=2e-6)

        else:
            # Case 2: Non-ViT model → freeze everything
            for name, param in self.modelo.model.named_parameters():
                param.requires_grad = False

            if("densenet" in self.nome.lower()):
                self.modelo.model.module.classifier = nn.Linear(self.modelo.model.module.classifier.in_features, 1).to(self.device)
            elif("resnet" in self.nome.lower()):
                self.modelo.model.module.fc = nn.Linear(self.modelo.model.module.fc.in_features, 1).to(self.device)
            elif("vit" in self.nome.lower()):
                self.modelo.model.module.head = nn.Linear(self.modelo.model.module.head.in_features, 1).to(self.device)
            
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.modelo.model.parameters()), 
                lr=self.lr
            )

        return optimizer



    def fit(self, epochs, train_loader, val_loader):

        optimizer = self.get_optimizer()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
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
                break
    




    def save_model(self, path):
        torch.save({
            'model_state_dict': self.modelo.model.state_dict(),
            'test_idxs': self.test_idxs
        }, path)











    """
    =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    """











    def gera_relatorios(self, X, y):
        metricas_modelo = {
            'TGS': {'precision': [], 'recall': [], 'specificity': [], 'f1-score': [], 'support': []},
            'Linfoma': {'precision': [], 'recall': [], 'specificity': [], 'f1-score': [], 'support': []},
            'accuracy': [],
            'AUC': [],
            'AUPRC': []
        }

        modelos = [x for x in os.listdir("working") if "fold" in x]

        for j, modelo in enumerate(modelos):
            rede, test_idxs = load_model(self.nome, modelo)

            test_loader = DataLoader(Model_Dataset([X[i] for i in test_idxs], [y[i] for i in test_idxs], input_size=(self.H, self.W)),
                                       batch_size=self.batch_size, shuffle=False,
                                       pin_memory=True, num_workers=2)

            self.teste(rede, metricas_modelo, test_loader)

            self.gera_matrizes_de_conf(rede ,test_loader, j)
        

        create_results_document(self.nome_bonito, metricas_modelo)
    




    def teste(self, rede, metricas_modelo, test_loader):
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
        report_modelo = classification_report(
            all_labels, 
            all_preds,
            target_names=["TGS", "Linfoma"],
            output_dict=True
        )
        
        auc = binary_auroc(torch.tensor(all_probs), torch.tensor(all_labels))
        auprc = binary_auprc(torch.tensor(all_probs), torch.tensor(all_labels))
        
        print(f"Accuracy: {100*report_modelo['accuracy']:.2f}%")
        print("\nClassification Report:")
        print(classification_report(
            all_labels, 
            all_preds,
            target_names=["TGS", "Linfoma"]
        ))
        
        print(f"AUC: {auc:.2f}, AUPRC: {auprc:.2f}")

        report_modelo["AUC"] = auc
        report_modelo["AUPRC"] = auprc

        report_modelo['TGS']['specificity'] = report_modelo['Linfoma']['recall']
        report_modelo['Linfoma']['specificity'] = report_modelo['TGS']['recall']

        #-----Adicionando as metricas do fold atual-----
        for key in metricas_modelo.keys():
            if key in report_modelo:
                if isinstance(report_modelo[key], dict):
                    for sub_key in report_modelo[key].keys():
                        metricas_modelo[key][sub_key].append(report_modelo[key][sub_key])
                else:
                    metricas_modelo[key].append(report_modelo[key])
    



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
        plt.title(self.nome_bonito)
        plt.ylabel('Classe Verdadeira')
        plt.xlabel('Classe Predita')
        plt.savefig(f'working/conf_matriz/matriz_fold_{fold+1}.tiff', dpi=600)