def load_model(modelo, path):
    model = nn.DataParallel(timm.create_model(modelo, pretrained=False, num_classes=1))

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    return model, checkpoint['test_idxs']



"""
Funcao que auxiliam no calculo das metricas
"""

def calculate_metrics(data, n_resamples=1000):
    """
    Calcula a média, desvio padrão (amostral) e o 
    IC de 95% usando o método percentile bootstrap.
    """

    # 1. Converte Tensors para floats, se necessário (do seu código original)
    data2 = data[:]
    try:
        data = [item.item() for item in data]
    except Exception:
        data = data2[:]
    

    # 2. Calcula a média e o desvio padrão dos 5 folds originais
    # A média e o DP reportados são os das métricas observadas.
    mean = statistics.mean(data)
    sd = statistics.stdev(data) # Desvio padrão amostral (ddof=1)
    
    # --- Início do Bootstrap ---
    
    # 3. Gera as médias das reamostragens
    bootstrapped_means = []
    n_size = len(data) # Tamanho da amostra original (5)

    for _ in range(n_resamples):
        # Sorteia 'n_size' amostras COM REPOSIÇÃO
        resample = np.random.choice(data, size=n_size, replace=True)
        
        # Calcula a média da reamostra
        boot_mean = np.mean(resample)
        bootstrapped_means.append(boot_mean)

    # 4. Calcula o Intervalo de Confiança (Percentil)
    # Para um IC de 95%, queremos os percentis 2.5 e 97.5
    
    alpha = 0.05 # Para 95% de confiança
    
    # np.percentile calcula o valor abaixo do qual 'q'% dos dados caem
    ci_lower = np.percentile(bootstrapped_means, (alpha / 2.0) * 100)
    ci_upper = np.percentile(bootstrapped_means, (1 - alpha / 2.0) * 100)
    
    # --- Fim do Bootstrap ---

    return {
        "mean": mean,
        "sd": sd,
        "95%_ci": (ci_lower, ci_upper)
    }

def format_metric(metric_data):
    """Formata os dados da métrica na string 'Média ± DP [95% CI]'."""
    mean = metric_data['mean']
    sd = metric_data['sd']
    ci_lower, ci_upper = metric_data['95%_ci']
    # Usando uma quebra de linha para melhor legibilidade na célula da tabela do Word
    return f"{mean:.2f} ± {sd:.3f}\n[{ci_lower:.2f}, {ci_upper:.2f}]"


# ----------------------------------------------------------------

def create_results_document(model_name, metrics):
    """
    Gera um documento Word com uma tabela de resultados formatada em uma única linha de dados.
    """

    filename = f"Resultados_{model_name}.docx"

    doc = Document()

    # Adiciona um título ao documento
    doc.add_heading(f'Resultados de Performance: {model_name}', level=1)
    doc.add_paragraph() # Adiciona um pequeno espaço

    # Define os cabeçalhos da tabela
    headers = [
        'Modelos e classes', 'Acurácia', 'Sensibilidade (Recall)', 'Especificidade',
        'Precisão', 'F1-score', 'AUC-ROC', 'AUPRC'
    ]

    # Cria a tabela (2 linhas: 1 cabeçalho, 1 linha de dados)
    table = doc.add_table(rows=2, cols=8)
    table.style = 'Table Grid'

    # --- Preenche a Linha do Cabeçalho ---
    header_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        cell = header_cells[i]
        cell.text = header
        # Deixa o texto do cabeçalho em negrito e centralizado
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # --- Função auxiliar para preencher uma célula ---
    def populate_cell(col_idx, text):
        # A linha de dados é sempre a linha 1 (a segunda linha)
        row_idx = 1
        cell = table.cell(row_idx, col_idx)
        cell.text = text
        # Alinha todo o conteúdo da célula vertical e horizontalmente
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.font.size = Pt(10)

    # --- Combina os dados para a única linha de dados ---
    model_col_text = f"{model_name}\nTGS\nLinfoma"
    accuracy_col_text = f"{format_metric(calculate_metrics(metrics['accuracy']))}\n-\n-"
    sensitivity_col_text = f"-\n{format_metric(calculate_metrics(metrics['TGS']['recall']))}\n{format_metric(calculate_metrics(metrics['Linfoma']['recall']))}"
    specificity_col_text = f"-\n{format_metric(calculate_metrics(metrics['TGS']['specificity']))}\n{format_metric(calculate_metrics(metrics['Linfoma']['specificity']))}"
    precision_col_text = f"-\n{format_metric(calculate_metrics(metrics['TGS']['precision']))}\n{format_metric(calculate_metrics(metrics['Linfoma']['precision']))}"
    f1_score_col_text = f"-\n{format_metric(calculate_metrics(metrics['TGS']['f1-score']))}\n{format_metric(calculate_metrics(metrics['Linfoma']['f1-score']))}"
    auc_roc_col_text = f"{format_metric(calculate_metrics(metrics['AUC']))}\n-\n-"
    auprc_col_text = f"{format_metric(calculate_metrics(metrics['AUPRC']))}\n-\n-"

    # --- Preenche a única linha de dados ---
    populate_cell(0, model_col_text)
    # Deixa o nome do modelo em negrito
    table.cell(1, 0).paragraphs[0].runs[0].font.bold = True

    populate_cell(1, accuracy_col_text)
    populate_cell(2, sensitivity_col_text)
    populate_cell(3, specificity_col_text)
    populate_cell(4, precision_col_text)
    populate_cell(5, f1_score_col_text)
    populate_cell(6, auc_roc_col_text)
    populate_cell(7, auprc_col_text)

    # Salva o documento
    doc.save(filename)
    print(f"Documento Word criado com sucesso: {filename}")