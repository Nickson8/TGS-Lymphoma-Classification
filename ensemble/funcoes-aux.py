"""
Funcao que auxiliam no calculo das metricas
"""

def calculate_metrics(data):
    """
    Calculates the mean, standard deviation, and 95% confidence interval for a list of 5 numbers.
    """

    # Convert list of Tensors to a list of floats
    data2 = data[:]
    try:
        data = [item.item() for item in data]
    except:
        data = data2[:]
    

    mean = statistics.mean(data)
    sd = statistics.stdev(data)
    n = len(data)
    df = n - 1
    t_stat = stats.t.ppf(0.975, df)
    margin_of_error = t_stat * (sd / math.sqrt(n))
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
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

# --- Função de exemplo (substitua pela sua implementação real) ---
def calculate_metrics(data_list):
    """Calcula a média, desvio padrão e IC de 95% para uma lista de dados."""
    mean = np.mean(data_list)
    sd = np.std(data_list)
    # Cálculo simples de IC, ajuste conforme necessário
    ci_lower = mean - 1.96 * (sd / np.sqrt(len(data_list)))
    ci_upper = mean + 1.96 * (sd / np.sqrt(len(data_list)))
    return {'mean': mean, 'sd': sd, '95%_ci': (ci_lower, ci_upper)}
# ----------------------------------------------------------------

def create_results_document(model_name, metrics, filename="Resultados_Ensemble.docx"):
    """
    Gera um documento Word com uma tabela de resultados formatada em uma única linha de dados.
    """
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