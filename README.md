# 🖼️ PyTorch Grad-CAM: Visualize Como a IA "Enxerga"

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-F472B6?logo=gradio&logoColor=white&style=for-the-badge)](https://gradio.app)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD166?logo=huggingface&logoColor=black&style=for-the-badge)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> 🔥 **Veja EXATAMENTE onde a IA olhou para tomar decisões** — regiões vermelhas = mais importantes!

[![Demo](https://i.imgur.com/placeholder-gradcam.png)](https://Danielfonseca1212-pytorch-gradcam.hf.space)

🔗 **Experimente online:** https://Danielfonseca1212-pytorch-gradcam.hf.space  
📂 **Repositório:** https://github.com/Danielfonseca1212/pytorch-gradcam

---

## 🎯 O Que é Grad-CAM?

**Grad-CAM** (Gradient-weighted Class Activation Mapping) é uma técnica de **IA Explicável (XAI)** que revela quais partes da imagem mais influenciaram a decisão do modelo:

| Sem Explicabilidade | Com Grad-CAM |
|---------------------|--------------|
| ❓ *"A IA disse que é um gato... mas por quê?"* | ✅ *"A IA viu o **rosto e olhos** (vermelho) para decidir 'gato'!"* |

### Exemplo Real:

Imagem: Gato siamês sentado no sofá
Heatmap: 🔴 Vermelho concentrado NO ROSTO do gato
Interpretação: A IA ignorou o sofá e focou nos features discriminativos!


---

## 🚀 Tecnologias Utilizadas

| Tecnologia | Papel no Projeto |
|------------|------------------|
| **PyTorch** | Framework principal para deep learning |
| **TorchVision** | ResNet18 pré-treinada no ImageNet (1.2M imagens) |
| **Grad-CAM** | Hooks para capturar feature maps + gradientes |
| **OpenCV** | Processamento de imagens e overlays coloridos |
| **Gradio** | Interface web interativa em 50 linhas |
| **Hugging Face Spaces** | Deploy em nuvem com 1 clique |

---

## 📊 Pipeline Completo

```mermaid
flowchart TD
    A[Upload de Imagem] --> B[Pré-processamento<br>Resize + Normalize]
    B --> C[Forward Pass<br>ResNet18]
    C --> D{Classe Predita?}
    D -->|Sim| E[Backward Pass<br>Gradientes da classe alvo]
    E --> F[Grad-CAM<br>Média ponderada de feature maps]
    F --> G[Heatmap Colorido<br>Vermelho = importante]
    G --> H[Visualização<br>Original \| Heatmap \| Overlay]

# 1. Clonar repositório
git clone https://github.com/Danielfonseca1212/pytorch-gradcam.git
cd pytorch-gradcam

# 2. Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar app
python app.py

 Abra o link http://127.0.0.1:7860 no navegador e comece a explorar!

📁 Estrutura do Projeto

pytorch-gradcam/
├── app.py              # Interface Gradio com 2 abas (Classificação + Grad-CAM)
├── requirements.txt    # Dependências mínimas e compatíveis
└── README.md           # Este arquivo 😎

🧠 Conceitos Técnicos Demonstrados
✅ Transfer Learning — Uso de ResNet18 pré-treinada no ImageNet
✅ Hooks em PyTorch — Captura de feature maps (forward_hook) e gradientes (backward_hook)
✅ Backpropagation Seletiva — Gradientes apenas para a classe predita
✅ Interpretabilidade (XAI) — Tornar decisões da IA transparentes e auditáveis
✅ Processamento de Imagens — OpenCV para overlays profissionais
✅ MLOps Básico — Deploy em nuvem com Gradio + Hugging Face

🔗 Links Diretos
Plataforma
Link
App Online
https://Danielfonseca1212-pytorch-gradcam.hf.space
Hugging Face
https://huggingface.co/spaces/Danielfonseca1212/pytorch-gradcam
GitHub
https://github.com/Danielfonseca1212/pytorch-gradcam
