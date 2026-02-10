"""
Classificador PyTorch com Grad-CAM
Deploy permanente no Hugging Face Spaces
"""

import torch
import torchvision
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np
import cv2

# ============================================
# 1. Carregar modelo ResNet18
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    # Torchvision >= 0.13
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    labels = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
except:
    # Torchvision < 0.13 (fallback)
    model = torchvision.models.resnet18(pretrained=True)
    labels = ["tench", "goldfish", "great white shark", "tiger", "zebra", 
              "dog", "cat", "elephant", "bear", "giraffe", "fox", "wolf", 
              "horse", "cow", "sheep", "airplane", "car", "truck", "motorcycle", "bicycle"]

model = model.to(device)
model.eval()

# ============================================
# 2. Pré-processamento
# ============================================
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================
# 3. Classe GradCAM
# ============================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        
        # Registrar hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Captura feature maps na forward pass"""
        self.feature_maps = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Captura gradientes na backward pass"""
        self.gradients = grad_output[0].detach()
    
    def __call__(self, input_tensor, class_idx=None):
        """
        Gera heatmap Grad-CAM
        
        Args:
            input_tensor: Tensor de entrada [1, C, H, W]
            class_idx: Índice da classe (None = classe com maior probabilidade)
        
        Returns:
            numpy array: Heatmap normalizado [H, W]
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Determinar classe alvo
        if class_idx is None:
            class_idx = output.argmax().item()
        
        # Backward pass
        score = output[0, class_idx]
        score.backward()
        
        # Calcular pesos (média dos gradientes)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Pesos aplicados às feature maps
        for i in range(self.feature_maps.shape[1]):
            self.feature_maps[:, i, :, :] *= pooled_gradients[i]
        
        # Heatmap (média ponderada das feature maps)
        heatmap = torch.mean(self.feature_maps, dim=1).squeeze()
        heatmap = torch.nn.functional.relu(heatmap)  # Apenas ativações positivas
        
        # Normalizar 0-1
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        return heatmap.cpu().numpy()

# Criar instância GradCAM
gradcam = GradCAM(model, target_layer=model.layer4[-1])

# ============================================
# 4. Funções auxiliares
# ============================================
def overlay_heatmap(heatmap, original_image, alpha=0.6):
    """
    Sobrepor heatmap colorido na imagem original
    
    Args:
        heatmap: Heatmap numpy [H, W] (0-1)
        original_image: Imagem PIL ou numpy
        alpha: Transparência do heatmap (0.0 a 1.0)
    
    Returns:
        PIL.Image: Imagem com overlay
    """
    # Converter para numpy se necessário
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # Redimensionar heatmap para tamanho da imagem
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Aplicar colormap (vermelho/amarelo)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Converter original para RGB se necessário
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    # Sobrepor
    overlay = alpha * heatmap_colored + (1 - alpha) * original_image
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return Image.fromarray(overlay)

def visualize_cam(original_image, heatmap, overlay):
    """
    Criar visualização lado a lado: original | heatmap | overlay
    
    Args:
        original_image: PIL.Image
        heatmap: Heatmap numpy
        overlay: PIL.Image com overlay
    
    Returns:
        PIL.Image: Imagem combinada
    """
    # Redimensionar tudo para mesmo tamanho
    size = (300, 300)
    original_resized = original_image.resize(size)
    overlay_resized = overlay.resize(size)
    
    # Converter heatmap para imagem PIL
    heatmap_resized = cv2.resize(heatmap, size)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), 
        cv2.COLORMAP_JET
    )
    heatmap_pil = Image.fromarray(
        cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    )
    
    # Criar imagem lado a lado
    combined = Image.new('RGB', (size[0] * 3, size[1]))
    combined.paste(original_resized, (0, 0))
    combined.paste(heatmap_pil, (size[0], 0))
    combined.paste(overlay_resized, (size[0] * 2, 0))
    
    return combined

# ============================================
# 5. Funções da interface
# ============================================
def predict(image):
    """Classificação simples (aba 1)"""
    if image is None:
        return "⚠️ Envie uma imagem primeiro!"
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img_t = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_t)
    
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top3_prob, top3_catid = torch.topk(probs, 3)
    
    result = "🏆 Top 3 previsões:\n\n"
    for i in range(3):
        prob = float(top3_prob[i]) * 100
        label = labels[int(top3_catid[i])]
        result += f"{i+1}. {label:<25} → {prob:.1f}%\n"
    
    return result

def gradcam_visualization(image):
    """Gera visualização Grad-CAM (aba 2)"""
    if image is None:
        return None, "⚠️ Envie uma imagem primeiro!"
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Salvar imagem original redimensionada
    original_resized = image.resize((224, 224))
    
    # Pré-processar para o modelo
    img_t = preprocess(image).unsqueeze(0).to(device)
    
    # Classificar primeiro
    with torch.no_grad():
        output = model(img_t)
    
    pred_idx = output.argmax().item()
    pred_label = labels[pred_idx]
    confidence = float(torch.nn.functional.softmax(output[0], dim=0)[pred_idx]) * 100
    
    # Gerar heatmap
    heatmap = gradcam(img_t, class_idx=pred_idx)
    
    # Criar overlay
    overlay = overlay_heatmap(heatmap, original_resized, alpha=0.6)
    
    # Criar visualização lado a lado
    combined = visualize_cam(original_resized, heatmap, overlay)
    
    # Texto explicativo
    explanation = f"""
    🎯 **Predição:** {pred_label}
    💯 **Confiança:** {confidence:.1f}%
    
    🔴 **Vermelho/Amarelo** = Áreas que MAIS influenciaram a decisão da IA
    🔵 **Azul** = Áreas menos relevantes
    
    A IA "olhou" principalmente para as regiões quentes para identificar o objeto!
    """
    
    return combined, explanation

# ============================================
# 6. Interface Gradio com abas
# ============================================
with gr.Blocks(title="PyTorch Grad-CAM") as demo:
    gr.Markdown("# 🖼️ Classificador PyTorch com Grad-CAM")
    gr.Markdown("### Veja EXATAMENTE onde a IA 'olhou' para tomar decisões!")
    
    with gr.Tabs():
        # ABA 1: Classificação
        with gr.TabItem("🔍 Classificação"):
            gr.Markdown("### Classifique qualquer imagem")
            
            with gr.Row():
                with gr.Column():
                    img_input_class = gr.Image(type="pil", label="📸 Envie uma imagem")
                    btn_class = gr.Button("Classificar", variant="primary")
                
                with gr.Column():
                    txt_output = gr.Textbox(label="🤖 Resultado", lines=8)
            
            btn_class.click(predict, inputs=img_input_class, outputs=txt_output)
        
        # ABA 2: Grad-CAM
        with gr.TabItem("🎨 Grad-CAM (Heatmap)"):
            gr.Markdown("### Visualize onde a IA 'olhou' para decidir!")
            
            with gr.Row():
                with gr.Column():
                    img_input_cam = gr.Image(type="pil", label="📸 Envie uma imagem")
                    btn_cam = gr.Button("Gerar Heatmap 🔥", variant="primary")
                
                with gr.Column():
                    img_output_cam = gr.Image(label="📊 Visualização: Original | Heatmap | Overlay")
                    txt_explanation = gr.Markdown()
            
            btn_cam.click(
                gradcam_visualization, 
                inputs=img_input_cam, 
                outputs=[img_output_cam, txt_explanation]
            )
    
    gr.Markdown("---")
    gr.Markdown("**🔗 Link permanente:** https://huggingface.co/spaces/Danielfonseca1212/pytorch-gradcam")
    gr.Markdown("**📂 Código:** https://github.com/Danielfonseca1212/pytorch-gradcam")

# Executar
if __name__ == "__main__":
    demo.launch()
