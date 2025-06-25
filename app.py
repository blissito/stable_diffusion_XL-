import gradio as gr
from diffusers import StableDiffusionXLPipeline
import torch
import tempfile
import threading
import time
import os

# Variables globales
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = None
model_loaded = False
progress_data = {"step": 0, "total_steps": 0, "status": "idle"}

# Información del dispositivo GPU
gpu_info = ""
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    gpu_info = f"GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)"
else:
    gpu_info = "GPU: No disponible (usando CPU)"

def load_model_in_background():
    """Carga el modelo Stable Diffusion XL optimizado para Fly.io"""
    global pipe, model_loaded
    
    try:
        print(f"Cargando modelo SDXL optimizado en {device.upper()}...")
        
        # Configuración más estable para evitar errores de meta tensor
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
            low_cpu_mem_usage=False  # Cambiar a False para evitar meta tensors
        )
        
        if device == "cuda":
            # Optimizaciones más conservadoras
            pipe.enable_attention_slicing(1)
            pipe.enable_vae_slicing()
            
            # Limpiar cache de CUDA
            torch.cuda.empty_cache()
        else:
            pipe = pipe.to(device)
            
        # Verificar que el modelo se cargó correctamente
        if pipe is None:
            raise Exception("El modelo no se pudo cargar")
            
        model_loaded = True
        print(f"✅ Modelo SDXL cargado exitosamente en {device.upper()}!")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        model_loaded = False

def progress_callback(step, timestep, latents):
    """Callback para actualizar el progreso"""
    global progress_data
    progress_data["step"] = step
    progress_data["total_steps"] = progress_data.get("total_steps", 15)
    progress_data["status"] = "generating"

def generate_image_file(prompt, num_steps=15, width=1024, height=1024, guidance_scale=7.0):
    """Genera una imagen con SDXL ultra-optimizado para Fly.io"""
    global pipe, model_loaded, progress_data
    
    if not model_loaded or pipe is None:
        return None, "⏳ El modelo aún se está cargando. Por favor espera unos minutos..."
    
    try:
        print(f"Generando imagen SDXL para: {prompt}")
        progress_data["status"] = "starting"
        progress_data["total_steps"] = num_steps
        progress_data["step"] = 0
        
        # Generar semilla aleatoria para variedad
        generator = torch.Generator("cpu").manual_seed(int(time.time()))
        
        # Configuración simplificada para evitar meta tensors
        result = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
            callback=progress_callback,
            callback_steps=1
        )
        
        # Verificar que el resultado sea válido
        if not result.images or len(result.images) == 0:
            raise Exception("No se generó ninguna imagen")
            
        image = result.images[0]
        
        # Verificar que la imagen sea válida
        if image is None:
            raise Exception("La imagen generada es None")
    
        progress_data["status"] = "completed"
        
        # Limpiar cache después de generar
        if device == "cuda":
            torch.cuda.empty_cache()
        
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(temp_file.name, optimize=True)
        temp_file.close()
        return temp_file.name, "✅ Imagen generada exitosamente!"
        
    except Exception as e:
        progress_data["status"] = "error"
        print(f"Error: {e}")
        if device == "cuda":
            torch.cuda.empty_cache()
        return None, f"❌ Error: {str(e)}"

# Crear la interfaz
with gr.Blocks(title="SDXL Ultra-Optimizado - Fly.io") as demo:
    gr.Markdown("# 🚀 Stable Diffusion XL Ultra-Optimizado")
    gr.Markdown(f"**Dispositivo:** {device.upper()}")
    gr.Markdown(f"**{gpu_info}**")
    gr.Markdown("**Optimizado para:** Velocidad máxima en Fly.io")
    
    # Status del modelo
    status_text = gr.Markdown("⏳ **Estado del modelo:** Cargando...")
    
    def update_status():
        if model_loaded:
            return "✅ **Estado del modelo:** ¡Listo para usar!"
        else:
            return "⏳ **Estado del modelo:** Descargando SDXL optimizado..."
    
    gr.Markdown("---")
    gr.Markdown("### ⚡ **Configuración Ultra-Rápida**")
    gr.Markdown("- **15 pasos** (más rápido que 20)")
    gr.Markdown("- **Guidance 7.0** (balance perfecto)")
    gr.Markdown("- **1024x1024** (tamaño óptimo)")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt", 
                placeholder="A kawaii cat professor wearing glasses, academic robe, and graduation cap, high quality, detailed",
                lines=3
            )
            steps_slider = gr.Slider(
                minimum=10, 
                maximum=30, 
                value=15, 
                step=1, 
                label="Pasos de inferencia (15 = ultra-rápido)"
            )
            guidance_slider = gr.Slider(
                minimum=1.0,
                maximum=15.0,
                value=7.0,
                step=0.1,
                label="Guidance Scale (7.0 = balance perfecto)"
            )
            width_slider = gr.Slider(
                minimum=512,
                maximum=1024,
                value=1024,
                step=64,
                label="Ancho (1024 recomendado)"
            )
            height_slider = gr.Slider(
                minimum=512,
                maximum=1024,
                value=1024,
                step=64,
                label="Alto (1024 recomendado)"
            )
            generate_btn = gr.Button("⚡ Generar Imagen Ultra-Rápida", variant="primary")
        
        with gr.Column():
            output_file = gr.File(label="Descargar Imagen Generada")
            message_output = gr.Textbox(label="Estado", interactive=False)
    
    generate_btn.click(
        fn=generate_image_file,
        inputs=[prompt_input, steps_slider, width_slider, height_slider, guidance_slider],
        outputs=[output_file, message_output]
    )
    
    # Actualizar status al cargar la página
    demo.load(update_status, outputs=status_text)

if __name__ == "__main__":
    print(f"🚀 Iniciando SDXL Ultra-Optimizado en {device.upper()}")
    
    # Iniciar carga del modelo en segundo plano
    model_thread = threading.Thread(target=load_model_in_background)
    model_thread.daemon = True
    model_thread.start()
    
    # Iniciar Gradio inmediatamente
    print("🌐 Iniciando interfaz web optimizada...")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False,
        show_api=True
    ) 