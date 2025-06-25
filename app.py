import gradio as gr
from diffusers import StableDiffusionXLPipeline
import torch
import tempfile
import threading
import time
import os
import boto3
from datetime import datetime
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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

# Configurar S3 usando variables de entorno de Fly.io
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'auto'),
    endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3')
)

# Configuración del bucket S3
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Función para generar URLs públicas usando el endpoint de Fly.io
def get_public_url(bucket_name, key):
    """Genera una URL pública para un objeto en S3 de Fly.io"""
    endpoint = os.getenv('AWS_ENDPOINT_URL_S3')
    if not endpoint:
        raise ValueError("AWS_ENDPOINT_URL_S3 no está configurado")
    
    # Extraer el dominio base del endpoint
    base_domain = endpoint.replace('https://', '').replace('http://', '')
    # Remover cualquier prefijo como 'fly.storage'
    base_domain = base_domain.split('.')[-2] + '.' + base_domain.split('.')[-1]
    
    return f"https://{bucket_name}.{base_domain}/{key}"

def load_model_in_background():
    """Carga el modelo Stable Diffusion XL de forma simple"""
    global pipe, model_loaded
    
    try:
        print(f"Cargando modelo SDXL en {device.upper()}...")
        
        # Configuración simple y estable
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        
        if device == "cuda":
            pipe = pipe.to(device)
            # Solo optimización básica
            pipe.enable_attention_slicing(1)
        else:
            pipe = pipe.to(device)
            
        model_loaded = True
        print(f"✅ Modelo SDXL cargado exitosamente en {device.upper()}!")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        model_loaded = False

def progress_callback(step, timestep, latents):
    """Callback para actualizar el progreso"""
    global progress_data
    progress_data["step"] = step
    progress_data["total_steps"] = progress_data.get("total_steps", 10)
    progress_data["status"] = "generating"

def upload_to_s3(image_path):
    """Sube un archivo de imagen a S3 y devuelve la URL pública"""
    try:
        # Generar nombre único para el archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_part = uuid.uuid4().hex[:6]
        image_name = f"sdxl_{timestamp}_{random_part}.png"
        
        # Configurar políticas públicas
        bucket_policy = {
            'Version': '2012-10-17',
            'Statement': [{
                'Sid': 'PublicReadGetObject',
                'Effect': 'Allow',
                'Principal': '*',
                'Action': 's3:GetObject',
                'Resource': f'arn:aws:s3:::{S3_BUCKET_NAME}/*'
            }]
        }
        
        # Subir a S3 con ACL public-read
        s3_client.upload_file(
            image_path,
            S3_BUCKET_NAME,
            image_name,
            ExtraArgs={'ACL': 'public-read', 'ContentType': 'image/png'}
        )
        
        # Obtener URL pública
        url = get_public_url(S3_BUCKET_NAME, image_name)
        return url
    except Exception as e:
        print(f"Error subiendo a S3: {e}")
        return None


def generate_image_file(prompt, num_steps=10, width=1024, height=1024, guidance_scale=7.0):
    """Genera una imagen con SDXL de forma simple (para la interfaz de Gradio)"""
    global pipe, model_loaded, progress_data
    
    if not model_loaded or pipe is None:
        return None, "⏳ El modelo aún se está cargando. Por favor espera unos minutos..."
    
    try:
        print(f"Generando imagen SDXL para: {prompt}")
        progress_data["status"] = "starting"
        progress_data["total_steps"] = num_steps
        progress_data["step"] = 0
        
        # Generar semilla aleatoria para variedad
        generator = torch.Generator(device).manual_seed(int(time.time()))
        
        result = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            callback=progress_callback,
            callback_steps=1
        )
        
        image = result.images[0]
        
        # Verificar que la imagen sea válida
        if image is None:
            raise Exception("La imagen generada es None")
    
        progress_data["status"] = "completed"
        
        # Guardar temporalmente la imagen
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(temp_file.name, optimize=True)
        
        # Subir a S3 y obtener URL
        url = upload_to_s3(temp_file.name)
        if url:
            return url, "✅ Imagen generada y subida a S3 exitosamente!"
        else:
            return None, "❌ Error subiendo la imagen a S3"
            
    except Exception as e:
        progress_data["status"] = "error"
        print(f"Error: {e}")
        return None, f"❌ Error: {str(e)}"

def generate_image_api(prompt, num_steps=10, width=1024, height=1024, guidance_scale=7.0):
    """Genera una imagen con SDXL para la API REST"""
    try:
        # Llamar a la función principal
        url, message = generate_image_file(prompt, num_steps, width, height, guidance_scale)
        
        if url:
            return {
                "success": True,
                "url": url,
                "message": message,
                "metadata": {
                    "prompt": prompt,
                    "steps": num_steps,
                    "width": width,
                    "height": height,
                    "guidance_scale": guidance_scale
                }
            }
        else:
            return {
                "success": False,
                "error": message
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Crear la interfaz
with gr.Blocks(title="SDXL Simple - Fly.io") as demo:
    gr.Markdown("# 🚀 Stable Diffusion XL Simple")
    gr.Markdown(f"**Dispositivo:** {device.upper()}")
    gr.Markdown(f"**{gpu_info}**")
    gr.Markdown("**Configuración:** Simple y estable")
    
    # Status del modelo
    status_text = gr.Markdown("⏳ **Estado del modelo:** Cargando...")
    
    def update_status():
        if model_loaded:
            return "✅ **Estado del modelo:** ¡Listo para usar!"
        else:
            return "⏳ **Estado del modelo:** Descargando SDXL..."
    
    gr.Markdown("---")
    gr.Markdown("### ⚡ **Configuración Simple y Rápida**")
    gr.Markdown("- **10 pasos** (más rápido)")
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
                minimum=5, 
                maximum=20, 
                value=10, 
                step=1, 
                label="Pasos de inferencia (10 = rápido)"
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
            generate_btn = gr.Button("⚡ Generar Imagen", variant="primary")
        
        with gr.Column():
            output_url = gr.Textbox(label="URL de la Imagen", interactive=False)
            output_image = gr.Image(label="Imagen Generada", interactive=False)
            download_btn = gr.Button("Descargar Imagen", variant="secondary")
            message_output = gr.Textbox(label="Estado", interactive=False)
    
    # Manejar la generación de imagen
    generate_btn.click(
        fn=generate_image_file,
        inputs=[prompt_input, steps_slider, width_slider, height_slider, guidance_slider],
        outputs=[output_url, message_output]
    )
    
    # Manejar el botón de descarga
    def download_image(url):
        if url:
            return url
        return None
    
    download_btn.click(
        fn=download_image,
        inputs=[output_url],
        outputs=[output_image]
    )
    
    # Actualizar status al cargar la página
    demo.load(update_status, outputs=status_text)

if __name__ == "__main__":
    print(f"🚀 Iniciando SDXL Simple en {device.upper()}")
    
    # Iniciar carga del modelo en segundo plano
    model_thread = threading.Thread(target=load_model_in_background)
    model_thread.daemon = True
    model_thread.start()
    
    # Iniciar Gradio inmediatamente
    print("🌐 Iniciando interfaz web...")
    demo.api_route = "/api/predict"
    demo.api_app = FastAPI()
    
    @demo.api_app.post("/api/predict")
    async def predict(request: Request):
        body = await request.json()
        prompt = body.get("data", [None])[0]
        if not prompt:
            return JSONResponse({
                "success": False,
                "error": "Prompt is required"
            }, status_code=400)
            
        # Obtener parámetros opcionales
        num_steps = body.get("data", [None, 10])[1]
        width = body.get("data", [None, None, 1024])[2]
        height = body.get("data", [None, None, None, 1024])[3]
        guidance_scale = body.get("data", [None, None, None, None, 7.0])[4]
        
        # Generar la imagen usando la función API
        result = generate_image_api(prompt, num_steps, width, height, guidance_scale)
        return JSONResponse(result)
    
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False,
        show_api=False
    ) 