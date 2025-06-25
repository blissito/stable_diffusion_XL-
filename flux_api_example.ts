// Ejemplo de cliente TypeScript para consumir la API de Stable Diffusion XL

interface SDXLRequest {
  prompt: string;
  num_steps?: number;
  width?: number;
  height?: number;
  guidance_scale?: number;
}

interface SDXLResponse {
  image: string; // base64 encoded image
  message: string;
}

class SDXLClient {
  private baseUrl: string;

  constructor(baseUrl: string = "https://flux-fixter.fly.dev") {
    this.baseUrl = baseUrl;
  }

  async generateImage(request: SDXLRequest): Promise<SDXLResponse> {
    const response = await fetch(`${this.baseUrl}/api/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        data: [
          request.prompt,
          request.num_steps || 30,
          request.width || 1024,
          request.height || 1024,
          request.guidance_scale || 7.5,
        ],
        fn_index: 0,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();

    if (result.error) {
      throw new Error(result.error);
    }

    return {
      image: result.data[0],
      message: result.data[1],
    };
  }

  // Método para decodificar imagen base64
  decodeBase64Image(base64String: string): string {
    return `data:image/png;base64,${base64String}`;
  }

  // Método para descargar imagen
  downloadImage(base64String: string, filename: string = "sdxl-image.png") {
    const link = document.createElement("a");
    link.href = this.decodeBase64Image(base64String);
    link.download = filename;
    link.click();
  }
}

// Ejemplo de uso
async function example() {
  const client = new SDXLClient();

  try {
    console.log("🚀 Generando imagen con SDXL...");

    const result = await client.generateImage({
      prompt:
        "A kawaii cat professor wearing glasses, academic robe, and graduation cap, high quality, detailed",
      num_steps: 30,
      width: 1024,
      height: 1024,
      guidance_scale: 7.5,
    });

    console.log("✅ Imagen generada:", result.message);

    // Mostrar imagen en el DOM
    const img = document.createElement("img");
    img.src = client.decodeBase64Image(result.image);
    img.style.maxWidth = "100%";
    document.body.appendChild(img);

    // Opcional: descargar imagen
    // client.downloadImage(result.image, "mi-imagen-sdxl.png");
  } catch (error) {
    console.error("❌ Error:", error);
  }
}

// Ejemplo con polling para progreso (simplificado)
async function generateWithProgress(prompt: string) {
  const client = new SDXLClient();

  console.log("⏳ Iniciando generación...");

  // Polling simple - en una app real usarías SSE o WebSockets
  const checkProgress = async () => {
    try {
      const result = await client.generateImage({ prompt });
      console.log("✅ Completado:", result.message);
      return result;
    } catch (error) {
      if (error.message.includes("cargando")) {
        console.log("⏳ Modelo aún cargando...");
        setTimeout(checkProgress, 5000); // Revisar cada 5 segundos
      } else {
        throw error;
      }
    }
  };

  return checkProgress();
}

// Exportar para uso en otros módulos
export { SDXLClient, generateWithProgress };
