#!/bin/bash

echo "🚀 Desplegando Stable Diffusion XL Ultra-Optimizado en Fly.io..."

# Limpiar máquinas existentes
echo "🧹 Limpiando máquinas existentes..."
fly machines list | grep -v "ID" | awk '{print $1}' | xargs -I {} fly machines destroy {} --force 2>/dev/null || true

# Crear volumen para cache (si no existe)
echo "💾 Creando volumen para cache..."
fly volumes create sdxl_cache --size 50 --region cdg 2>/dev/null || echo "Volumen ya existe"

# Desplegar la aplicación
echo "📦 Desplegando aplicación ultra-optimizada..."
fly deploy --remote-only

# Esperar a que esté lista
echo "⏳ Esperando a que la aplicación esté lista..."
sleep 30

# Mostrar información
echo "✅ Despliegue completado!"
echo "🌐 URL: https://stable-diffusion-xl.fly.dev"
echo "📊 Estado: fly status"
echo "📝 Logs: fly logs"
echo ""
echo "⚡ Configuración Ultra-Rápida:"
echo "   - 15 pasos de inferencia (más rápido)"
echo "   - Guidance scale 7.0 (balance perfecto)"
echo "   - Imagen 1024x1024 (tamaño óptimo)"
echo "   - Optimizaciones agresivas de memoria"
echo "   - Cache automático para modelos" 