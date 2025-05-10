import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Configurar las variables de entorno para la caché de modelos
os.environ['TRANSFORMERS_CACHE'] = './model_cache'

def cargar_modelo(nombre_modelo):
    tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
    modelo = AutoModelForCausalLM.from_pretrained(nombre_modelo)
    modelo.eval()
    
    if torch.cuda.is_available():
        modelo = modelo.half().to("cuda")
    else:
        modelo = modelo.to("cpu")

    return modelo, tokenizador


def verificar_dispositivo():
    """
    Verifica el dispositivo disponible (CPU/GPU) y muestra información relevante.
    
    Returns:
        torch.device: Dispositivo a utilizar
    """
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"Memoria disponible: {torch.cuda.mem_get_info(0)[0] // (1024**2)} MB")
    else:
        dispositivo = torch.device("cpu")
        print("No se detectó GPU, utilizando CPU.")
    return dispositivo

# Función principal de prueba
def main():
    dispositivo = verificar_dispositivo()
    print(f"Utilizando dispositivo: {dispositivo}")
    
    # Cargar un modelo pequeño adecuado para chatbots
    nombre_modelo = "PlanTL-GOB-ES/gpt2-base-bne" 
    modelo, tokenizador = cargar_modelo(nombre_modelo)

    # Realizar una prueba simple de generación de texto
    prompt = "Hola, ¿cómo estás?"
    entradas = tokenizador(prompt, return_tensors="pt").to(dispositivo)
    salida = modelo.generate(
        **entradas,
        max_length=60,
        do_sample=True,
        top_k=40,
        temperature=0.8,
        num_return_sequences=1
    )


    texto_generado = tokenizador.decode(salida[0], skip_special_tokens=True)
    print("Texto generado:")
    print(texto_generado)

if __name__ == "__main__":
    main()
