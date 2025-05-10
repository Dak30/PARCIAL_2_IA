import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Setting `pad_token_id`")


# Configuración del modelo LoRA (PEFT)
def configurar_peft(modelo, r=8, lora_alpha=32):
    """
    Configura el modelo para fine-tuning con PEFT/LoRA.
    
    Args:
        modelo: Modelo base
        r (int): Rango de adaptadores LoRA
        lora_alpha (int): Escala alpha para LoRA
    
    Returns:
        modelo: Modelo adaptado para fine-tuning
    """
    lora_config = LoraConfig(r=r, lora_alpha=lora_alpha, task_type=TaskType.CAUSAL_LM)
    modelo_peft = get_peft_model(modelo, lora_config)
    return modelo_peft

# Guardar el modelo y el tokenizador
def guardar_modelo(modelo, tokenizador, ruta):
    """
    Guarda el modelo y tokenizador en una ruta específica.
    
    Args:
        modelo: Modelo a guardar
        tokenizador: Tokenizador del modelo
        ruta (str): Ruta donde guardar
    """
    os.makedirs(ruta, exist_ok=True)
    
    modelo.save_pretrained(ruta)
    tokenizador.save_pretrained(ruta)
    print(f"Modelo guardado en {ruta}")

# Cargar el modelo y tokenizador personalizados
def cargar_modelo_personalizado(ruta):
    """
    Carga un modelo personalizado desde una ruta específica.
    
    Args:
        ruta (str): Ruta del modelo
        
    Returns:
        tuple: (modelo, tokenizador)
    """
    modelo = AutoModelForCausalLM.from_pretrained(ruta)
    tokenizador = AutoTokenizer.from_pretrained(ruta)
    return modelo, tokenizador

# Crear interfaz web con Gradio
def crear_interfaz_web(chatbot):
    """
    Crea una interfaz web simple para el chatbot usando Gradio.
    
    Args:
        chatbot: Instancia del chatbot
        
    Returns:
        gr.Interface: Interfaz de Gradio
    """
    def respuesta(input_text):
        return chatbot(input_text)
    
    interfaz = gr.Interface(fn=respuesta, inputs="text", outputs="text")
    return interfaz

# Función principal para el despliegue
def main_despliegue():
    # Ruta donde guardar el modelo
    ruta_modelo = "C:/UNIIDAD_2_PARCIAL_2/modelo_guardado"
    
    # Cargar o entrenar el modelo
    modelo, tokenizador = cargar_modelo_personalizado("gpt2")  # Cargar un modelo preentrenado como ejemplo
    
    # Configurar el modelo LoRA
    modelo_peft = configurar_peft(modelo)
    
    # Guardar el modelo adaptado y el tokenizador
    guardar_modelo(modelo_peft, tokenizador, ruta_modelo)
    
    # Crear la función de chatbot, que ahora tokeniza el input antes de pasarlo al modelo
    def chatbot(input_text):
        # Tokenizar el texto de entrada
        inputs = tokenizador(input_text, return_tensors="pt")
        
        # Generar la respuesta usando el modelo
        output = modelo_peft.generate(**inputs)
        
        # Decodificar la respuesta generada
        respuesta = tokenizador.decode(output[0], skip_special_tokens=True)
        
        return respuesta
    
    # Crear una interfaz de Gradio para el chatbot
    interfaz = crear_interfaz_web(chatbot)
    
    # Lanzar la interfaz web
    interfaz.launch()

if __name__ == "__main__":
    main_despliegue()
