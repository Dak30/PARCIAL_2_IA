from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def cargar_modelo(model_name="gpt2"):
    """
    Carga el modelo y el tokenizador.
    
    Args:
        model_name (str): El nombre del modelo de transformers (por ejemplo, 'gpt2')
    
    Returns:
        modelo: El modelo de lenguaje cargado
        tokenizador: El tokenizador cargado
    """
    tokenizador = GPT2Tokenizer.from_pretrained(model_name)
    modelo = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Asignar el token de padding como el token de fin de secuencia (eos_token)
    tokenizador.pad_token = tokenizador.eos_token
    
    return modelo, tokenizador

def preprocesar_entrada(texto, tokenizador, longitud_maxima=512):
    """
    Preprocesa el texto de entrada para pasarlo al modelo.
    
    Args:
        texto (str): Texto de entrada del usuario
        tokenizador: Tokenizador del modelo
        longitud_maxima (int): Longitud máxima de la secuencia
    
    Returns:
        dict: Tensor de entrada para el modelo
    """
    # Tokenizar la entrada y truncar si es necesario
    entrada = tokenizador(texto, return_tensors="pt", max_length=longitud_maxima, truncation=True, padding=True)
    return entrada


def generar_respuesta(modelo, entrada_procesada, tokenizador, parametros_generacion=None):
    """
    Genera una respuesta basada en la entrada procesada.
    
    Args:
        modelo: Modelo de lenguaje
        entrada_procesada: Tokens de entrada procesados
        tokenizador: Tokenizador del modelo
        parametros_generacion (dict): Parámetros para controlar la generación
        
    Returns:
        str: Respuesta generada
    """
    if parametros_generacion is None:
        parametros_generacion = {
            "max_new_tokens": 50,  # Limitar la longitud de la respuesta generada
            "num_beams": 5,    # Uso de beam search para mejorar la calidad
            "no_repeat_ngram_size": 2,  # Evitar repeticiones
            "temperature": 0.7,  # Controlar la creatividad
        }
    
    # Generación de la respuesta
    salida = modelo.generate(input_ids=entrada_procesada["input_ids"], 
                             attention_mask=entrada_procesada["attention_mask"],
                             **parametros_generacion)
    
    # Decodificar la salida y convertirla en texto legible
    respuesta = tokenizador.decode(salida[0], skip_special_tokens=True)
    return respuesta


def crear_prompt_sistema(instrucciones):
    """
    Crea un prompt de sistema para dar instrucciones al modelo.
    
    Args:
        instrucciones (str): Instrucciones sobre cómo debe comportarse el chatbot
    
    Returns:
        str: Prompt formateado
    """
    prompt_sistema = f"Este es un modelo de IA que responderá preguntas de acuerdo a las siguientes instrucciones:\n{instrucciones}\n"
    return prompt_sistema

def interaccion_simple():
    # Cargar el modelo y el tokenizador
    modelo, tokenizador = cargar_modelo("gpt2")

    # Crear un prompt de sistema (opcional)
    instrucciones = "Responde de forma clara y concisa."
    prompt = crear_prompt_sistema(instrucciones)
    
    # Procesar una entrada de ejemplo
    pregunta = "What is the capital of Colombia?"
    entrada_procesada = preprocesar_entrada(prompt + pregunta, tokenizador)

    # Generar y mostrar la respuesta
    respuesta = generar_respuesta(modelo, entrada_procesada, tokenizador)
    print(f"Pregunta: {pregunta}")
    print(f"Respuesta generada: {respuesta}")

# Ejecutar la interacción simple
interaccion_simple()
