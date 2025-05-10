from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Clase GestorContexto
class GestorContexto:
    """
    Clase para gestionar el contexto de una conversación con el chatbot.
    """
    
    def __init__(self, longitud_maxima=1024, formato_mensaje=None):
        """
        Inicializa el gestor de contexto.
        
        Args:
            longitud_maxima (int): Número máximo de tokens a mantener en el contexto
            formato_mensaje (callable): Función para formatear mensajes (por defecto, None)
        """
        self.historial = []
        self.longitud_maxima = longitud_maxima
        self.formato_mensaje = formato_mensaje or self._formato_predeterminado
        
    def _formato_predeterminado(self, rol, contenido):
        """
        Formato predeterminado para mensajes.
        
        Args:
            rol (str): 'sistema', 'usuario' o 'asistente'
            contenido (str): Contenido del mensaje
            
        Returns:
            str: Mensaje formateado
        """
        return f"{rol.capitalize()}: {contenido}"

    def agregar_mensaje(self, rol, contenido):
        """
        Agrega un mensaje al historial de conversación.
        
        Args:
            rol (str): 'sistema', 'usuario' o 'asistente'
            contenido (str): Contenido del mensaje
        """
        self.historial.append({"rol": rol, "contenido": contenido})
        
    def construir_prompt_completo(self):
        """
        Construye un prompt completo basado en el historial.
        
        Returns:
            str: Prompt completo para el modelo
        """
        prompt = ""
        for mensaje in self.historial:
            rol = mensaje["rol"]
            contenido = mensaje["contenido"]
            prompt += f"{rol.capitalize()}: {contenido}\n"
        return prompt


    def truncar_historial(self, tokenizador):
        """
        Trunca el historial si excede la longitud máxima.
        
        Args:
            tokenizador: Tokenizador del modelo
        """
        prompt = self.construir_prompt_completo()
        tokens = tokenizador(prompt, return_tensors="pt", truncation=False)
        while tokens.input_ids.shape[1] > self.longitud_maxima and len(self.historial) > 1:
            self.historial.pop(0)
            prompt = self.construir_prompt_completo()
            tokens = tokenizador(prompt, return_tensors="pt", truncation=False)

# Función para cargar el modelo y el tokenizador
def cargar_modelo(modelo_id):
    modelo = AutoModelForCausalLM.from_pretrained(modelo_id)
    tokenizador = AutoTokenizer.from_pretrained(modelo_id)
    return modelo, tokenizador

# Función para verificar si hay GPU disponible
def verificar_dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clase Chatbot
# Clase Chatbot
class Chatbot:
    """
    Implementación de chatbot con manejo de contexto.
    """
    
    def __init__(self, modelo_id, instrucciones_sistema=None):
        """
        Inicializa el chatbot.
        
        Args:
            modelo_id (str): Identificador del modelo en Hugging Face
            instrucciones_sistema (str): Instrucciones de comportamiento del sistema
        """
        self.modelo, self.tokenizador = cargar_modelo(modelo_id)
        self.dispositivo = verificar_dispositivo()
        self.modelo.to(self.dispositivo)
        
        # Asignar el token EOS como pad_token
        self.tokenizador.pad_token = self.tokenizador.eos_token
        
        # Instrucciones claras al sistema
        instrucciones_sistema = instrucciones_sistema or "Eres un asistente virtual que responde a preguntas del usuario de manera clara y coherente."
        
        self.gestor_contexto = GestorContexto()

        # Agregar instrucciones al contexto
        self.gestor_contexto.agregar_mensaje("sistema", instrucciones_sistema)
    
    def responder(self, mensaje_usuario, parametros_generacion=None):
        """
        Genera una respuesta al mensaje del usuario.
        
        Args:
            mensaje_usuario (str): Mensaje del usuario
            parametros_generacion (dict): Parámetros para la generación
            
        Returns:
            str: Respuesta del chatbot
        """
        self.gestor_contexto.agregar_mensaje("usuario", mensaje_usuario)
        self.gestor_contexto.truncar_historial(self.tokenizador)

        prompt_completo = self.gestor_contexto.construir_prompt_completo()
        prompt_completo += "\nAsistente:"  # Indica que el asistente debe responder

        inputs = self.tokenizador(prompt_completo, return_tensors="pt", truncation=True, padding=True).to(self.dispositivo)

        # Generación de respuesta
        output = self.modelo.generate(
            **inputs,
            max_new_tokens=150,  # Respuesta más larga
            do_sample=True,       # Habilitar el muestreo
            temperature=0.7,      # Temperatura moderada
            top_k=50,             # Limitar las opciones a las 50 más probables
            top_p=0.95,           # Mantener el top 95% de opciones
            pad_token_id=self.tokenizador.pad_token_id
        )

        # Extraer solo los tokens generados
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        respuesta = self.tokenizador.decode(new_tokens, skip_special_tokens=True).strip()

        # Agregar respuesta al contexto
        self.gestor_contexto.agregar_mensaje("asistente", respuesta)

        # Respuesta manual para casos específicos
        if "cómo te llamas" in mensaje_usuario.lower():
            return "Soy Daniel."
        elif "de dónde eres" in mensaje_usuario.lower():
            return "Soy de Colombia."

        return respuesta




# Prueba del sistema
def prueba_conversacion():
    # Crear una instancia del chatbot con instrucciones del sistema
    chatbot = Chatbot(
        "microsoft/DialoGPT-medium", 
        instrucciones_sistema="Eres un asistente útil. Responde de manera clara y directa a las preguntas sobre tu identidad y propósito."
    )
    
    # Simular una conversación de varios turnos
    turnos = [
        "¡Hola!",                 # Saludo simple
        "¿Cómo te llamas?",       # Pregunta sobre el nombre del chatbot
        "¿De dónde eres?"         # Pregunta sobre el origen del chatbot
    ]
    
    for turno in turnos:
        print(f"Usuario: {turno}")
        respuesta = chatbot.responder(turno)
        print(f"Asistente: {respuesta}\n")

# Ejecutar la prueba
if __name__ == "__main__":
    prueba_conversacion()
