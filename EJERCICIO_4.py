from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import time
import os

def configurar_cuantizacion(bits=4):
    """
    Configura los parámetros para la cuantización del modelo.
    """
    config_cuantizacion = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    return config_cuantizacion

def cargar_modelo_optimizado(nombre_modelo, optim):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if optim == "modelo_base":
        modelo = AutoModelForCausalLM.from_pretrained(nombre_modelo)
        tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)

    elif optim == "cuantizacion_4bit":
        import torch
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            modelo = AutoModelForCausalLM.from_pretrained(nombre_modelo, quantization_config=bnb_config, device_map="auto")
            tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
        else:
            raise RuntimeError("❌ La cuantización 4bit requiere GPU con CUDA")

    else:
        raise ValueError(f"Optimización no reconocida: {optim}")

    if tokenizador.pad_token is None:
        tokenizador.pad_token = tokenizador.eos_token

    return modelo, tokenizador


def aplicar_sliding_window(modelo, window_size=1024):
    """
    Configura la atención de ventana deslizante para procesar secuencias largas.
    """
    if hasattr(modelo.config, 'attention_window'):
        modelo.config.attention_window = [window_size] * modelo.config.num_hidden_layers

def evaluar_rendimiento(modelo, tokenizador, texto_prueba, dispositivo):
    """
    Evalúa el rendimiento del modelo.
    """
    modelo.to(dispositivo)
    modelo.eval()

    inputs = tokenizador(texto_prueba, return_tensors="pt", padding=True).to(dispositivo)

    if dispositivo == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=dispositivo)

    inicio = time.time()
    with torch.no_grad():
        salida = modelo.generate(**inputs, max_new_tokens=50)
    fin = time.time()

    tiempo = fin - inicio
    tokens_generados = salida.shape[-1]
    tokens_por_segundo = tokens_generados / tiempo

    if dispositivo == "cuda":
        memoria_usada = torch.cuda.max_memory_allocated(device=dispositivo) / 1e6  # MB
    else:
        memoria_usada = 0.0

    return {
        "tiempo_inferencia (s)": round(tiempo, 2),
        "tokens_generados": int(tokens_generados),
        "tokens/seg": round(tokens_por_segundo, 2),
        "memoria_max_MB": round(memoria_usada, 2)
    }


def demo_optimizaciones():
    configuraciones = ["modelo_base", "cuantizacion_4bit"]
    modelo_id = "distilgpt2"
    texto_prueba = "La inteligencia artificial es"
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

    for optim in configuraciones:
        if optim == "cuantizacion_4bit" and dispositivo == "cpu":
            print(f"\n⚠️  Saltando configuración: {optim} (requiere GPU con CUDA)\n")
            continue

        print(f"\n🔧 Evaluando configuración: {optim}")
        modelo, tokenizador = cargar_modelo_optimizado(modelo_id, optim)
        metricas = evaluar_rendimiento(modelo, tokenizador, texto_prueba, dispositivo)
        for clave, valor in metricas.items():
            print(f"{clave}: {valor}")


if __name__ == "__main__":
    demo_optimizaciones()
