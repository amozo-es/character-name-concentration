# Standalone Batch Story Generator

Este paquete es **completamente autocontenido** e independiente del backend y frontend de la aplicación principal. Está diseñado para poder copiarse a cualquier otra máquina y ejecutarse de forma aislada.

## Contenido del Paquete
- `generate.py`: Script principal que ejecuta la generación de historias en lote muestreando nuevas imágenes de forma aleatoria.
- `generate_multiphase.py`: Script complementario que procesa las listas de imágenes ya existentes en la base de datos para generar historias usando una estrategia de dos fases (Phase 1: borrador con placeholders, Phase 2: asignación de nombres creativos).
- `generate_multiphase_prob.py`: Script complementario que procesa las listas de imágenes existentes utilizando **Verbalized Sampling** en dos fases (Phase 1: borrador con placeholders, Phase 2: obtención de una lista de 5 asignaciones de nombres con sus probabilidades en formato JSON).
- `generate_speculative.py`: Script que aplica **Speculative Naming** — traslada el paradigma draft-verify-reject del speculative decoding al control de slop de nombres. Reutiliza el pipeline de verbalized sampling (fases 1-2) e inyecta verificación contra fingerprints de slop model-specific. Parámetro clave `--lambdas` controla la agresividad anti-slop.
- `generate_slop_fingerprints.py`: Script de preprocesamiento que genera `slop_fingerprints.json` desde las extracciones de personajes del baseline (historias con `prompt_modification IS NULL`). Deriva el fingerprint de slop de cada modelo con 3 métodos (top-20, frecuencia ≥2%, knee detection). Debe ejecutarse **antes** de `generate_speculative.py`.
- `slop_fingerprints.json`: Fingerprints estáticos de slop por modelo, generados por `generate_slop_fingerprints.py`. Contiene las frecuencias reales de cada nombre en el baseline del modelo, usadas por `generate_speculative.py` como señal de verificación.
- `model.json`: Configuración de los modelos de IA a utilizar.
- `reelsData.json`: Datos de los reels y sus imágenes correspondientes.
- `requirements.txt`: Lista de dependencias de Python necesarias.
- `prompts/`: Carpeta con las plantillas de prompts, incluyendo `story_generation_multiphase/` y `verbalized_sampling/` (para la asignación probabilística de nombres).
- `icons/`: Copia local de los iconos de la aplicación para poder compilar las tiras de imágenes para los modelos de visión.
- `utils/`: Módulos de utilidad necesarios para la interacción con los LLM a través de LangChain.

---

## Requisitos e Instalación

Para ejecutar el generador en otra máquina, solo necesitas instalar las dependencias incluidas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Configuración de Modelos (`model.json`)

El archivo `model.json` define qué modelos se ejecutarán y con qué parámetros. 

### Parámetro de Razonamiento (`think`)
Puedes controlar el comportamiento del razonamiento del modelo usando la clave `"think"` en la configuración de cada modelo:
- Si se incluye `"think": false`, se desactivará el razonamiento del modelo (se pasa `"think": false` y `"reasoning_effort": "none"` en la petición al modelo).
- Si no se define o se omite, no se enviará ninguna configuración de razonamiento al generador.

Ejemplo de configuración para Ollama local (puerto 1234 con Qwen):
```json
[
  {
    "name": "Qwen 3.5 4B",
    "host": "http://localhost:1234/v1",
    "model": "qwen3.5:4b",
    "model_type": ["chat", "vision"],
    "api_key": "OLLAMA_APIKEY",
    "provider": "ollama",
    "think": false
  }
]
```

---

## Modificaciones del Prompt de Usuario (`modifications.json`)

Para analizar y mitigar el "slop" (comportamientos y frases repetitivas de las IA), el generador soporta modificaciones que se añaden al prompt de usuario.

### Funcionamiento
1. En la carpeta `prompts/story_generation/`, el generador busca el archivo `modifications.json`.
2. Si el archivo existe, el script ejecutará **tanto el prompt base** como **cada una de las modificaciones** listadas en él para cada lote.
3. Para cada historia generada, se registrará el ID de la modificación en la columna `prompt_modification` de la base de datos (y `NULL`/`None` para la ejecución del prompt base).

### Estructura de `modifications.json`
El archivo contiene un array de objetos con las traducciones en los cuatro idiomas soportados:
```json
[
  {
    "id": "avoid_slop_1",
    "es": "Debes evitar las expresiones típicas de las IA, no elegir siempre la primera opción en nombres, tramas, comportamientos, etc.",
    "en": "You must avoid typical AI expressions, and avoid always choosing the first option in names, plots, behaviors, etc.",
    "fr": "Vous devez éviter les expressions typiques des IA, et ne pas toujours choisir la première option pour les noms, les intrigues, les comportements, etc.",
    "de": "Sie müssen typische KI-Ausdrücke vermeiden und dürfen nicht immer die erste Option bei Namen, Handlungen, Verhaltensweisen usw. wählen."
  }
]
```

---

## ¿Cómo Funciona?

Para cada iteración solicitada, el generador realiza los siguientes pasos de forma automática:

1. **Selección de Imágenes**: Selecciona aleatoriamente **5 imágenes**, garantizando que cada una pertenezca a un reel diferente (cargados desde `reelsData.json`).
2. **Selección del Idioma**: Elige el idioma de la historia con la siguiente probabilidad ponderada:
   - **40%** Inglés (`en`)
   - **20%** Francés (`fr`)
   - **20%** Alemán (`de`)
   - **20%** Español (`es`)
3. **Traducción**: Obtiene el nombre de cada tag de imagen localizado al idioma seleccionado.
4. **Estrategia Dual (Chat y Visión)**: Para cada modelo definido en `model.json`, el script evalúa sus capacidades en `model_type`:
   - **`chat`**: Genera la historia enviando una petición de texto plano con los nombres de las imágenes en el prompt.
   - **`vision`**: Combina los 5 iconos seleccionados en una sola tira horizontal (PNG), la convierte a base64, y la envía junto con el prompt al modelo.
   - Si el modelo soporta ambos (`"chat"` y `"vision"`), se ejecutan **ambas** llamadas y se generan dos historias independientes para esa misma combinación de imágenes.
5. **Persistencia Local**: Las historias y sus listas de imágenes se almacenan en una base de datos local SQLite independiente (`custom_stories.db`). La tira de imágenes combinada (strip) se asocia y guarda directamente con la lista de imágenes (`image_lists.image_data`) en lugar de duplicarse en cada historia, optimizando así el almacenamiento.

---

## Parámetros de la Línea de Comandos

El script acepta argumentos para personalizar la ejecución:

- `--count` / `-c`: Número de conceptos de historias a generar (por defecto: `10`).
- `--models-config` / `-m`: Ruta al archivo JSON con la configuración de los modelos (por defecto: `model.json` en la misma carpeta).
- `--db-path` / `-d`: Ruta de salida de la base de datos SQLite (por defecto: `custom_stories.db` en la misma carpeta).
- `--language` / `-l`: Fuerza la generación de historias en un idioma específico (`en`, `fr`, `de`, `es`). Si se omite, se seleccionará de manera aleatoria siguiendo la distribución de probabilidad (40% inglés, 20% francés, alemán y español).
- `--threads` / `-t`: Número máximo de hilos/ejecuciones de modelos en paralelo (por defecto: `4`). Permite controlar la concurrencia y evitar saturar instancias locales como Ollama (puedes usar `1` para ejecución secuencial).
- `--min-p`: Lista de valores `min_p` para iterar la generación (ej. `--min-p 0.05 0.1 0.2`). Por defecto, si no se incluye este parámetro, no se pasa `min_p` a la llamada. Cuando se especifican valores, el generador iterará sobre cada uno de ellos y guardará la historia correspondiente añadiendo el sufijo `_min_p_X` a la columna `prompt_modification`.

### Ejemplos de Uso

#### 1. Generador Estándar (`generate.py`)
Genera conceptos aleatorios nuevos escribiendo historias directamente.

**Generar las 10 historias por defecto usando la configuración estándar:**
```bash
python generate.py
```

**Generar 5 historias forzando el idioma francés de forma secuencial (1 solo hilo):**
```bash
python generate.py --count 5 --language fr --threads 1
```

**Generar 50 historias usando una configuración de base de datos específica:**
```bash
python generate.py --count 50 --db-path mis_historias.db
```

**Generar historias iterando sobre diferentes valores de `min_p`:**
```bash
python generate.py --min-p 0.05 0.1
```
*(Esto creará historias independientes y se guardarán en la base de datos con los identificadores de modificación `_min_p_0.05` y `_min_p_0.1` o sufijados a las modificaciones aplicadas, por ejemplo `avoid_slop_1_min_p_0.05`)*

#### 2. Generador Multifase Complementario (`generate_multiphase.py`)
No genera imágenes aleatorias nuevas, sino que lee las composiciones de imágenes que ya existen en tu base de datos (`custom_stories.db`) y genera sus correspondientes variantes de historias utilizando la estrategia en dos fases.
El idioma se detecta automáticamente de la historia previa ya generada para cada composición.

**Ejecutar la generación multifase para todas las listas de imágenes existentes:**
```bash
python generate_multiphase.py --threads 1
```

**Ejecutar la generación multifase especificando una base de datos y configuración alternativas:**
```bash
python generate_multiphase.py --models-config model2.json --db-path otra_base.db
```

*Nota: La estrategia multifase se registra bajo el campo `prompt_modification = "multiphase"` en la base de datos.*

#### 3. Generador Multifase Probabilístico / Verbalized Sampling (`generate_multiphase_prob.py`)
Lee las composiciones de imágenes existentes en la base de datos para generar historias con marcadores de posición en la Fase 1, e identifica y asigna una distribución de probabilidad para 5 conjuntos de nombres en la Fase 2 (estrategia de muestreo verbalizado).
El resultado final se serializa como JSON y se guarda en la columna `text` de la base de datos.

**Ejecutar la generación multifase probabilística para todas las listas de imágenes existentes:**
```bash
python generate_multiphase_prob.py --threads 1
```

**Ejecutar especificando una configuración de modelos y base de datos alternativas:**
```bash
python generate_multiphase_prob.py --models-config model2.json --db-path otra_base.db
```

*Nota: La estrategia de muestreo verbalizado se registra bajo el campo `prompt_modification = "multiphase_prob"` (o `multiphase_prob_min_p_X`) en la base de datos.*

#### 4. Generador Multiagente (`generate_multiagent.py`)
Utiliza un sistema multiagente cooperativo basado en **LangGraph** (Writer + Reviewer) para redactar y validar las historias. El Reviewer analiza los nombres propios asignados y los rechaza si están en la lista de nombres prohibidos (`prompts/story_generation_multiagent/forbidden_names.md`) o si ese modelo específico los ha usado demasiadas veces.

Las historias finales aprobadas se guardan en la base de datos de historias (`custom_stories.db`) bajo el campo `prompt_modification = "multiagent"`, y los contadores de nombres usados se actualizan en una base de datos independiente llamada `name_repetitions.db`.

**Ejecutar la generación multiagente para todas las listas de imágenes existentes:**
```bash
python generate_multiagent.py --threads 1
```

**Parámetros específicos del generador multiagente:**
- `--reps-db-path` / `-r`: Ruta de la base de datos SQLite independiente donde se registran las repeticiones (por defecto: `name_repetitions.db`).
- `--threshold` / `-k`: Límite de veces que se permite usar un nombre antes de que el Reviewer lo empiece a rechazar para ese modelo (por defecto: `2`, lo que significa que a la segunda aparición será rechazado).
- `--reviewer-config` / `-rc`: Ruta del archivo JSON de configuración del modelo Revisor/Evaluador (por defecto: `reviewer_model.json`). Esto permite desacoplar al revisor en un endpoint comercial o diferente al del escritor local.

**Configurar y diagnosticar el Revisor:**
1. Edita el archivo `reviewer_model.json` para configurar el endpoint del modelo evaluador (puedes apuntar a endpoints de OpenAI, Claude, OpenRouter, o a una instancia local de Ollama de mayor tamaño como `llama3.3:70b` o `qwen3.6:35b`).
2. Utiliza el script de diagnóstico `check_reviewer.py` para verificar de forma aislada que el modelo revisor responde y extrae correctamente los nombres (muy útil para validar conectividad en entornos restringidos):
   ```bash
   python check_reviewer.py
   ```

**Ejecutar con configuración alternativa y límite de 3 repeticiones por nombre:**
```bash
python generate_multiagent.py --threshold 3 --reps-db-path mis_repeticiones.db --reviewer-config mi_revisor.json --threads 1
```

---

#### 5. Generador Speculative Naming (`generate_speculative.py`)

Aplica el paradigma **draft-verify-reject** del speculative decoding al control de slop de nombres. El modelo genera 5 candidatos de nombre (draft, reutilizando verbalized sampling), se verifican contra el fingerprint de slop real del modelo (verify, desde `slop_fingerprints.json`), y se selecciona con una probabilidad que penaliza los nombres sobreusados:

```
accept_prob(nombre) ∝ p_verbalizada × exp(−λ × slop_score)
```

Donde `λ` (lambda) es un parámetro continuo que controla la agresividad anti-slop: `λ=0` equivale a verbalized sampling puro (control), `λ=20` es muy agresivo (el slop apenas aparece).

**Requisito previo**: ejecutar `generate_slop_fingerprints.py` primero (ver sección 6 más abajo).

Las historias se guardan bajo `prompt_modification = "speculative_l00"`, `"speculative_l05"`, `"speculative_l10"`, `"speculative_l20"`, `"speculative_hard"` (y variantes con sufijo `_min_p_X` si se usa `--min-p`).

**Parámetros específicos del generador speculative:**
- `--lambdas`: Lista de valores λ para la regla de aceptación (por defecto: `0 5 10 20`). `0` = control sin penalización (equivale a verbalized sampling). Valores más altos = más agresividad anti-slop.
- `--hard`: Genera también la variante HARD (selección por threshold en vez de λ). Acepta el primer set donde ningún nombre supere el threshold de slop.
- `--hard-threshold`: Frecuencia de slop máxima permitida en modo HARD (por defecto: `0.02` = 2%).
- `--slop-fingerprints`: Ruta al archivo JSON de fingerprints (por defecto: `slop_fingerprints.json` en la misma carpeta).
- `--limit`: Limita el número total de historias a generar en esta ejecución.
- `--min-p`: Lista de valores `min_p` para combinar con speculative (ej. `--min-p 0.02`). Genera variantes adicionales con sufijo `_min_p_0.02`.
- `--dry-run`: Ejecuta sin guardar en la base de datos (útil para verificar configuración).

**Pilot controlado (10 historias, 2 valores de λ):**
```bash
python generate_speculative.py --lambdas 0 10 --limit 10
```

**Experimento completo (4 lambdas + hard + min_p):**
```bash
python generate_speculative.py --lambdas 0 5 10 20 --hard --min-p 0.02
```

**Solo modelos específicos (crear `model_pilot.json` con los modelos deseados):**
```bash
python generate_speculative.py --models-config model_pilot.json --lambdas 0 10 --limit 20
```

**Dry-run para verificar configuración sin llamar al LLM:**
```bash
python generate_speculative.py --dry-run --lambdas 0 5 10 20 --hard
```

**Filtrar por un solo modelo (ej. solo Gemma):**
```bash
python generate_speculative.py --lambdas 0 10 --limit 5
# (editar model.json para contener solo los modelos deseados, o usar --models-config)
```

> **Nota sobre el analizador**: El script `analyze_stories.py` (en `wiki/code/name-analyzer/`) procesa las historias speculative **sin necesidad de cambios**. Detecta automáticamente los nuevos valores de `prompt_modification` (`speculative_l05`, etc.) y los integra en la matriz de análisis existente.

---

#### 6. Generador de Slop Fingerprints (`generate_slop_fingerprints.py`)

Script de preprocesamiento que debe ejecutarse **antes** de `generate_speculative.py`. Lee las extracciones de personajes del baseline (historias con `prompt_modification IS NULL`) desde la base de datos de análisis y deriva el fingerprint de slop de cada modelo.

El fingerprint es **model-specific**: Tiber cuenta como slop para Gemma (donde aparece al 22-31%) pero no para Ministral (donde nunca se produce). Esto corrige el problema del `SLOP_NAMES` estático universal que mezclaba slop de distintos modelos.

Para cada modelo se derivan **3 métodos en paralelo**:
1. **top-20**: los 20 nombres más frecuentes del baseline
2. **≥2%**: nombres con frecuencia ≥ 2%
3. **knee detection**: detección de codo automática sobre la curva de frecuencias

**Parámetros:**
- `--db`: Ruta a la base de datos de análisis (por defecto: `wiki/code/name-analyzer/custom_analysis.db`).
- `--output` / `-o`: Ruta de salida del JSON (por defecto: `slop_fingerprints.json`).
- `--top-n`: N para el método top-N (por defecto: `20`).
- `--pct-threshold`: Umbral de frecuencia % para el método de threshold (por defecto: `2.0`).
- `--model`: Filtrar por nombre de modelo (match por substring). Si se omite, procesa todos.

**Generar fingerprints para todos los modelos:**
```bash
python generate_slop_fingerprints.py
```

**Generar solo para modelos Gemma:**
```bash
python generate_slop_fingerprints.py --model "Gemma"
```

**Ver resultado (los fingerprints se imprimen en consola):**
```bash
python generate_slop_fingerprints.py --model "Gemma 3:27b (chat)"
# Salida:
# Model                                     chars  uniq  top20   ≥2%  knee
# Gemma 3:27b (chat)                          306   107     20     5     9
#   slop (knee): tiber, elara, elias, silas, leo, tiberius, zorp, alistair, lila
```

**Usar una base de datos de análisis diferente:**
```bash
python generate_slop_fingerprints.py --db path/to/otra_analysis.db --output mis_fingerprints.json
```

> **Importante**: Los fingerprints son estáticos (se congelan al generarlos). No se actualizan durante la generación de historias speculative, lo que evita contaminación con artefactos de estrategias. Regenerar los fingerprints después de añadir nuevas historias baseline para mejorar la cobertura.

---

## Visualizar las Imágenes Generadas (`view_image.py`)

Para extraer y visualizar la tira de imágenes combinada enviada en una historia específica, puedes usar el script `view_image.py` pasando el ID de la historia (completo o solo el inicio/prefijo del UUID):

```bash
python view_image.py <id_de_la_historia>
```

**Ejemplo:**
```bash
python view_image.py 5737667b
```

Esto buscará la historia en la base de datos `custom_stories.db`, decodificará la imagen base64 y la guardará como un archivo PNG local (por ejemplo, `story_5737667b.png`).
