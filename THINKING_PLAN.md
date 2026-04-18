# Plan: añadir "thinking" al LLM de ajedrez

Branch de desarrollo: `claude/chess-llm-thinking-0Fcmf`

## Contexto

El objetivo es que el modelo genere un bloque de razonamiento intermedio
(`<think>...</think>`) antes de emitir la jugada final, al estilo de los
modelos con extended thinking (Opus, o1, DeepSeek-R1).

Enfoque general: ajedrez es un dominio verificable con Stockfish, así que
podemos entrenar el formato con SFT y reforzarlo con RL usando el eval del
motor como reward.

---

## Nivel 1 — SFT con thinking (empezar por aquí)

**Estimación:** 1–2 días de código + 1–5 días de entrenamiento.

### Cambios de código

1. **`domains/chess/tokenizer.py`**
   - Añadir tokens especiales: `<think>`, `</think>`, `<eval=+N>`,
     `<cand:MOVE>`, `<pv>`, `</pv>`.
   - Incluirlos en `RESULT_TOKENS` / vocabulario base.
   - Mantener tokens SAN existentes sin cambios.

2. **Nuevo script `domains/chess/prepare_thinking.py`** (o extender
   `prepare.py`)
   - Para cada posición del corpus:
     - Llamar a Stockfish (reutilizar `evaluator.py`) para extraer:
       - Variación principal (PV)
       - Eval numérica
       - 2–3 jugadas candidatas
     - Serializar como texto dentro de `<think>...</think>`.
   - Formato por jugada:
     ```
     <think> <eval=+0.3> <cand:Nf3> <cand:Bc4> <pv> Nf3 Nc6 Bc4 </pv> </think> Bc4
     ```

3. **`core/dataset.py`**
   - Verificar que admite las secuencias más largas (el thinking infla
     la longitud; probablemente haya que subir `max_len`).

4. **`core/generator.py`**
   - En inferencia: permitir generación libre dentro de
     `<think>...</think>`, y al cerrar el bloque volver a restringir el
     sampling solo a jugadas legales SAN.
   - Exponer la traza de thinking por separado (como hace la API de
     Claude con bloques `thinking` vs `text`).

5. **`domains/chess/play.py` y `ui.py`**
   - Mostrar el bloque de thinking opcionalmente (flag `--show-thinking`).
   - En modo normal, ocultarlo y mostrar solo la jugada.

### Pruebas
- Unit test del tokenizador con secuencias que contengan thinking.
- Sanity check: entrenar con un corpus pequeño y ver si el modelo
  produce el formato correcto.

---

## Nivel 2 — RL estilo DeepSeek-R1 (después del Nivel 1)

**Estimación:** 1–2 semanas de código + 1–3 semanas de entrenamiento RL.

### Cambios de código

1. **`core/rlhf.py`**
   - Añadir máscara de reward: tokens dentro de `<think>...</think>`
     reciben reward 0 (o pequeño coste de longitud).
   - Reward fuerte solo en la jugada posterior al `</think>`.

2. **Nuevo `domains/chess/rl_thinking.py`**
   - Loop de rollouts: modelo genera `<think>...</think> jugada` para
     posiciones sampleadas de un pool.
   - Reward = eval(Stockfish, jugada) − eval(posición) + bonus por
     ganar la partida si se juega hasta el final.
   - Algoritmo: REINFORCE con baseline, o PPO si estabilidad es
     problema.

3. **Guardrails anti reward hacking**
   - Penalizar bloques de thinking vacíos o degenerados.
   - Límite de tokens en thinking (budget, como `budget_tokens` de la
     API de Claude).
   - Verificar periódicamente la ELO contra un baseline fijo.

### Experimentos
- Comparar:
  - Baseline SFT sin thinking
  - SFT con thinking (Nivel 1)
  - SFT + RL con thinking (Nivel 2)
- Medir ELO aproximada con partidas vs Stockfish a profundidad limitada.

---

## Nivel 3 — Escalar (solo si Niveles 1–2 funcionan)

No realista sin presupuesto serio de GPU. Ideas:
- Aumentar tamaño del modelo y corpus.
- MCTS + LLM como policy/value (estilo AlphaZero con razonamiento).
- Interleaved thinking: pensar → consultar motor → pensar → jugar.

---

## Orden de trabajo sugerido

1. Crear branch `claude/chess-llm-thinking-0Fcmf` (ya asignado).
2. Implementar tokenizador + generación de dataset con Stockfish.
3. Entrenar un modelo pequeño con Nivel 1 y verificar formato.
4. Evaluar si el thinking ayuda en ELO antes de invertir en RL.
5. Si pasa el filtro: implementar Nivel 2.

## Archivos clave a tocar

- `domains/chess/tokenizer.py:11` — añadir tokens especiales
- `domains/chess/prepare.py` — extender o duplicar para thinking
- `domains/chess/evaluator.py` — reutilizar para extraer PV/eval
- `core/dataset.py` — posible aumento de `max_len`
- `core/generator.py:35` — lógica de sampling condicional al bloque
- `core/rlhf.py` — máscara de reward (Nivel 2)
- `domains/chess/play.py` / `ui.py` — mostrar/ocultar thinking
