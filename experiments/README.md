# experiments/

Log de runs de `evaluate`, `tournament` y `train`. Cada ejecución guarda
aquí su config, métricas y metadata de git para reproducibilidad y bisect
de regresiones.

## Layout

```
experiments/
├── README.md                           (este fichero, versionado)
├── FAILED.md                           (log de experimentos descartados, versionado)
└── {YYYY-MM-DD_HHMMSS}_{action}/       (gitignored)
    ├── config.json                     (args del run)
    ├── result.json                     (métricas devueltas)
    └── meta.json                       (git SHA, branch, dirty flag)
```

## Convenciones

- **Los runs individuales están gitignored** (`experiments/*/`). Solo se versiona este README y `FAILED.md`.
- **Nunca editar runs a mano**: si un run fue inválido, mover a `archived/` o anotarlo en `FAILED.md`.
- **Runs dirty**: si `meta.json.git_dirty == true`, el código tenía cambios sin commitear cuando se lanzó — la reproducibilidad no está garantizada. Lanzar runs "oficiales" solo con el working tree limpio.

## Cómo se genera

Automáticamente al correr cualquier acción que llame a `core.experiments.log_experiment()`:

```bash
python cli.py chess evaluate --games 10       # → experiments/{ts}_evaluate/
python cli.py chess tournament --a base --b rlhf --games 50   # → experiments/{ts}_tournament/
```

## Consultas útiles

```bash
# Ver todos los runs de evaluación ordenados por fecha
ls -t experiments/ | grep evaluate

# Ver la ELO estimada de todos los runs
for d in experiments/*_evaluate/; do
  echo "$d: $(/opt/homebrew/bin/jq -r '.estimate' "$d/result.json")"
done

# Bisect: qué commit causó una regresión
# (comparar meta.json.git_sha entre runs buenos y malos)
```
