#!/usr/bin/env bash
# Flujo completo: train -> evaluate -> quality gate -> resumen.
# El modelo solo se considera "aceptable" si supera el threshold de ELO.
#
# Threshold por defecto: 600 ELO (significativamente por encima del baseline 420).
# Override: THRESHOLD_ELO=500 bash finalize.sh

set -euo pipefail

cd ~/custom_llm
source venv/bin/activate

THRESHOLD_ELO="${THRESHOLD_ELO:-600}"

echo "======================================================"
echo "  PIPELINE COMPLETO H100"
echo "  Threshold de aceptacion: >= $THRESHOLD_ELO ELO"
echo "======================================================"

# 1. Training
echo ""
echo ">>> [1/3] TRAINING"
bash cloud/train.sh

# 2. Evaluacion ELO contra niveles 0-4
echo ""
echo ">>> [2/3] EVALUACION ELO"
python cli.py chess evaluate --games 30 --levels 0 1 2 3 4 --seed 42 \
    2>&1 | tee /tmp/eval.log

# 3. Quality gate: parsear el ELO estimado y comparar con threshold
echo ""
echo ">>> [3/3] QUALITY GATE"
ELO_ESTIMATE=$(grep -oE "(Estimacion puntual|Cota inferior): ~[0-9-]+" /tmp/eval.log | head -1 | grep -oE "[0-9-]+$" || echo "0")
ELO_KIND=$(grep -oE "(Estimacion puntual|Cota inferior|Cota superior|Sin datos)" /tmp/eval.log | head -1)

echo ""
echo "Resultado: $ELO_KIND ~$ELO_ESTIMATE ELO  (threshold: >= $THRESHOLD_ELO)"

if [ "$ELO_ESTIMATE" -ge "$THRESHOLD_ELO" ]; then
    STATUS="ACEPTADO"
    EXIT_CODE=0
else
    STATUS="RECHAZADO - no supera el threshold"
    EXIT_CODE=1
fi

echo ""
echo "======================================================"
echo "  $STATUS"
echo "======================================================"

# Marcar resultado en disco para que el usuario lo vea via SSH
cat > /tmp/RESULT.txt <<EOF
Status:    $STATUS
ELO:       $ELO_KIND ~$ELO_ESTIMATE
Threshold: $THRESHOLD_ELO
Date:      $(date)
EOF

if [ "$EXIT_CODE" -eq 0 ]; then
    echo ""
    echo "Modelo aceptable. Para descargar al local (desde tu Mac):"
    echo "  bash cloud/download.sh ubuntu@HOST"
    echo ""
    echo "Y despues TERMINA la instancia en lambdalabs.com"
else
    echo ""
    echo "El modelo NO supera el threshold. NO descargar."
    echo "Opciones:"
    echo "  - Ajustar hiperparams y volver a lanzar finalize.sh"
    echo "  - Bajar el threshold con: THRESHOLD_ELO=400 bash cloud/finalize.sh"
    echo "  - Investigar el log: cat /tmp/eval.log"
fi

exit $EXIT_CODE
