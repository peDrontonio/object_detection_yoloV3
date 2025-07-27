#!/usr/bin/env bash

# Quantas vezes você quer repetir (com seeds diferentes)
REPEATS=5

# Quantos epochs máximos em cada run
MAX_EPOCHS=300

# Loop de experimentos
for i in $(seq 1 $REPEATS); do
  SEED=$((100 + i))    # semente diferente (100, 101, 102…)
  RUN_NAME="${NAME}_exp${i}"
  
  echo "=== RUN $i: seed=$SEED name=$RUN_NAME ==="
  
  # exporta a seed pro teu script
  export SEED
  export NAME="$RUN_NAME"
  export MAX_EPOCHS
  
  # chama teu script; ele já deve ler $SEED e $MAX_EPOCHS
  bash yolo_train_od.sh
done
