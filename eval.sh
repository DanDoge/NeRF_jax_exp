#!/bin/bash

CONFIG=llff
DATA_ROOT=./data
ROOT_DIR=./log/"$CONFIG"
if [ $CONFIG == "llff" ]
then
  SCENES="room fern leaves fortress orchids flower trex horns"
  DATA_FOLDER="nerf_llff_data"
else
  SCENES="lego chair drums ficus hotdog materials mic ship"
  DATA_FOLDER="nerf_synthetic"
fi

# launch evaluation jobs for all scenes.
for scene in $SCENES; do
  python -m jaxnerf.eval \
    --data_dir="$DATA_ROOT"/"$DATA_FOLDER"/"$scene" \
    --train_dir="$ROOT_DIR"/"$scene" \
    --chunk=4096 \
    --config=configs/"$CONFIG"
done

# collect PSNR of all scenes.
touch "$ROOT_DIR"/psnr.txt
for scene in $SCENES; do
  printf "${scene}: " >> "$ROOT_DIR"/psnr.txt
  cat "$ROOT_DIR"/"$scene"/test_preds/psnr.txt >> \
    "$ROOT_DIR"/psnr.txt
  printf $'\n' >> "$ROOT_DIR"/psnr.txt
done
