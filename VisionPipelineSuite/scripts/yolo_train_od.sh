#!/bin/bash
# Setup parameters
PROJECT="/home/segreto/Documents/Runs"
NAME="SoyCotton-OD-L"
EXIST_OK="True"

DATA="/home/segreto/Documents/Data/SoyCotton-FinalSplit-OD/data.yaml"
MODEL="yolo11l.pt"

BATCH="10"
WORKERS="24"
DEVICE="0"
CACHE="True"

# Hyperparameters
AGNOSTIC_NMS="False"
AMP="True"
AUGMENT="False"
AUTO_AUGMENT="randaugment"
BGR="0.0"
BOX="6.0"
CLS="1.0"
CLOSE_MOSAIC="10"
COS_LR="True"
CROP_FRACTION="1.0"
DEGREES="0.0"
DETERMINISTIC="True"
DFL="0.3"
DNN="False"
DROPOUT="0.04"
DYNAMIC="False"
EPOCHS="250"
ERASING="0.4"
FLIPLR="0.5"
FLIPUD="0.0"
FORMAT="torchscript"
FRACTION="1.0"
HALF="False"
HSV_H="0.015"
HSV_S="0.7"
HSV_V="0.4"
IMGSZ="1024"
INT8="False"
IOU="0.7"
KOBJ="1.0"
LR0="0.0001"
LRF="0.095"
MASK_RATIO="4"
MAX_DET="300"
MIXUP="0.3"
MOMENTUM="0.92"
MOSAIC="1.0"
MULTI_SCALE="False"
NBS="64"
NMS="False"
OPTIMIZER="AdamW"
OVERLAP_MASK="True"
PATIENCE="100"
POSE="12.0"
PRETRAINED="True"
RECT="False"
RESUME="False"
RETINA_MASKS="False"
SAVE="True"
SAVE_CONF="False"
SAVE_CROP="False"
SAVE_PERIOD="-1"
SAVE_TXT="False"
SCALE="0.9"
SEED="0"
SHEAR="0.0"
TRANSLATE="0.1"
VAL="True"
VERBOSE="True"
VID_STRIDE="1"
VISUALIZE="False"
WARMUP_BIAS_LR="0.1"
WARMUP_EPOCHS="4"
WARMUP_MOMENTUM="0.8"
WEIGHT_DECAY="0.0001"

# Override default parameters with command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --*) 
            PARAM=${1:2}
            VALUE=$2
            export "$PARAM"="$VALUE"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Execute YOLO training
yolo train \
    agnostic_nms="$AGNOSTIC_NMS" \
    amp="$AMP" \
    augment="$AUGMENT" \
    auto_augment="$AUTO_AUGMENT" \
    batch="$BATCH" \
    bgr="$BGR" \
    box="$BOX" \
    cache="$CACHE" \
    cls="$CLS" \
    close_mosaic="$CLOSE_MOSAIC" \
    cos_lr="$COS_LR" \
    crop_fraction="$CROP_FRACTION" \
    data="$DATA" \
    degrees="$DEGREES" \
    deterministic="$DETERMINISTIC" \
    device="$DEVICE" \
    dfl="$DFL" \
    dnn="$DNN" \
    dropout="$DROPOUT" \
    dynamic="$DYNAMIC" \
    epochs="$EPOCHS" \
    erasing="$ERASING" \
    exist_ok="$EXIST_OK" \
    fliplr="$FLIPLR" \
    flipud="$FLIPUD" \
    format="$FORMAT" \
    fraction="$FRACTION" \
    half="$HALF" \
    hsv_h="$HSV_H" \
    hsv_s="$HSV_S" \
    hsv_v="$HSV_V" \
    imgsz="$IMGSZ" \
    int8="$INT8" \
    iou="$IOU" \
    kobj="$KOBJ" \
    lr0="$LR0" \
    lrf="$LRF" \
    mask_ratio="$MASK_RATIO" \
    max_det="$MAX_DET" \
    mixup="$MIXUP" \
    model="$MODEL" \
    momentum="$MOMENTUM" \
    mosaic="$MOSAIC" \
    multi_scale="$MULTI_SCALE" \
    name="$NAME" \
    nbs="$NBS" \
    nms="$NMS" \
    optimizer="$OPTIMIZER" \
    overlap_mask="$OVERLAP_MASK" \
    patience="$PATIENCE" \
    pose="$POSE" \
    pretrained="$PRETRAINED" \
    project="$PROJECT" \
    rect="$RECT" \
    resume="$RESUME" \
    retina_masks="$RETINA_MASKS" \
    save="$SAVE" \
    save_conf="$SAVE_CONF" \
    save_crop="$SAVE_CROP" \
    save_period="$SAVE_PERIOD" \
    save_txt="$SAVE_TXT" \
    scale="$SCALE" \
    seed="$SEED" \
    shear="$SHEAR" \
    translate="$TRANSLATE" \
    val="$VAL" \
    verbose="$VERBOSE" \
    vid_stride="$VID_STRIDE" \
    visualize="$VISUALIZE" \
    warmup_bias_lr="$WARMUP_BIAS_LR" \
    warmup_epochs="$WARMUP_EPOCHS" \
    warmup_momentum="$WARMUP_MOMENTUM" \
    weight_decay="$WEIGHT_DECAY" \
    workers="$WORKERS"
