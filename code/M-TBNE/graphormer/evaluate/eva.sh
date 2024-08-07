    python evaluate.py \
    --user-dir ../../graphormer \
    --num-workers 4 \
    --ddp-backend=legacy_ddp \
    --dataset-name my_dataset \
    --user-data-dir ../../examples/customized_dataset \
    --task graph_prediction \
    --criterion l1_loss  \
    --arch graphormer_slim1 \
    --num-classes 60 \
    --batch-size 24 \
    --save-dir ../../examples/property_prediction/ckpts21\
    --split valid \
    --metric mae \
