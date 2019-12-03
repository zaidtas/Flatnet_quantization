CHECKPOINT='flatnet_new_randtoep'
MODEL_ROOT='flatnet_new_randtoep'
INIT='Random'
python main.py --dev_id 6 --wtmse 1 --wtp 0 --wta 0.6 --disPreEpochs 5 --numEpoch 25 --modelRoot $MODEL_ROOT --init $INIT --checkpoint $CHECKPOINT
