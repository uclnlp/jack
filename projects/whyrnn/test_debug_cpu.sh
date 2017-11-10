export PYTHONPATH=/Users/cdevelder/CODE/UCL/jtr:$PYTHONPATH
SCRIPT="/Users/cdevelder/CODE/UCL/jtr/projects/whyrnn/whyrnn_baseline_with_generator.py"
### SCRIPT="/Users/cdevelder/CODE/UCL/jtr/projects/whyrnn/whyrnn_baseline.py"
echo "[$(date)]... STARTING ${SCRIPT}"
python3 ${SCRIPT} --debug --vocab_max_size 1000000 --vocab_min_freq 1 --init_embeddings uniform --hidden_dim 100 --batch_size 50 --eval_batch_size 100 --learning_rate 0.005 --l2 1e-05 --dropout 0.4 --clip_value 0.0 --epochs 5 --seed 1337 --lowercase --normalize_embeddings --jtr_path /Users/cdevelder/CODE/UCL/jtr --tensorboard_path test_tb/test_debug_cpu.log --model_path test_model_checkpoints 2>&1
echo "[$(date)]... FINISHED running ${SCRIPT}"
