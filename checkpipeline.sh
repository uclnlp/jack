set -e
trap 'echo ""; echo "ERROR!"; echo ""; echo $BASH_COMMAND; echo ""; echo "DATA=$DATA"' EXIT

export PYTHONPATH=$PYTHONPATH:.

DATA=quebap/data/TACKBP/tackbp_snippet.json

python3 ./quebap/training_pipeline.py --train $DATA --dev $DATA --test $DATA --model boenosupport --supports none --negsamples 1
python3 ./quebap/training_pipeline.py --train $DATA --dev $DATA --test $DATA --model boenosupport --supports none
python3 ./quebap/training_pipeline.py --train $DATA --dev $DATA --test $DATA --model boe_nosupport_cands --supports none

DATA=quebap/data/SNLI/snippet_quebapformat_v1.json

python3 ./quebap/training_pipeline.py --train $DATA --dev $DATA --test $DATA --model bicond_singlesupport_reader
python3 ./quebap/training_pipeline.py --train $DATA --dev $DATA --test $DATA --model bicond_singlesupport_reader_with_cands
python3 ./quebap/training_pipeline.py --train $DATA --dev $DATA --test $DATA --model boe_support_cands
python3 ./quebap/training_pipeline.py --train $DATA --dev $DATA --test $DATA --model boe_nosupport_cands
python3 ./quebap/training_pipeline.py --train $DATA --dev $DATA --test $DATA --model boe
python3 ./quebap/training_pipeline.py --train $DATA --dev $DATA --test $DATA --model boenosupport

echo ""
echo "SUCCESS!"

trap EXIT
