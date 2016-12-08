set -e
trap 'echo ""; echo "ERROR!"; echo ""; echo $BASH_COMMAND' EXIT

PYTHONPATH=. python3 ./quebap/training_pipeline.py --train ./quebap/data/NYT/naacl2013_test.quebap.json --dev ./quebap/data/NYT/naacl2013_test.quebap.json --test ./quebap/data/NYT/naacl2013_test.quebap.json --model boenosupport --supports none --negsamples 1
PYTHONPATH=. python3 ./quebap/training_pipeline.py --train ./quebap/data/NYT/naacl2013_test.quebap.json --dev ./quebap/data/NYT/naacl2013_test.quebap.json --test ./quebap/data/NYT/naacl2013_test.quebap.json --model boenosupport --supports none
PYTHONPATH=. python3 ./quebap/training_pipeline.py --train ./quebap/data/NYT/naacl2013_test.quebap.json --dev ./quebap/data/NYT/naacl2013_test.quebap.json --test ./quebap/data/NYT/naacl2013_test.quebap.json --model boe_nosupport_cands --supports none
PYTHONPATH=. python3 ./quebap/training_pipeline.py --train ./quebap/data/SNLI/snippet_quebapformat_v1.json --dev ./quebap/data/SNLI/snippet_quebapformat_v1.json --test ./quebap/data/SNLI/snippet_quebapformat_v1.json --model bicond_singlesupport_reader
PYTHONPATH=. python3 ./quebap/training_pipeline.py --train ./quebap/data/SNLI/snippet_quebapformat_v1.json --dev ./quebap/data/SNLI/snippet_quebapformat_v1.json --test ./quebap/data/SNLI/snippet_quebapformat_v1.json --model bicond_singlesupport_reader_with_cands
PYTHONPATH=. python3 ./quebap/training_pipeline.py --train ./quebap/data/SNLI/snippet_quebapformat_v1.json --dev ./quebap/data/SNLI/snippet_quebapformat_v1.json --test ./quebap/data/SNLI/snippet_quebapformat_v1.json --model boe_support_cands
PYTHONPATH=. python3 ./quebap/training_pipeline.py --train ./quebap/data/SNLI/snippet_quebapformat_v1.json --dev ./quebap/data/SNLI/snippet_quebapformat_v1.json --test ./quebap/data/SNLI/snippet_quebapformat_v1.json --model boe_nosupport_cands
PYTHONPATH=. python3 ./quebap/training_pipeline.py --train ./quebap/data/SNLI/snippet_quebapformat_v1.json --dev ./quebap/data/SNLI/snippet_quebapformat_v1.json --test ./quebap/data/SNLI/snippet_quebapformat_v1.json --model boe
PYTHONPATH=. python3 ./quebap/training_pipeline.py --train ./quebap/data/SNLI/snippet_quebapformat_v1.json --dev ./quebap/data/SNLI/snippet_quebapformat_v1.json --test ./quebap/data/SNLI/snippet_quebapformat_v1.json --model boenosupport

echo ""
echo "SUCCESS!"

trap EXIT
