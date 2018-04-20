echo "This script might take a while (a couple of hours)."
echo "You can set parallelism up with the first argument and the working dir with the 2nd argument. Try to use an SSD to speed up things."

#set parallelism
if [ $# -lt 1 ]; then
    N=`nproc --all`
else
    N=$1
fi

if [ $# -lt 2 ]; then
    DOWNLOADPATH=data/triviaqa
else
    DOWNLOADPATH=$2
fi

export TRIVIAQA_HOME=$DOWNLOADPATH/triviaqa-rc
if [ ! -d $TRIVIAQA_HOME ]; then
    echo "Downloading and extracting dataset..."
    wget -P $DOWNLOADPATH http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz
    tar xf $DOWNLOADPATH/triviaqa-rc.tar.gz -C $DOWNLOADPATH
fi

git clone https://github.com/dirkweissenborn/document-qa.git data/triviaqa/document-qa

cp data/triviaqa/config.py data/triviaqa/document-qa/docqa/
export PYTHONPATH=data/triviaqa/document-qa:$PYTHONPATH

echo "Third-party preprocessing..."
python3 data/triviaqa/document-qa/docqa/triviaqa/evidence_corpus.py -n $N
python3 data/triviaqa/document-qa/docqa/triviaqa/build_span_corpus.py wiki --n_processes $N
python3 data/triviaqa/document-qa/docqa/triviaqa/build_span_corpus.py web --n_processes $N

echo "Converting to Jack format..."
# We only extract the top (tf/idf) 6 paragraphs (merged/split to maximum of 600 tokens each) to save disk space.
# In case you want all paragraphs, change 6 to -1.

# for dev and test take all paragraphs
python3 data/triviaqa/convert2jack.py web-dev $N -1 600
python3 data/triviaqa/convert2jack.py wiki-dev $N -1 600
python3 data/triviaqa/convert2jack.py web-test $N -1 600
python3 data/triviaqa/convert2jack.py wiki-test $N -1 600

# for training we only need the top k paragraphs
python3 data/triviaqa/convert2jack.py web-train $N 4 600
python3 data/triviaqa/convert2jack.py wiki-train $N 6 600

echo "Removing data/triviaqa/document-qa repository, since it is not needed anymore."
rm -rf data/triviaqa/document-qa

echo "Find prepared datasets in data/triviaqa/. If you want, you can safely remove $DOWNLOADPATH/triviaqa-rc now."
