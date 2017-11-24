echo "This script might take a while (a couple of hours)."
echo "You can set parallelism up with the first argument and the working dir with the 2nd argument. Try to use an SSD to speed up things."

#set parallelism
if [ $# -lt 1 ]; then
    N=1
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

git clone https://github.com/allenai/document-qa.git data/triviaqa/document-qa

cp data/triviaqa/config.py data/triviaqa/document-qa/docqa/
export PYTHONPATH=data/triviaqa/document-qa:$PYTHONPATH

echo "Third-party preprocessing..."
python3 data/triviaqa/document-qa/docqa/triviaqa/evidence_corpus.py -n $N
#python3 data/triviaqa/document-qa/docqa/triviaqa/build_span_corpus.py wiki --n_processes $N
python3 data/triviaqa/document-qa/docqa/triviaqa/build_span_corpus.py web --n_processes $N

echo "Converting to Jack format..."
python3 data/triviaqa/convert2jack.py $N

echo "Removing data/triviaqa/document-qa repository, since it is not needed anymore."
rm -r data/triviaqa/document-qa

echo "Find prepared datasets in data/triviaqa/web. If you want, you can safely remove $DOWNLOADPATH/triviaqa-rc now."
