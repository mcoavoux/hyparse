

datadir=$1


## Download and compile evaluator and evaluation parameters.
if [ ! -e evalb_spmrl2013.tar.gz ]
then
    wget http://pauillac.inria.fr/~seddah/evalb_spmrl2013.tar.gz
    tar xvzf evalb_spmrl2013.tar.gz
    cd evalb_spmrl2013.final
    make
    cp spmrl.prm ..
    cp evalb_spmrl ..
    cd ..
    rm -r evalb_spmrl2013.final
fi

mkdir results

# Parse SPMRL
for lang in ARABIC BASQUE FRENCH GERMAN HEBREW HUNGARIAN KOREAN POLISH SWEDISH
do
    for type in dev test
    do
        echo parsing ${lang} ${type}
        echo
        ./nnp -I ${datadir}/${lang}/pred/ptb/${type}/treebank.raw -O results/${lang}_${type}.parsed -m ../pretrained_models/${lang}
        echo evaluating ${lang} ${type}
        echo
        ./evalb_spmrl -X -p spmrl.prm ${datadir}/${lang}/pred/ptb/${type}/treebank.mrg results/${lang}_${type}.parsed > results/${lang}_${type}_eval
        echo 
    done
done