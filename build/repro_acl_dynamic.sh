#!/bin/bash

# Download and compile evaluator if needed
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


DATA=$1
N=$2
LANGS=$3

TYPE=pred
EVALB_PARAMS="-X -p spmrl.prm"

maindir=repro_spmrl_dynamic
mkdir -p $maindir/results

for h in 512
do
for language in ${LANGS}
do
    datadir="$DATA/$language/$TYPE/ptb"
    
    # DATA
    TRAIN="$datadir/train/treebank.tbk"
    DEV="$datadir/dev/treebank.tbk"
    DEVRAW="$datadir/dev/treebank.raw"
    DEVMRG="$datadir/dev/treebank.mrg"
    TESTRAW="$datadir/test/treebank.raw"
    TESTMRG="$datadir/test/treebank.mrg"

    TPLS=../data/dense_templates/generic_gmorph_$language.tpl
    DIMS=../data/dimensions/NN_DIMS_$language

    modeldir=$maindir/$language
    mkdir -p $modeldir

    it=24
    
    for lr in 0.01 0.02
    do
        for dc in 0 1e-6
        do  
        for p in 0.5 0.9
        do
        for k in 8 16
        do
            ((i=i%N)); ((i++==0)) && wait
            (
            
            modelname=$modeldir/h${h}_it${it}_lr${lr}_dc${dc}_layer1_morph_explore_p${p}k${k}
            mkdir $modelname

            echo ./nnt -e $p -k $k -f 2 -a -i $it -t $TPLS -m $modelname -b 1 -l $lr -d $dc -H $h -K $DIMS $TRAIN $DEV > $modelname/trainer_log.txt
            ./nnt -e $p -k $k -f 2 -a -i $it -t $TPLS -m $modelname -b 1 -l $lr -d $dc -H $h -K $DIMS $TRAIN $DEV 2>> $modelname/trainer_log.txt

            ./nnp -I $DEVRAW -O $modelname/parse_result_dev.mrg -m $modelname
            ./evalb_spmrl $EVALB_PARAMS $DEVMRG $modelname/parse_result_dev.mrg > $modelname/evalb_dev.txt

            ./nnp -I $TESTRAW -O $modelname/parse_result_test.mrg -m $modelname
            ./evalb_spmrl $EVALB_PARAMS $TESTMRG $modelname/parse_result_test.mrg > $modelname/evalb_test.txt

            for j in `seq 1 $it`
            do
                for f in "embed_dims" "encoder" "templates" "ttd"
                do
                    cat $modelname/$f > $modelname/iteration$j/$f
                done
                
                ./nnp -I $DEVRAW -O $modelname/iteration$j/parse_result_dev -m $modelname/iteration$j
                ./evalb_spmrl $EVALB_PARAMS $DEVMRG $modelname/iteration$j/parse_result_dev > $modelname/evalb_dev_it$j

                ./nnp -I $TESTRAW -O $modelname/iteration$j/parse_result_test -m $modelname/iteration$j
                ./evalb_spmrl $EVALB_PARAMS $TESTMRG $modelname/iteration$j/parse_result_test > $modelname/evalb_test_it$j
            done
            ) &
        done
        done
    done
    done
done
done
wait
