
# hyparse

Code for [Neural Greedy Constituent Parsing with Dynamic Oracles](http://www.aclweb.org/anthology/P/P16/P16-1017.pdf) (ACL 2016) [[bib]](http://www.aclweb.org/anthology/P/P16/P16-1017.bib).

# Compile

Dependencies: Eigen, boost, g++.

    mkdir build

    # Download headers for boost and Eigen
    cd lib
    sh get_dependencies.sh
    cd ..

    # Compile
    cd src/dense_parser
    make
    cd ../..


Two executable files should appear in the `build` folder: `nnt` (trainer) and `nnp` (parser).

# Train

The data must be in a pseudo-xml format (see example files) where each
line is either a constituent markup (e.g. `<NP>` or `</NP>`)
or a token and a number of associated fields (typically morphological attributes).
Headedness information is indicated with the suffix `-head` on non-terminal
symbols or POS tags.

Example run to train a model:

    #./nnt -i <iterations> -t <templates> -m <modelname> -l <learning rate> -d <decrease constant for lr> 
    #  -H <int>     adds a hidden layer with int units
    #  -D <int>     size of embeddings (customizable with -K option)
    #  -a           use Averaged SGD
    #  -f 2         use ReLU activation (0 is tanh, 1 is cube)
    #  -k <int>     training with exploration every other 8 sentences
    #  -e <double>  when training with exploration, follow predicted action (rather than best) with probability <double>
    #  <train set> <dev set>
    
    ./nnt -i 20 -t ../data/dense_templates/generic_ttd.tpl -m model_name  -l 0.03 -d 1e-7 -H 128 -H 64 -D 32 -a -f 2 -k 8 -e 0.9  ../data/test1_ttd.tbk ../data/test1_ttd.tbk 


# Parse


Example parse with the new trained model:

    # ./nnp -I <input> -O <output> -m <model>
    ./nnp -I ../data/test1_ttd.raw -O output -m model_name/ 


# Replicate Final Results


These lines replicate the results on the dev and test sets (SPMRL)
in the 'dynamic' setting, using the provided pretrained models.
The data is not included in the repository, due to license restrictions.

    cd build
    bash parse.bash <path to data>

# Retrain the parser

Static setting:

    cd build
    # bash repro_acl_static.sh <path to data> <threads> <list of languages>
    bash repro_acl_static.sh ../../SPMRL2015 4 "POLISH SWEDISH"
    # -> launch 4 experiments per language (choose a number of threads that is a multiple of 4)

Dynamic setting:

    cd build
    # bash repro_acl_static.sh <path to data> <threads> <list of languages>
    bash repro_acl_dynamic.sh ../../SPMRL2015 16 "POLISH SWEDISH"
    # 16 experiments per language


NB:
- Hyperparameters are hardcoded in the experiment scripts.
- Training time should take less than 24h for small corpora (Polish, Swedish, Basque, Hebrew),
  several days for Korean, Hungarian, French and Arabic, and around 7 days for German.

