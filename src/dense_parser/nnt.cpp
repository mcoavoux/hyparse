
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <getopt.h>
#include <vector>
#include "globals.h"
#include "utilities.h"
#include "nnparser.h"
#include "treebank.h"
#include "dense_encoder.h"
#include "layers.h"
#include "neural_net.h"

#ifdef ENABLE_PARALLEL
   #include "omp.h"
#endif


using namespace std;

void display_help_message(){

  cerr << endl << "This is the trainer command line for a dense features feed-forward neural network parser."<<endl;
  cerr <<         " When used without option, the architecture contains 2 layers : lookup + output (softmax or perceptron)." << endl;
  cerr <<         " Use -H option to add any number of additional hidden layers." << endl;
  cerr << " Usage:"<<endl<<endl;
  cerr << "    nnt [options] train_filename dev_filename" << endl << endl;
  cerr << "    -h   --help                        : prints this help message and exit." <<endl;
  cerr << "    -i   --iterations          [INT]   : sets the number of epochs " <<endl;
  cerr << "    -t   --templates-filename  [STR]   : sets the file where to seek for templates" <<endl;
  cerr << "    -m   --model-name          [STR]   : sets the directory path where to save the model" <<endl;
  cerr << "    -b   --beam-size           [INT]   : sets the size of the beam" <<endl;
  cerr << "    -l   --learning-rate       [FLOAT] : sets the learning rate of the network" << endl;
  cerr << "    -d   --decrease-constant   [FLOAT] : sets the decrease constant for the learning rate of the network" << endl;
  cerr << "    -H   --hidden-units        [INT]   : adds a hidden layer with <int> units (use this option several times for a very deep architecture)" << endl;
  cerr << "    -D   --dimension           [INT]   : sets the dimensionality of the embeddings (all symbol types will have same dimensionality)" << endl;
  cerr << "    -K   --dim-filename        [STR]   : sets the dimensionality of the embeddings (use customized dimensionality for each category of symbol)" << endl;
  cerr << "    -L   --lookuptable         [STR]   : uses pre-trained word vectors contained in lookup table" << endl;
  cerr << "    -p   --lr-down                     : uses down scaling strategy for decreasing learning rate. (lr is halfed every time accuracy on dev diminishes)" << endl;
  cerr << "    -P   --lr-adjust                   : uses adjust scaling strategy for learning rate (lr is halfed or multiplied by 1.1)" << endl;
  cerr << "    -T   --threads             [INT]   : trains in parallel mode with maximum t processors [if available] (default = 1)" <<endl;
  //cerr << "    -R   --shallow                   : uses a shallow network without a non-linear hidden layer." << endl;
  //cerr << "    -e   --maxent                    : uses a MaxEnt classifier with unigram features (no 'complex' features" << endl;
  //cerr << "    -o   --dropout                     : uses dropout regularisation (Hinton et al. 2012). Dropout parameter is 0.5. [NOT SUPPORTED YET]" << endl;
  cerr << "    -A   --adagrad                     : uses adagrad (Duchi et al. 2011) instead of classical SGD" << endl;
  cerr << "    -a   --asgd                        : uses ASGD (see Bottou et al.2010) instead of classical SGD" << endl;
  cerr << "    -G   --gradient-check              : performs a gradient checking on several NN models, prints average distances between analytical and empirical gradients." << endl;
  cerr << "    -g   --gradient-clipping   [FLOAT] : clips the gradient if its norm is superior to [FLOAT]" << endl;
  cerr << "    -f   --activation-function [INT]   : "<< endl;
  cerr << "                                (0) tanh"<<endl;
  cerr << "                                (1) cube"<<endl;
  cerr << "                                (2) rectified linear" << endl;
  cerr << "    -O   --objective           [INT]   : "<< endl;
  cerr << "                                (0) cross entropy (optimise likelihood)"<<endl;
  cerr << "                                (1) perceptron style" << endl;
  cerr << "    -r   --regularization      [FLOAT] : adds L2 regularization with parameter <float>" << endl;
  cerr << "    -R   --logistic-regression         : use a multinomial logistic regression instead of a neural net (no distributed representations" << endl;
  cerr << "  Learning with exploration (dynamic oracle)" << endl;
  cerr << "    -k   --from-iteration      [INT]   : start exploring from iteration k (default = 1)" <<endl;
  cerr << "    -e   --exploration         [FLOAT] : explores non gold path with probability p (default = 0.1)" << endl;
  cerr << "  Convolution over characters" << endl;
  cerr << "    -F   --filter-size         [INT]   ; adds a convolutional layer over characters of lexical features (tokens) to compute a char-based embedding and specifies the size of the convolutional filter" << endl;
  cerr << "    -C   --char-embedding      [INT]   : size of character embeddings (default=20)" << endl;
  cerr << "    -Y   --char-based-embedding[INT]   : size of char-based word embedding" << endl;
  cerr << "  Global training (see Weiss et al. 2015)" << endl;
  cerr << "    -M   --global-perceptron           : retrain globally a neural net model with an additional perceptron layer (see Weiss et al. 2015)" << endl;
  cerr << endl;
}

void display_summary(int beam_size,int epochs,size_t train_size,size_t dev_size){

  cerr << "***** Training setup summary **********"<<endl;
  cerr << "   Max iterations       : " << epochs <<endl;
  cerr << "   Beam size            : " << beam_size <<endl;
  cerr << "   Training set size    : " << train_size <<endl;
  cerr << "   Development set size : " << dev_size <<endl;
  cerr << "****************************************"<<endl<<endl;
}


string train_path = "../data/ftb20-train-ttd.tbk";
string dev_path = "../data/ftb20-dev-ttd.tbk";
string tpl_path = "../data/dense_ttd.tpl";
string model_path = "default_dense.model";
string dim_filename;
string lookup_filename;

int num_procs = 1;
int beam = 1;
int iterations = 15;

NeuralNetParams params;
int learning_rate_policy = POWER;

vector<int> coldims;

int dimension = 50;     // dimension of embeddings


Treebank::Decoration transformation;

bool exploration = false;
int exploration_k = 0;
double exploration_p = 0.1;

bool neural_net = true;

bool global_training = false;

int main(int argc,char *argv[]){

    srand(1);

    char c;
    /*************************************************
     *************************************************
     *************************************************/
    while(true){
        static struct option long_options[] ={
            {"help",no_argument,0,'h'},
            {"iterations",required_argument,0,'i'},
            {"templates-filename",required_argument,0,'t'},
            {"model-name",required_argument,0,'m'},
            {"beam-size",required_argument,0,'b'},
            {"learning-rate", required_argument, 0, 'l'},
            {"decrease-constant", required_argument, 0, 'd'},
            {"hidden-units", required_argument, 0, 'H'},
            {"dimension", required_argument, 0, 'D'},
            {"dim-filename", required_argument, 0, 'K'},
            {"lookuptable", required_argument, 0, 'L'},
            {"lr-down", no_argument, 0, 'p'},
            {"lr-adjust", no_argument, 0, 'P'},
            {"threads",required_argument,0,'T'},
            {"adagrad", no_argument, 0, 'A'},
            {"objective", required_argument, 0, 'I'},
            {"activation-function", required_argument, 0, 'f'},
            {"gradient-check", no_argument, 0, 'G'},
            {"gradient-clipping", no_argument, 0, 'g'},
            {"regularization", required_argument, 0, 'r'},
            {"exploration", required_argument, 0, 'e'},
            {"from-iteration", required_argument, 0, 'k'},
            {"filter-size", required_argument, 0, 'F'},
            {"char-embedding", required_argument, 0, 'C'},
            {"char-based-embedding", required_argument, 0, 'Y'},
            {"logistic-regression", no_argument, 0, 'R'},
            {"global-perceptron", no_argument, 0, 'M'},
            {"asgd", no_argument, 0, 'a'}
        };
        int option_index = 0;
        c = getopt_long (argc, argv, "?hi:t:m:b:l:d:H:D:K:L:pPT:AGO:f:r:g:e:k:F:C:Y:RMa",long_options, &option_index);
        if(c==-1){break;}
        switch(c){
            case '?': display_help_message();  exit(0);
            case 'h': display_help_message();  exit(0);
            case 'i': iterations = atoi(optarg);                break;
            case 't': tpl_path = string(optarg);                break;
            case 'm': model_path = string(optarg);              break;
            case 'b': beam = atoi(optarg);                      break;
            case 'l': params.learning_rate = atof(optarg);      break;
            case 'd': params.decrease_constant = atof(optarg);  break;
            case 'H': params.n_hidden.push_back(atoi(optarg));  break;
            case 'D': dimension = atoi(optarg);                 break;
            case 'K': dim_filename = string(optarg);            break;
            case 'L': lookup_filename=string(optarg);           break;
            case 'p': learning_rate_policy = DOWN;              break;
            case 'P': learning_rate_policy = ADJUST;            break;
            case 'T': num_procs = atoi(optarg);                 break;
            case 'A': params.ada_grad = true;                   break;
            case 'O': params.loss_function = atoi(optarg);      break;
            case 'f': params.hidden_activation = atoi(optarg);  break;
            case 'r': params.regularization = true;
                      params.reg_lambda = atof(optarg);         break;
            case 'g': params.gradient_clipping=true;
                      params.clipping_threshold=atof(optarg);   break;
            case 'G':
                      omp_set_num_threads(num_procs);
                      NeuralNetwork::run_gradient_checking();
                      exit(0);
                      break;
            case 'e': exploration = true;
                      exploration_p = atof(optarg);             break;
            case 'k': exploration_k = atoi(optarg);             break;
            case 'F': params.convolution = true;
                      params.filter_size = atoi(optarg);        break;
            case 'C': params.char_embeddings_dimension = atoi(optarg);              break;
            case 'Y': params.char_based_word_embeddings_dimension = atoi(optarg);   break;
            case 'R': neural_net = false;                       break;
            case 'M': global_training = true;                   break;
            case 'a': params.asgd = true;                       break;
        }
    }
    if (optind == 1){
        cerr << "Missing args, aborting" << endl;
        exit(0);
    }//abort if no arg at all
    if(optind < argc+1){
      train_path = string(argv[optind]);
      dev_path = string(argv[optind+1]);
    }else{
      cerr << "Missing either training or dev file name ... aborting." <<endl;
      exit(1);
    }
    /**************************************************************************
     **************************************************************************
     **************************************************************************/

    #ifdef ENABLE_PARALLEL
        omp_set_num_threads(num_procs);
        cerr << "Number of processors used = " << omp_get_max_threads() << endl;
    #endif

    //TREEBANK SETUP
    Treebank base_tbk(train_path);
    TemplateTypeDefinition ttd(base_tbk.colnames);
    DenseEncoder templates(tpl_path, ttd);
    transformation = Treebank::NO_DECORATIONS;
    Treebank trans_tbk;
    Treebank dev_bank(dev_path);
    Treebank dev_binary;
    base_tbk.transform(trans_tbk,templates.has_morpho(), 0,transformation);
    dev_bank.transform(dev_binary,templates.has_morpho(), 0,transformation);

    trans_tbk.update_encoder();
    dev_binary.update_encoder();

    SrGrammar grammar = SrGrammar(trans_tbk, dev_binary, templates.has_morpho());

    trans_tbk.update_encoder();   // The PS symbols must be encoded before the lookup table (max code^2 size table is used further)



    // Parameters initialisation

    if (global_training){
        string dirname = model_path + "_global_training_b" + std::to_string(beam) + "_it"+std::to_string(iterations);
        mkdir(dirname.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);

        display_summary(beam, iterations, trans_tbk.size(),dev_bank.size());

        cerr << "Loading model ..." << endl;
        IntegerEncoder::get()->load(model_path + "/encoder");
        IntegerEncoder::get()->save(dirname + "/encoder");
        GlobalNnParser global_parser(model_path, false);
        global_parser.summary(cerr);

        cerr << "Training ..." << endl;
        global_parser.train_global_model(trans_tbk, dev_bank, beam, iterations, dirname);
        global_parser.save(dirname, NULL);
        global_parser.save_global_parameters(dirname);

        copy_file(model_path+"/embed_dims", dirname + "/embed_dims");
        copy_file(model_path+"/ttd", dirname + "/ttd");
        copy_file(model_path+"/templates", dirname + "/templates");


        cerr << "Training complete" << endl;
    }else{
        mkdir(model_path.c_str(),S_IRUSR | S_IWUSR | S_IXUSR);//creates model dir
        NnSrParser nnp(templates, neural_net);
        nnp.set_lr_policy(learning_rate_policy);

        params.n_output = grammar.num_actions();
        params.n_input_features = templates.ntemplates();

        params.n_input.clear();
        if (! dim_filename.empty()){
            read_dimension_file(dim_filename, coldims);

            for (int i = 0; i < params.n_input_features; i++){
                params.n_input.push_back(coldims[templates.get_colidx(i)]);
            }
        }else{
            coldims.clear();
            for (int i = 0; i < 30; i++) coldims.push_back(dimension);
            for (int i = 0; i < params.n_input_features; i++) params.n_input.push_back(dimension);
        }
        params.embeddings_dimensions = coldims;

        params.lexical_features.clear();
        for (int i = 0; i < templates.ntemplates(); i++){
            if (templates.get_colidx(i) == 1){
                params.lexical_features.push_back(i);
            }
        }

        cerr << endl;
        display_summary(beam, iterations, trans_tbk.size(),dev_bank.size());

        cerr << "Initialising parameters ..." << endl;
        nnp.initialize_params(params);
        nnp.summary(cerr);


        if (! lookup_filename.empty()){
            cerr << "Loading lookup table ..." << endl;
            nnp.load_lu(lookup_filename);
            cerr << "Lookup table loaded" << endl;
        }


        //SAVE MODEL
        ofstream out_summary(model_path+"/summary");
        nnp.summary(out_summary);
        IntegerEncoder::get()->save(model_path+"/encoder");
        ttd.save(model_path+"/ttd");
        copy_file(tpl_path,model_path+"/templates");//copies the templates to model dir

        //copy_file(dim_filename, model_path+"/embed_dims");    // bugs : if -D is used, creates an empty file, memory leak when loading embeddings
        ofstream out_embeddims(model_path+"/embed_dims");
        out_embeddims << coldims.size() << endl;
        for (int i = 0; i < coldims.size(); i++){
            out_embeddims << coldims[i] << " ";
        }
        out_embeddims.close();


        nnp.set_tmp_filename(model_path);

        cerr << endl << endl;
        cerr << "Training ..." << endl;
        string logfile = model_path +"/train.log";
        LLOGGER_START(logfile.c_str());//turns logger on
        nnp.train_local_model(trans_tbk, dev_bank, 1, iterations, exploration, exploration_k, exploration_p);
        cerr << "Training completed" << endl;
        cerr << "Dumping model" << endl;
        LLOGGER_STOP();//turns logger off
        nnp.save(model_path, NULL);

//        //SAVE MODEL
//        ofstream out_summary(model_path+"/summary");
//        nnp.summary(out_summary);
//        IntegerEncoder::get()->save(model_path+"/encoder");
//        ttd.save(model_path+"/ttd");
//        copy_file(tpl_path,model_path+"/templates");//copies the templates to model dir

//        //copy_file(dim_filename, model_path+"/embed_dims");    // bugs : if -D is used, creates an empty file, memory leak when loading embeddings
//        ofstream out_embeddims(model_path+"/embed_dims");
//        out_embeddims << coldims.size() << endl;
//        for (int i = 0; i < coldims.size(); i++){
//            out_embeddims << coldims[i] << " ";
//        }
//        out_embeddims.close();
    }


    IntegerEncoder::kill();
    LLOGGER_STOP();//turns logger off
    cerr << "done."<<endl;
    return 0;
}

