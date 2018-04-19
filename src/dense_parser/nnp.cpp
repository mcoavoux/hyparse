
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <string>
#include "globals.h"
#include "utilities.h"
#include "nnparser.h"
#include "treebank.h"
#include "dense_encoder.h"

#ifdef ENABLE_PARALLEL
   #include "omp.h"
#endif

using namespace std;

string infile;
string outfile;
string conllfile;
string nativefile;
string modelfile;
int K=1;
int c;
int processors = 1;

bool neural_net = true;
bool global_perceptron = false;

void display_help(){
    cerr << "This is the command line for the Shift Reduce Neural Network parser"<<endl;
    cerr << "Usage:"<<endl;
    cerr << "        -h --help                    displays this message and exits." << endl;
    cerr << "        -I --infile      [FILENAME]  sets the input file to FILENAME (defaults to <STDIN>)" <<endl;
    cerr << "        -O --outfile     [FILENAME]  sets the output ptb file to FILENAME (defaults to <STDOUT>)" <<endl;
    cerr << "        -C --conllx      [FILENAME]  sets the output conll file to FILENAME (defaults to none)" <<endl;
    cerr << "        -N --native      [FILENAME]  sets the output native file to FILENAME (defaults to none)" <<endl;
    cerr << "        -m --model       [DIRNAME]   sets the model DIRNAME (mandatory)"   <<endl;
    cerr << "        -p --processors  [INT]       sets the number of processors to use" <<endl;
    cerr << "        -K --kbest       [INT]       sets the number of solutions to output" <<endl;
    cerr << "        -R --logistic-regression     uses a multinomial logistic regression instead of a neural net" << endl;
    cerr << "        -G --global-perceptron       uses a global perceptron (see Weiss et al. 2015)" << endl;
    cerr << endl;
    exit(1);
}

int main(int argc,char *argv[]){

    while(true){
        static struct option long_options[] ={
            {"help",no_argument,0,'h'},
            {"infile",required_argument,0,'I'},
            {"model",required_argument,0,'m'},
            {"kbest",required_argument,0,'K'},
            {"outfile",required_argument,0,'O'},
            {"conllx",required_argument,0,'C'},
            {"native",required_argument,0,'N'},
            {"processors",required_argument,0,'p'},
            {"logistic-regression", no_argument, 0, 'R'},
            {"global-perceptron", no_argument, 0, 'G'},
        };
        int option_index = 0;
        c = getopt_long (argc, argv, "I:hm:K:O:C:N:Rp:G",long_options, &option_index);
        if(c==-1){break;}
        switch(c){
            case 'I': infile = optarg;                  break;
            case 'm': modelfile = optarg;               break;
            case 'K': K = atoi(optarg);                 break;
            case 'O': outfile = optarg;                 break;
            case 'C': conllfile = optarg;               break;
            case 'N': nativefile = optarg;              break;
            case 'p': processors = atoi(optarg);        break;
            case 'R': neural_net = false;               break;
            case 'h': display_help();                   break;
            case 'G': global_perceptron = true;             break;
        }
    }
    if(modelfile.empty()){cerr<<"Oops! no model provided"<<endl<<"aborting."<<endl;exit(1);}

    #ifdef ENABLE_PARALLEL
        omp_set_num_threads(processors);
        cerr << "Number of processors used = " << omp_get_max_threads() << endl;
    #endif

    //Sets up output streams
    PennTreebankOutStream ptbstream(true);
    ConllOutStream conllstream;
    NativeOutStream nativestream;

    cerr << "Opening output files" << endl;
    if (!outfile.empty()){ptbstream = PennTreebankOutStream(outfile);}
    if(!conllfile.empty()){conllstream = ConllOutStream(conllfile);}
    if(!nativefile.empty()){nativestream = NativeOutStream(nativefile);}
    cerr << "Opening output files: done" << endl;

    //Start up the logger
    cerr << "Opening logging file" << endl;
    string logfile = modelfile+"/parse.log";
    PLOGGER_START(logfile.c_str());
    cerr << "Opening logging file: done" << endl;

    //Loads the parser
    cerr << "Loading string encoder" << endl;
    IntegerEncoder::get()->load(modelfile+"/encoder");
    cerr << "Loading string encoder: done" << endl;

    //IntegerEncoder::get()->save("foo/encoder2");

    cerr << "Initializing lexer" << endl;
    AbstractLexer *lex = new TbkLexer();
    cerr << "Initializing lexer: done" << endl;
    if (global_perceptron){
        GlobalNnParser nnp(modelfile, true);
        nnp.summary(cerr);

        //RUN
        //if(!infile.empty() && processors == 1){
        if(!infile.empty()){
            ifstream input_source(infile);
            nnp.parse_corpus(lex,input_source,K,ptbstream,conllstream,nativestream);
            input_source.close();
        }else{// if(processors == 1){
            nnp.parse_corpus(lex,cin,K,ptbstream,conllstream,nativestream);
        }
    }else{
        cerr << "Loading models" << endl;
        NnSrParser nnp(modelfile, neural_net);
        cerr << "Loading models: done" << endl;
        
        cerr << "Precomputing hidden layer ... " << endl;
        nnp.set_test_mode(true);
        cerr << "Precomputing hidden layer: done" << endl;
        //IntegerEncoder::get()->save("foo/encoder3");

        nnp.summary(cerr);

        //RUN
        //if(!infile.empty() && processors == 1){
        if(!infile.empty()){
            cerr << "Reading input data" << endl;
            ifstream input_source(infile);
            cerr << "Reading input data: done" << endl;
            cerr << "Starting parsing" << endl;
            nnp.parse_corpus(lex,input_source,K,ptbstream,conllstream,nativestream);
            cerr << "Starting parsing: done" << endl;
            
            input_source.close();
        }else{// if(processors == 1){
            nnp.parse_corpus(lex,cin,K,ptbstream,conllstream,nativestream);
        }
    }
    delete lex;

    conllstream.close();
    ptbstream.close();
    nativestream.close();
    PLOGGER_STOP();

    return 0;
}
