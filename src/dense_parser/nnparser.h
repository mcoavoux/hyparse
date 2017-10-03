#ifndef NNPARSER_H
#define NNPARSER_H

#include <tuple>
#include <vector>
#include <algorithm>
#include "globals.h"
#include "state_graph.h"
#include "srgrammar.h"
#include "lexer.h"
#include "treebank.h"
#include "dense_encoder.h"
#include "neural_net.h"
#include "dynamic_oracle.h"

#define DBG(x) std::cerr << x << std::endl;

enum {POWER, DOWN, ADJUST};

class NnSrParser{

public :

    NnSrParser(bool neural_net);
    NnSrParser(DenseEncoder const &dense_encoder, bool neural_net);
    NnSrParser(const string &modelfile, bool neural_net);
    virtual ~NnSrParser(){
        delete nn;
    }

    //parse a sentence
    virtual AbstractParseTree* predict_one(InputDag &input_sequence);
    pair<pair<float, float>, float> local_eval_one(AbstractParseTree const *ref);
    pair<float, float> local_eval_model(const Treebank &eval_set);

    void parse_corpus(AbstractLexer *lex,istream &input_source,int Kbest,ParserOutStream &ptbstream,ParserOutStream &conllstream,ParserOutStream &nativestream);
    AbstractParseTree* predict_kth_parse(int k, InputDag const &input_sequence);
    size_t get_num_parses()const;
    //train a model
    void train_local_model(Treebank &train_set, Treebank const &dev_set, size_t beam_size, size_t epochs, bool explore=false, int K = 0, double exploration_p = 0.1);
    void extract_local_dataset(Treebank & set, vector<vector<int> > &X, vector<int> & Y);
    //eval a model
    tuple<float,float,float> eval_model(Treebank const &eval_set);

    //I/O
    void save(const string &dirname, Classifier *classifier); //saves the model into directory
    void reload(string const &modelfile);
    void initialize_params(NeuralNetParams &params);
    virtual void summary(ostream &os);
    void load_lu(const std::string & lu_filename);
    void set_lr_policy(int policy);
    void set_test_mode(bool t);
    void set_tmp_filename(string tmp_file);

protected:

    int local_learn_one(AbstractParseTree const *ref);
    int local_learn_one_exploration(AbstractParseTree const *root, DynamicOracle &oracle, double p);

    vector<float> y_scores ;

    //Main components
    TSSBeam       beam;
    SrGrammar     grammar;

    vector<int> idx_vec;
    DenseEncoder dense_encoder;


    //NeuralNetwork nn;
    Classifier *nn;

    int learning_rate_policy;
    int objective;

    string tmp_filename;
};

class GlobalNnParser : public NnSrParser{
    NeuralPerceptron model;
    NeuralPerceptron avg_model;
    Vec perceptron_input;
    Vec perceptron_output;
    int input_size;
    size_t beam_size;

public:
    //GlobalNnParser(const char* modelfile, bool neural_net) : GlobalNnParser(string(modelfile), neural_net){}
    GlobalNnParser(const string &modelfile, bool test_time) : NnSrParser(modelfile, true){
        input_size = nn->get_concatenated_layers_size();
        cerr << "Size of concatenated layers : " << input_size << endl;
        model = NeuralPerceptron(grammar.num_actions(), input_size);
        avg_model = NeuralPerceptron(grammar.num_actions(), input_size);
        nn->set_test_mode(true);
        perceptron_input = Vec::Zero(input_size);
        perceptron_output = Vec::Zero(grammar.num_actions());

        ifstream in_beam(modelfile + "/beam");
        in_beam >> beam_size;
        in_beam.close();
        cerr << "beam size (loaded from file) : " << beam_size << endl;
        beam = TSSBeam(beam_size, grammar);
        if (test_time){
            model.load(modelfile);
        }
    }
    virtual ~GlobalNnParser(){}

    AbstractParseTree* predict_one(InputDag &input_sequence){
        StateSignature sig;
        size_t N = input_sequence.size();
        size_t eta = dense_encoder.has_morpho() ? (2 * N)-1 : (3 * N)-1;

        beam.reset();
        vector<bool> actions(grammar.num_actions());
        for(int i = 0; i < eta;++i){
          for(int k = 0; k < beam.top_size();++k){
              ParseState *stack_top = beam[k];
              stack_top->get_signature(sig,&input_sequence,N);
              dense_encoder.encode(idx_vec, sig, true);
              grammar.select_actions(actions,stack_top->get_incoming_action(),sig);

              //nn->scores(idx_vec, y_scores, actions);
              nn->fprop(idx_vec, 0, actions);
              nn->get_concatenated_layers(perceptron_input);
              model.dot(perceptron_input, perceptron_output);
              for (int j = 0; j < y_scores.size(); j++){
                  y_scores[j] = perceptron_output[j];
              }

              grammar.select_actions(y_scores,stack_top->get_incoming_action(),sig);
              beam.push_candidates(stack_top, y_scores);
          }
          beam.next_step(grammar, input_sequence, N);
        }
        if (beam.has_best_parse()){return beam.best_parse(input_sequence);}
        else{return NULL;}
    }

    void summary(ostream &os){
        NnSrParser::summary(os);
        os << "- Training a global perceptron over hidden hidden and output layers of neural net" << endl;
        os << "- size of input : " << input_size << endl;
        os << "- beam size : " << beam_size << endl;
    }

    struct EpochSummary{
        int epoch_id;
        float loss;
        float trainF;
        float devF;
        float loc_acc_train;
        float loc_acc_dev;
        float eta;
        void init(){
            loss = 0.0;
            trainF = 0;
            devF = 0;
            loc_acc_train = 0;
            loc_acc_dev   = 0;
            eta = 0.0;
        }
        void display(){
            cerr << "Iteration "<<epoch_id << " : loss="<<loss<< " eta=" << eta <<" train-F-Score="<<trainF<<" dev-F-Score="<<devF;
            if(loc_acc_train > 0){cerr << " train local acc="<<loc_acc_train<< " dev local acc="<<loc_acc_dev;}
            cerr << endl;
        }
    };


    void train_global_model(Treebank &train_set,
                            Treebank const &dev_set,
                            size_t beam_size,
                            size_t epochs,
                            const string model_path){
        this->beam_size = beam_size;

        //#ifdef ENABLE_LEARN_LOGGER
        // Evaluation : accuracy + Fscore, need to have a binary version for extracting derivations and an n-ary version to compute Fscore
        Treebank train_sample;
        train_set.sample_trees(train_sample,dev_set.size());
        Treebank train_sample_binary(train_sample);
        train_sample.detransform();
        Treebank tmp(dev_set);
        Treebank dev_set_binary;
        tmp.transform(dev_set_binary, dense_encoder.has_morpho());
        //#endif


        cerr << "Setting beam size to " << beam_size << endl;
        beam = TSSBeam(beam_size,grammar);

        //reset registers
        idx_vec.resize(dense_encoder.ntemplates());
        y_scores.resize(grammar.num_actions());

        //setup stats
        EpochSummary summary;
        float T = 1;
        for(unsigned int e = 1; e < epochs+1;++e){

            int n_updates = 0;
            summary.init();
            summary.epoch_id = e;

            //train_set.shuffle();
            for(int i = 0; i < train_set.size();++i){
                n_updates += global_learn_one(train_set[i], T);
                T ++;
            }


            NeuralPerceptron tmp_params(model);
            NeuralPerceptron tmp_avg(avg_model);
            tmp_avg /= T;
            tmp_params -= tmp_avg;
            tmp_params.save(model_path, e);

            summary.loss = n_updates;

            tuple<float,float,float> evalT = eval_model(train_sample);
            summary.trainF = std::get<2>(evalT);

            tuple<float,float,float> evalD = eval_model(dev_set);
            summary.devF = std::get<2>(evalD);

            LLOGGER_SET(LearnLogger::EPOCH,e);
            LLOGGER_SET_TRAIN(std::get<0>(evalT),std::get<1>(evalT),std::get<2>(evalT));
            LLOGGER_SET_DEV(std::get<0>(evalD),std::get<1>(evalD),std::get<2>(evalD));
            LLOGGER_SET(LearnLogger::LOSS,summary.loss);
            LLOGGER_WRITE();

            summary.display();

//            model = tmp_params;
        }
        // TODO : averaging
        avg_model /= T;
        model -= avg_model;
    }
    bool global_learn_one(AbstractParseTree const *root, float C){
        InputDag input_sequence;
        tree_yield(root,input_sequence);
        ParseDerivation ref_deriv(root, grammar, input_sequence, dense_encoder.has_morpho());
        size_t N = input_sequence.size();

        StateSignature sig;

        //PARSING STEP
        size_t eta = dense_encoder.has_morpho() ? (2 * N) : (3 * N);
        TSSBeam beam(beam_size,grammar);
        vector<bool> actions(grammar.num_actions());

        for(int i = 1; i < eta;++i){
            for(int k = 0; k < beam.top_size();++k){
                ParseState *stack_top = beam[k];
                stack_top->get_signature(sig,&input_sequence,N);
                dense_encoder.encode(idx_vec, sig, true);
                grammar.select_actions(actions,stack_top->get_incoming_action(),sig);

                nn->fprop(idx_vec, 0, actions);
                nn->get_concatenated_layers(perceptron_input);
                model.dot(perceptron_input, perceptron_output);

                for (int j = 0; j < y_scores.size(); j++){
                    y_scores[j] = perceptron_output[j];
                }
                grammar.select_actions(y_scores, stack_top->get_incoming_action(), sig);
                beam.push_candidates(stack_top,y_scores);
            }
            beam.next_step(grammar,input_sequence,N);
        }

        //UPDATE STEP
        ParseState *last = early_update_subsequence(ref_deriv, beam, input_sequence, N);//early only

        if (NULL != last){//if has update
            ParseDerivation pred_deriv(last);

            Vec ref_xvec(input_size);
            Vec pred_xvec(input_size);
            vector<int> ref_idx;
            vector<int> pred_idx;
            vector<bool> ref_actions(grammar.num_actions());
            vector<bool> pred_actions(grammar.num_actions());

            for(int i = 0; i < ref_deriv.size()-1 && i < pred_deriv.size()-1;++i){
                StateSignature ref_sig;
                StateSignature pred_sig;

                //update
                int gold = grammar.get_action_index(ref_deriv[i+1]->get_incoming_action());
                ref_deriv[i]->get_signature(ref_sig,&input_sequence,N);
                dense_encoder.encode(ref_idx, ref_sig, true);
                grammar.select_actions(ref_actions,ref_deriv[i]->get_incoming_action(),ref_sig);

                nn->fprop(ref_idx, 0, ref_actions);
                nn->get_concatenated_layers(ref_xvec);


                int pred = grammar.get_action_index(pred_deriv[i+1]->get_incoming_action());
                pred_deriv[i]->get_signature(pred_sig,&input_sequence,N);
                dense_encoder.encode(pred_idx, pred_sig, true);
                grammar.select_actions(pred_actions,pred_deriv[i]->get_incoming_action(),pred_sig);


                nn->fprop(pred_idx, 0, pred_actions);
                nn->get_concatenated_layers(pred_xvec);

                model.add_row(ref_xvec, gold);
                model.sub_row(pred_xvec, pred);


                //model.add_bias(1, gold);
                //model.add_bias(-1, pred);


                //averaging
                ref_xvec  *= C;
                pred_xvec *= C;
                avg_model.add_row(ref_xvec, gold);
                avg_model.sub_row(pred_xvec, pred);

                //avg_model.add_bias(C, gold);
                //avg_model.add_bias(-C, pred);

            }
            return true;
        }
        return false;
    }
    ParseState* early_update_subsequence(ParseDerivation &ref_deriv,
                                         TSSBeam &beam,
                                         InputDag const &input_sequence,
                                         size_t N){// early

            for(int i = 1; (i < ref_deriv.size()-1) ;++i){
                StateSignature ref_sig;
                if (beam.size_at(i) == 0){
                    return beam.best_at(i-1);
                }
                ref_deriv[i]->get_signature(ref_sig,&input_sequence,N);
                if(!beam.sig_in_kth_beam(ref_sig,i)){
                    return beam.best_at(i);
                }
            }

        StateSignature ref_sig;
        StateSignature pred_sig;

        beam.best_success_state()->get_signature(pred_sig,&input_sequence,N);
        ref_deriv[ref_deriv.size()-1]->get_signature(ref_sig, &input_sequence,N);
        if(pred_sig == ref_sig){
            return NULL;
        }else{return beam.best_success_state();}
    }

    void save_global_parameters(string const &model_path){
        model.save(model_path, 0);
        ofstream outbeam(model_path + "/beam");
        outbeam << beam_size << endl;
        outbeam.close();
    }
};





#endif // NNPARSER_H
