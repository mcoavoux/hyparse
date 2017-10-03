#include "nnparser.h"

#define DBG(x) std::cerr << x << std::endl;
//#define DEBUG_ORACLE

unsigned int __argmax(vector<float> &scores,size_t Vs){
    float mmax = scores[0];
    unsigned int amax = 0;

    for(int j = 1; j <Vs ;++j){
        if(scores[j] > mmax){
            mmax = scores[j];
            amax = j;
        }
    }
    return amax;
}


struct EpochSummaryNN{
    int epoch_id;

    float trainF;
    float trainAcc;
    float trainObj;
    float devF;
    float devAcc;
    float devObj;
    float learning_rate;

    float last_devAcc;
    float last_devObj;
    int n_updates;
    void init(){
        trainF = 0.0;
        trainAcc = 0.0;
        trainObj = 0.0;
        devF = 0.0;
        devAcc = 0.0;
        devObj = 0.0;
        n_updates = 0;
    }

    void display(bool expl, const DynamicOracle &oracle){
        cerr << "Epoch " << epoch_id << " : "
             << "train={"<< " F1="   << trainF
                         << " locAcc="  << trainAcc
                         << " locObj="  << trainObj
             << " } dev={"  << " F1="   << devF
                            << " locAcc="  << devAcc
                            << " locObj="  << devObj
             << " } learning rate=" << learning_rate
             << " n_updates = " << n_updates;
        if (expl){
            float x1 = oracle.n_predictions;
            float x2 = oracle.n_followed_predictions;
            cerr << " Oracle preditions : " << oracle.n_predictions << " (" << oracle.n_non_determinism << " ambiguities) Exploration (%): " << (oracle.n_predictions > 0 ? (x2 / x1) : 0);
        }
        cerr << endl;
    }
};


NnSrParser::NnSrParser(bool neural_net){
    tmp_filename = "tmp";
    if (neural_net) nn = new NeuralNetwork();
    else            nn = new LogisticRegression();
}
NnSrParser::NnSrParser(DenseEncoder const &dense_encoder, bool neural_net) : NnSrParser(neural_net){
    this->dense_encoder = dense_encoder;

}

//NnSrParser::NnSrParser(const char* modelfile, bool neural_net) : NnSrParser(string(modelfile), neural_net){}

NnSrParser::NnSrParser(const string &modelfile, bool neural_net) : NnSrParser(neural_net){
    reload(modelfile);
}


void NnSrParser::train_local_model(Treebank &train_set,
                                   Treebank const &dev_set,
                                   size_t beam_size,
                                   size_t epochs,
                                   bool explore, int K, double exploration_p){

    train_set.update_encoder();

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

    grammar = SrGrammar(train_set, dev_set_binary, dense_encoder.has_morpho());
    cerr << "Grammar : number of actions = " << grammar.num_actions() << endl;
    beam = TSSBeam(1, grammar);

    //reset registers
    idx_vec.resize(dense_encoder.ntemplates());
    y_scores.resize(grammar.num_actions());


    vector<vector<int> > trainX;
    vector<int> trainY;
    //  vector<vector<int> > devX;
    //  vector<int> devY;
    extract_local_dataset(train_set, trainX, trainY);  // DEPRECATED -> gradient checking


    //setup stats
    EpochSummaryNN summary;
    summary.last_devAcc = 0;
    summary.last_devObj = 1e200;

    //vector<int> idx(trainX.size());
    vector<int> idx(train_set.size());
    for (int i = 0; i < idx.size(); i++){
        idx[i] = i;
        //      for (int j = 0; j < trainX[i].size(); j++)
        //         cerr << trainX[i][j] << " ";
        //      cerr << "  target : " << trainY[i] <<endl;
    }

    DynamicOracle oracle(grammar);

    bool accuracy_decrease = false;

    for(unsigned int e = 1; e < epochs+1;++e){
        oracle.n_non_determinism = 0;
        oracle.n_predictions = 0;
        oracle.n_followed_predictions = 0;

        int n_updates = 0;
        summary.init();
        summary.epoch_id = e;

        /*
        if (explore && e > K){
        //if (explore && e > K && e % 2 == 0){
            //float pp = (e*1.0/epochs)* exploration_p;
            //cerr << pp << endl;
            for(int i = 0; i < train_set.size();++i){
                n_updates += local_learn_one_exploration(train_set[i], oracle, exploration_p);
//                double s = (double)rand() / (double)RAND_MAX;
//                if (s < exploration_p){
//                    n_updates += local_learn_one_exploration(train_set[i], oracle, 1.0);
//                }else{
//                    n_updates += local_learn_one(train_set[i]);
//                }
                //n_updates += local_learn_one_exploration(train_set[i], oracle, pp);
            }
        }else{
            for(int i = 0; i < train_set.size();++i){
                n_updates += local_learn_one(train_set[i]);
            }
        }*/
        random_shuffle(idx.begin(), idx.end());
        for (int i = 0; i < train_set.size(); i++){
            if (explore && ((i % K) == 0)){
                n_updates += local_learn_one_exploration(train_set[idx[i]], oracle, exploration_p);
            }else{
                n_updates += local_learn_one(train_set[idx[i]]);
            }
        }

        summary.n_updates = n_updates;
//        random_shuffle(idx.begin(), idx.end());
//        for (int i = 0; i < trainX.size(); i++){
//            parsing_model.train_one(trainX[idx[i]], trainY[idx[i]]);
//        }

        nn->rescale_weights(true);        // if dropout : rescale weights matrices at test time  (~ averaging NNs)
        summary.learning_rate = nn->get_learning_rate();
        tuple<float,float,float> evalT = eval_model(train_sample);
        summary.trainF = std::get<2>(evalT);

        tuple<float,float,float> evalD = eval_model(dev_set);
        summary.devF = std::get<2>(evalD);

        pair<float, float> resTrain = local_eval_model(train_sample_binary);
        summary.trainAcc = resTrain.first;
        summary.trainObj = resTrain.second;

        pair<float, float> resDev = local_eval_model(dev_set_binary);
        summary.devAcc = resDev.first;
        summary.devObj = resDev.second;

        LLOGGER_SET(LearnLogger::EPOCH,e);
        LLOGGER_SET_TRAIN(std::get<0>(evalT),std::get<1>(evalT),std::get<2>(evalT));
        LLOGGER_SET_DEV(std::get<0>(evalD),std::get<1>(evalD),std::get<2>(evalD));
        LLOGGER_SET(LearnLogger::LOSS,summary.trainObj);
        LLOGGER_WRITE();

        nn->rescale_weights(false);   // get back to training weights

        if (summary.devAcc < summary.last_devAcc){
            accuracy_decrease = true;
        }

        if (learning_rate_policy == DOWN && accuracy_decrease){
            nn->scale_learning_rate(0.5);
        }
        if (learning_rate_policy == ADJUST){
            if (summary.devAcc < summary.last_devAcc){
                nn->scale_learning_rate(0.5);
            }else{
                nn->scale_learning_rate(1.1);
            }
        }
        summary.last_devAcc = summary.devAcc;
        summary.last_devObj = summary.devObj;
        //summary.display(explore, oracle.n_predictions, oracle.n_non_determinism);
        summary.display(explore, oracle);


        Classifier * tmp_model = nn->deepcopy();
        tmp_model->average();
        save(tmp_filename + "/iteration"+std::to_string(e), tmp_model);
        delete tmp_model;
    }
    nn->rescale_weights(true);      // end of traning : average by rescaling weights


    nn->average();

    // DEBUG :
//    vector<bool> actions(grammar.num_actions());
//    for (int i = 0; i < actions.size(); i++){
//        actions[i] = rand() % 2 == 0;
//    }
//    actions[trainY[idx[0]]] = true;
    //nn->gradient_checking(trainX[idx[0]], trainY[idx[0]], actions);
}


//void NnSrParser::local_learn_one(AbstractParseTree const *root){

////    ParseDerivation deriv(root,grammar);
////    vector<InputToken> input_sequence;
////    tree_yield(root,input_sequence);
////    size_t N = input_sequence.size();

//    InputDag input_sequence;
//    tree_yield(root,input_sequence);
//    ParseDerivation deriv(root,grammar,input_sequence,dense_encoder.has_morpho());
//    size_t N = input_sequence.size();
//    StateSignature sig;

//    for(int i = 0; i < deriv.size()-1;++i){
//        deriv[i]->get_signature(sig,&input_sequence,N);
//        dense_encoder.encode(idx_vec, sig, true);
//        parsing_model.train_one(idx_vec, deriv[i+1]->get_incoming_action().get_action_code());
//    }
//}

int NnSrParser::local_learn_one(AbstractParseTree const *root){
    InputDag input_sequence;
    tree_yield(root,input_sequence);
    ParseDerivation deriv(root,grammar,input_sequence,dense_encoder.has_morpho());
    size_t N = input_sequence.size();
    StateSignature sig;

    int nupdates = 0;
    vector<float> y_scores(grammar.num_actions(),0.0);
    vector<bool> actions(grammar.num_actions(),false);

    for(int i = 0; i < deriv.size()-1;++i){
        deriv[i]->get_signature(sig,&input_sequence,N);
        dense_encoder.encode(idx_vec, sig, true);

        int target = deriv[i+1]->get_incoming_action().get_action_code();

        grammar.select_actions(actions,deriv[i]->get_incoming_action(),sig);
        //if (! actions[target]) cerr << "Erreur ! le gold n'est pas possible" << endl;
        nn->fprop(idx_vec,target, actions);
        nn->scores(y_scores);

        unsigned int argmax = __argmax(y_scores,y_scores.size());

        if (objective != PERCEPTRON || argmax != target){
            nupdates ++;

            nn->set_pred(argmax);
            nn->bprop_std(idx_vec, target);
            nn->update();
        }
    }
    return nupdates;
}

int NnSrParser::local_learn_one_exploration(AbstractParseTree const *root, DynamicOracle &oracle, double p){

#ifdef DEBUG_ORACLE
    cerr << "GOLD ROOT : " <<  *root << endl;
#endif

    InputDag input_sequence;
    tree_yield(root,input_sequence);
    ParseDerivation deriv(root,grammar,input_sequence,dense_encoder.has_morpho());
    size_t N = input_sequence.size();
    StateSignature sig;

    ConstituentSet gold_set(deriv, grammar);

    int nupdates = 0;
    vector<float> y_scores(grammar.num_actions(),0.0);
    vector<bool> actions(grammar.num_actions(),false);
    vector<bool> y_oracle(grammar.num_actions(),false);

    bool gold = true;

    TSSBeam beam(1, grammar);
    for(int i = 0; i < deriv.size()-1;++i){
        ParseState *s0 = beam[0];
        s0->get_signature(sig,&input_sequence,N);
        //s0->encode(xvec,spencoder,sig);

        //deriv[i]->get_signature(sig,&input_sequence,N);
        dense_encoder.encode(idx_vec, sig, true);

        grammar.select_actions(actions,s0->get_incoming_action(),sig);
        //if (! actions[target]) cerr << "Erreur ! le gold n'est pas possible" << endl;
        nn->fprop(idx_vec, 0 , actions);
        nn->scores(y_scores);

        int target;
        unsigned int argmax = __argmax(y_scores,y_scores.size());
        if (gold){
            target = deriv[i+1]->get_incoming_action().get_action_code();
        }else{
            if (! oracle(s0, gold_set, y_oracle, actions)){cerr << "Problem in dynamic oracle" << endl;}
            if (y_oracle[argmax]){
                target = argmax;
            }else{
                for (int j = 0; j < y_oracle.size(); j++){
                    if (y_oracle[j]){
                        target = j;
                        break;
                    }
                }
            }
        }

        if (objective != PERCEPTRON || argmax != target){
            nupdates ++;
            nn->set_pred(argmax);
            nn->bprop_std(idx_vec, target);
            nn->update();
        }

        if (target != argmax){
            double s = (double)rand() / (double)RAND_MAX;
            //if (s < p && gold){
            if (s < p){
                gold = false;       // Let's explore !
                oracle.n_followed_predictions ++;
            }else{
                y_scores[target] = y_scores[argmax] + 1;       // Let's stay on the righteous path !
            }
        }
        beam.push_candidates(s0, y_scores);
        beam.next_step(grammar,input_sequence,N);
    }
    return nupdates;
}



// DEPRECATED : do not train with this
void NnSrParser::extract_local_dataset(Treebank & set, vector<vector<int> > &X, vector<int> & Y){
    cerr << "Extraction of local multiclass dataset ... " << endl;

    X.clear();
    Y.clear();
    for (int i = 0; i < set.size(); i++){


        InputDag input_sequence;
        tree_yield(set[i],input_sequence);
        ParseDerivation deriv(set[i],grammar,input_sequence,dense_encoder.has_morpho());
        size_t N = input_sequence.size();
        StateSignature sig;

        for(int j = 0; j < deriv.size()-1; ++j){
            deriv[j]->get_signature(sig,&input_sequence,N);
            vector<int> x;
            dense_encoder.encode(x, sig, true);

            // First pass on dataset : allocates all embeddings -> avoid concurrency problems with on the fly allocations
            for (int k = 0; k < x.size(); k++){
                nn->allocate_embedding(x[k]);
            }

            int y = deriv[j+1]->get_incoming_action().get_action_code();

            X.push_back(x);
            Y.push_back(y);
        }
    }
    cerr << "Number of training examples : " << X.size() << endl;
}

//AbstractParseTree* NnSrParser::predict_one(vector<InputToken> const &input_sequence){
AbstractParseTree* NnSrParser::predict_one(InputDag &input_sequence){
    StateSignature sig;
    size_t N = input_sequence.size();
    size_t eta = dense_encoder.has_morpho() ? (2 * N)-1 : (3 * N)-1;
    beam.reset();

    vector<bool> actions(grammar.num_actions());

    for(int i = 0; i < eta;++i){
      for(int k = 0; k < beam.top_size();++k){

          ParseState *stack_top = beam[k];
          stack_top->get_signature(sig,&input_sequence,N);
          dense_encoder.encode(idx_vec, sig, true);             // @@@m Lines modified from sparse version
          grammar.select_actions(actions,stack_top->get_incoming_action(),sig);
          nn->scores(idx_vec, y_scores, actions);            //  TODO : compute cost here
          grammar.select_actions(y_scores,stack_top->get_incoming_action(),sig);
          beam.push_candidates(stack_top,y_scores);
      }
      beam.next_step(grammar,input_sequence,N);
    }
    if (beam.has_best_parse()){return beam.best_parse(input_sequence);}
    else{return NULL;}
}

pair<pair<float, float>, float> NnSrParser::local_eval_one(const AbstractParseTree *ref){

    InputDag input_sequence;
    tree_yield(ref,input_sequence);
    ParseDerivation deriv(ref,grammar,input_sequence,dense_encoder.has_morpho());
    size_t N = input_sequence.size();
    StateSignature sig;

    int gold;

    float objective = 0.0;
    float correct = 0.0;
    float total = 0.0;
    vector<bool> actions(grammar.num_actions());

    for(int i = 0; i < deriv.size()-1;++i){

        deriv[i]->get_signature(sig,&input_sequence,N);

        dense_encoder.encode(idx_vec, sig, true);

        gold = deriv[i+1]->get_incoming_action().get_action_code();
        grammar.select_actions(actions, deriv[i]->get_incoming_action(),sig);  // filter impossible actions
        //if (! actions[gold]) cerr << "Erreur ! le gold n'est pas possible" << grammar[gold] << endl;

        if (actions[gold]){
            objective += nn->scores(idx_vec, y_scores, gold, actions);
        }else{
            nn->scores(idx_vec, y_scores, gold, actions);
        }

        grammar.select_actions(y_scores, deriv[i]->get_incoming_action(),sig);  // filter impossible actions

        //find argmax action
        unsigned int argmax = __argmax(y_scores,y_scores.size());
        if(grammar[argmax] == deriv[i+1]->get_incoming_action()){
            correct++;
        }
        total++;
    }
    return make_pair(make_pair(correct, total), objective);
}

pair<float, float> NnSrParser::local_eval_model(const Treebank &eval_set){
    float ncorrect = 0;
    float ntotal = 0;
    float objective = 0;

    for(int i = 0; i < eval_set.size();++i){
          pair<pair<float,float>, float> res = local_eval_one(eval_set[i]);
          ncorrect += res.first.first;
          ntotal   += res.first.second;
          objective += res.second;
    }
    return make_pair(ncorrect/ntotal, objective/ntotal);
}

//tuple<float,float,float> NnSrParser::eval_model(const Treebank &eval_set){

//    float p = 0;
//    float r = 0;
//    float f = 0;
//    size_t N = eval_set.size();

//    for(int i = 0; i < N;++i){

//        AbstractParseTree *root;

//        InputDag input_sequence;
//        tree_yield(eval_set[i],input_sequence);
//        root = predict_one(input_sequence);

//        if (beam.has_best_parse()){
//            unpack_unaries(root);
//            unbinarize(root);
//            tuple<float,float,float> E = compare(root,eval_set[i]);
//            p += get<0>(E);
//            r += get<1>(E);
//            f += get<2>(E);
////            cout << "Ref:" << *eval_set[i] << endl;
////            cout << "Res:" << *root << endl;
////            cout << endl;
//            destroy(root);
//        }else{
////            cout << "(())"<<endl;
////            cout << *eval_set[i] << endl;
//        }
//    }
//    return make_tuple(p/N,r/N,f/N);
//}


tuple<float,float,float> NnSrParser::eval_model(const Treebank &eval_set){

    float good = 0;
    float pred = 0;
    float gold = 0;
    size_t N = eval_set.size();

    for(int i = 0; i < N;++i){

        AbstractParseTree *root;
        InputDag input_sequence;
        tree_yield(eval_set[i],input_sequence);
        root = predict_one(input_sequence);

        if (beam.has_best_parse()){
            unpack_unaries(root);
            unbinarize(root);
            tuple<float,float,float> E = compare_evalb(root,eval_set[i]);
            good += get<0>(E);
            pred += get<1>(E);
            //f += get<2>(E);
            gold += get<2>(E);
//            cout << "Ref:" << *eval_set[i] << endl;
//            cout << "Res:" << *root << endl;
//            cout << endl;
            destroy(root);
        }else{
//            cout << "(())"<<endl;
//            cout << *eval_set[i] << endl;
        }
    }
    float p = good / pred;
    float r  = good / gold;
    float f = 2 * p * r / (p + r);
    return make_tuple(p,r,f);
}

AbstractParseTree* NnSrParser::predict_kth_parse(int k, InputDag const &input_sequence){
    return beam.kth_best_parse(input_sequence,k);
}


size_t NnSrParser::get_num_parses()const{return beam.num_parses();}


void NnSrParser::parse_corpus(AbstractLexer *lex,istream &input_source,int K,ParserOutStream &ptbstream,ParserOutStream &conllstream,ParserOutStream &nativestream){
    lex->skip_header(input_source);

    InputDag input_sequence;

    while(lex->next_sentence(input_source,input_sequence)){

        //parsing
        PLOGGER_START_TIMER();
        AbstractParseTree *p = predict_one(input_sequence);
        PLOGGER_END_TIMER();
        PLOGGER_SET(ParseLogger::LEN,input_sequence.size());
        PLOGGER_WRITE();

        //postprocessing
        if(K > 1){ //K-best parsing

            unpack_unaries(p);
            unbinarize(p);
            ptbstream.flush_parse(p,input_sequence);
            conllstream.flush_parse(p,input_sequence);
            destroy(p);

            for(int i = 1; i < K && i < beam.num_parses();++i){
                p = beam.kth_best_parse(input_sequence,i);
                unpack_unaries(p);
                unbinarize(p);
                ptbstream.flush_parse(p,input_sequence);
                conllstream.flush_parse(p,input_sequence);
                nativestream.flush_parse(p,input_sequence);
                destroy(p);
            }
            cout << endl;

        }else{ //1-best parsing
            if (beam.has_best_parse()){
                unpack_unaries(p);
                unbinarize(p);
                ptbstream.flush_parse(p,input_sequence);
                conllstream.flush_parse(p,input_sequence);
                nativestream.flush_parse(p,input_sequence);
                destroy(p);
            }
        }
    }
}

void NnSrParser::save(const string &dirname, Classifier *classifier){
    mkdir(dirname.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    grammar.save(dirname+"/grammar");

    ofstream beamout(dirname+"/beam");
    beamout << beam.maxsize() << endl;
    beamout.close();

    if (classifier == NULL){
        nn->save(dirname);
    }else{
        classifier->save(dirname);
    }
}

void NnSrParser::reload(string const &modelfile){
    //1.Reads sparse encoder
    TemplateTypeDefinition ttd;
    ttd.load(modelfile+"/ttd");

    this->dense_encoder = DenseEncoder(modelfile+"/templates",ttd);
    //2. Grammar
    this->grammar = SrGrammar(modelfile+"/grammar");
    //3. beam
    string bfr;
    ifstream beamin(modelfile+"/beam");
    getline(beamin,bfr);
    beamin.close();
    beam = TSSBeam(stoi(bfr),grammar);
    //4.Init registers
    idx_vec = std::vector<int>(dense_encoder.ntemplates());
    y_scores.resize(grammar.num_actions());
    //5. Weights
    nn->load(modelfile);

}

void NnSrParser::initialize_params(NeuralNetParams &params){
    nn->set_hyperparameters(params);
    objective = params.loss_function;
}

void NnSrParser::summary(ostream &os){
    nn->summary(os);
}

void NnSrParser::load_lu(const std::string & lu_filename){
    nn->load_pretrained_embeddings(lu_filename);
}

void NnSrParser::set_lr_policy(int policy){
    learning_rate_policy = policy;
}


void NnSrParser::set_test_mode(bool t){
    nn->set_test_mode(t);
}
void NnSrParser::set_tmp_filename(string tmp_file){
    tmp_filename = tmp_file;
}



