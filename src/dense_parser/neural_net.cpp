#include "neural_net.h"


/********
 *
 *  Classifier
 *
 */

void Classifier::set_test_mode(bool t){
    test_mode = t;
}
int Classifier::get_concatenated_layers_size(){
    throw "Not implemented exception : int get_concatenated_layers_size()";
    return 0;
}
void Classifier::get_concatenated_layers(Vec &perceptron_input){
    throw "Not implemented exception : void get_concatenated_layers(Vec &perceptron_input)";
    return;
}
void Classifier::average(){
    throw "Not implemented exception : void average()s";
    return;
}



/*********
 *
 * LogisticRegression
 *
 */


void LogisticRegression::set_hyperparameters(const NeuralNetParams &params){
    learning_rate = params.learning_rate;
    n_units = params.n_output;
    n_features = params.n_input_features;
    output = Vec::Zero(n_units);
    bias = Vec::Zero(n_units);
    gbias = Vec::Zero(n_units);
    avg_bias = Vec::Zero(n_units);
    features = vector<int>(n_features);
    asgd = params.asgd;
    n_updates = 1;
}
void LogisticRegression::set_learning_rate(double lr){                          learning_rate = lr; }
double LogisticRegression::get_learning_rate(){                                 return learning_rate; }
void LogisticRegression::rescale_weights(bool down){                            return; }
void LogisticRegression::scale_learning_rate(double d){                         learning_rate *= d; }
void LogisticRegression::summary(ostream &os){
    os << "Summary" << endl;
    os << "-------" << endl;
    os << "- Model : Multinomial Logistic Regression" << endl;
    os << "- Number of feature tempaltes : " << n_features << endl;
    os << "- Number of output units : " << n_units << endl;
    os << "- Learning rate : " << learning_rate << endl;
    os << "- Averaged SGD : " << asgd << endl;
}

void LogisticRegression::set_pred(int newp){
    prediction = newp;
}

void LogisticRegression::find_features(const vector<int> & input){
    for (int i = 0; i < input.size(); i++){
        pair<int,int> feature = make_pair(i, input[i]);
        const auto it = map.find(feature);
        if (it == map.end()){
            int idx;
            idx = weights.size();
            map[feature] = idx;
            weights.push_back(Vec::Zero(n_units));
            avg_weights.push_back(Vec::Zero(n_units));
            features[i] = idx;
        }else{
            features[i] = it->second;
        }
    }
}

double LogisticRegression::fprop(const vector<int> & input, int target, const vector<bool> &actions){
    find_features(input);
    output = bias;
    for (int i = 0; i < features.size(); i++){
        output += weights[features[i]];
    }
    output = (output.array() - output.maxCoeff()).exp();
    for (int i = 0; i < actions.size(); i++){
        if (!actions[i])
            output[i] = 0;
    }
    output /= output.sum();
    return - log(output[target]);
}
void LogisticRegression::scores(vector<float> &score){
    for (int i = 0; i < output.size(); i++){
        score[i] = log(output[i]);
    }
}
void LogisticRegression::scores(const vector<int> &input, vector<float> &output, const vector<bool> &actions){
    scores(input, output, 0, actions);
}
double LogisticRegression::scores(const vector<int> &input, vector<float> &output, int target, const vector<bool> &actions){
    double loss = fprop(input, target, actions);
    scores(output);
    return loss;
}
void LogisticRegression::bprop_std(const vector<int> & input, int target){
    gbias = output;
    gbias[target] -= 1;
}
void LogisticRegression::update(){
    if (asgd){
        for (int i = 0; i < features.size(); i++){
            weights[features[i]] -= learning_rate * gbias;
        }
        bias -= learning_rate * gbias;
    }else{
        for (int i = 0; i < features.size(); i++){
            weights[features[i]] -= learning_rate * gbias;
            avg_weights[features[i]] -= n_updates * learning_rate * gbias;
        }
        bias -= learning_rate * gbias;
        avg_bias -= n_updates * learning_rate * gbias;
    }
    n_updates ++;
}

void LogisticRegression::average(){
    for (int i = 0; i < weights.size(); i++){
        weights[i] -= avg_weights[i] / n_updates;
    }
    bias -= avg_bias / n_updates;
}

void LogisticRegression::save(const string dirname){
    mkdir(dirname.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    std::ofstream outfile(dirname + "/mlr_model");
    outfile << n_units << endl;
    outfile << n_features << endl;
    outfile << weights.size() << endl;
    outfile.close();
    outfile.open(dirname + "/mlr_parameters");
    for (int i = 0; i < weights.size(); i++){
        outfile << weights[i].transpose() << endl;
    }
    outfile.close();
    outfile.open(dirname + "/mlr_bias");
    outfile << bias.transpose() << endl;
    outfile.close();
    outfile.open(dirname + "/mlr_mapping");
    for (auto it = map.begin(); it != map.end(); it++){
        outfile << it->first.first << " " << it->first.second << " " << it->second << endl;
    }
    outfile.close();
}

void LogisticRegression::load(const string dirname){
    string line;
    vector<string> sline;
    int weights_size;

    std::ifstream infile(dirname + "/mlr_model");
    infile >> n_units;
    infile >> n_features;
    infile >> weights_size;
    infile.close();

    features = vector<int>(n_features);
    output = bias = gbias = Vec::Zero(n_features);
    weights.clear();

    infile.open(dirname + "/mlr_parameters");
    while(getline(infile, line)){
        boost::trim(line);
        boost::split(sline, line, boost::is_any_of(" "), boost::token_compress_on);
        assert(sline.size() == n_units);
        Vec v(n_units);
        for (int i = 0; i < sline.size(); i++){
            v[i] = stod(sline[i]);
        }
        weights.push_back(v);
    }
    assert(weights.size() == weights_size);
    infile.close();

    infile.open(dirname + "/mlr_bias");
    getline(infile, line);
    boost::trim(line);
    boost::split(sline, line, boost::is_any_of(" "), boost::token_compress_on);
    assert(sline.size() == n_units);
    bias = Vec(n_units);
    for (int i = 0; i < sline.size(); i++){
        bias[i] = stod(sline[i]);
    }
    infile.close();

    infile.open(dirname + "/mlr_mapping");
    while(getline(infile, line)){
        boost::trim(line);
        boost::split(sline, line, boost::is_any_of(" "), boost::token_compress_on);
        assert(sline.size() == 3);
        map[make_pair(atoi(sline[0].c_str()), atoi(sline[1].c_str()))] = atoi(sline[2].c_str());
    }
    assert(map.size() == weights.size());
    infile.close();
}
void LogisticRegression::load_pretrained_embeddings(const string &lu_filename){
    cerr << "ERORR, illegal state, aborting" << endl;
    exit(1);
}
void LogisticRegression::allocate_embedding(int i){}




























/*****************
 *
 *  Neural nets
 *
 */

const string NeuralNetwork::NC  = "number of output units";
const string NeuralNetwork::NH  = "number of hidden layers";
const string NeuralNetwork::HL  = "number of units per hidden layer";
const string NeuralNetwork::NI  = "number input vectors";
const string NeuralNetwork::NE  = "number of units per vector";
const string NeuralNetwork::LOSS = "loss function";
const string NeuralNetwork::ACTIVATION = "activation function";
const string NeuralNetwork::CONV_NET = "convolution";

NeuralNetwork::NeuralNetwork(int n_input_features, vector<int> n_input, vector<int> n_hidden, int n_output) : n_input_features(n_input_features), n_output(n_output), n_input(n_input), n_hidden(n_hidden) {
    learning_rate = 0.01;
    decrease_constant = 0;
    ada_epsilon = 1e-6;
    reg_lambda = 1e-6;
    clipping_threshold = 5;
    ada_grad = false;
    dropout = false;
    regularisation = false;
    gradient_clipping = false;
    loss_function = PERCEPTRON;
    hidden_activation = TANH;
    input_layer = NULL;
    n_updates = 0;
    test_mode = false;
    convolution = false;
    asgd = false;
}

NeuralNetwork::NeuralNetwork() : NeuralNetwork(0, vector<int>(0), vector<int>(0), 0){}

NeuralNetwork::~NeuralNetwork(){
    delete input_layer;
    for (int i = 0; i < layers.size(); i++){
        delete layers[i];
        layers[i] = NULL;
    }
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork &other){
    test_mode           = other.test_mode;
    n_input_features    = other.n_input_features;
    n_output            = other.n_output;
    n_input             = other.n_input;
    n_hidden            = other.n_hidden;
    learning_rate       = other.learning_rate;
    decrease_constant   = other.decrease_constant;
    reg_lambda          = other.reg_lambda;
    ada_epsilon         = other.ada_epsilon;
    clipping_threshold  = other.clipping_threshold;
    ada_grad            = other.ada_grad;
    dropout             = other.dropout;
    regularisation      = other.regularisation;
    gradient_clipping   = other.gradient_clipping;
    asgd                = other.asgd;
    n_updates           = other.n_updates;
    input_layer         = other.input_layer->deepcopy();  // LookupToHiddenLayer *input_layer;
    layers              = vector<Layer*>(other.layers.size());
    for (int i = 0; i < layers.size(); i++) layers[i] =  other.layers[i]->deepcopy();
    loss_function       = other.loss_function;
    hidden_activation   = other.hidden_activation;
    convolution         = other.convolution;
}

NeuralNetwork & NeuralNetwork::operator=(const NeuralNetwork &other){
    test_mode           = other.test_mode;
    n_input_features    = other.n_input_features;
    n_output            = other.n_output;
    n_input             = other.n_input;
    n_hidden            = other.n_hidden;
    learning_rate       = other.learning_rate;
    decrease_constant   = other.decrease_constant;
    reg_lambda          = other.reg_lambda;
    ada_epsilon         = other.ada_epsilon;
    clipping_threshold  = other.clipping_threshold;
    ada_grad            = other.ada_grad;
    dropout             = other.dropout;
    regularisation      = other.regularisation;
    gradient_clipping   = other.gradient_clipping;
    asgd                = other.asgd;
    n_updates           = other.n_updates;
    input_layer         = other.input_layer->deepcopy();  // LookupToHiddenLayer *input_layer;
    layers              = vector<Layer*>(other.layers.size());
    for (int i = 0; i < layers.size(); i++) layers[i] =  other.layers[i]->deepcopy();
    loss_function       = other.loss_function;
    hidden_activation   = other.hidden_activation;
    convolution         = other.convolution;
    return (*this);
}





void NeuralNetwork::summary(ostream &os){
    string yesno[2] = {"No", "Yes"};
    os << endl;
    os << "Summary" << endl;
    os << "=======" << endl;
    os << endl;
    os << "- Input features       : " << n_input_features << endl;
    os << "- Embedding units      : [ "; for (int i = 0; i < n_input.size(); i++) os <<  n_input[i] << " "; os << "]"  << endl;
    os << "- Hidden units         : [ "; for (int i = 0; i < n_hidden.size(); i++) os << n_hidden[i] << " "; os << "]"  << endl;
    os << "- Output units         : " << n_output << endl;
    os << "- Learning rate        : " << learning_rate << endl;
    os << "- Decrease constant    : " << decrease_constant << endl;
    os << "- Regularisation       : " << yesno[regularisation] << " (alpha=" << reg_lambda << ")" << endl;
    os << "- AdaGrad              : " << yesno[ada_grad] << " (epsilon=" << ada_epsilon << ")" << endl;
    os << "- Dropout              : " << yesno[dropout] << endl;
    os << "- Gradient clipping    : " << yesno[gradient_clipping] << " (threshold=" << clipping_threshold << ")" << endl;
    os << "- Averaged SGD         : " << yesno[asgd] << endl;
    os << "- Loss function        : " << LOSS_FUNCTIONS[loss_function] << endl;
    os << "- Hidden activation    : " << ACTIVATIONS[hidden_activation] << endl;

    input_layer->display_summary_convolution(os);
}
void NeuralNetwork::initialize(){
    if (n_hidden.empty()){
        input_layer = get_new_lookup_layer(n_input, n_input_features, n_output, true);
    }else{
        input_layer = get_new_lookup_layer(n_input, n_input_features, n_hidden.at(0), false);
        for (int i = 1; i < n_hidden.size(); i++){
            layers.push_back(get_new_hidden_layer(n_hidden.at(i-1), n_hidden.at(i)));
        }
        layers.push_back(get_new_output_layer(n_hidden.back(), n_output));

        //linking between layers :
        input_layer->set_next_layer(layers[0]);
        for (int i = 0; i < layers.size()-1; i++){
            layers[i]->set_next_layer(layers[i+1]);
        }
    }
}

LookupToHiddenLayer* NeuralNetwork::get_new_lookup_layer(vector<int> n_input, int n_input_features, int n_units, bool output){
    if (output){
        switch(loss_function){
            case CROSS_ENTROPY :    return new LookupLayerSoftmax(n_input, n_input_features, n_units);
            case PERCEPTRON :       return new LookupLayerPerceptron(n_input, n_input_features, n_units);
            default : cerr << "NN error, aborting..."<< endl; exit(1);
        }
    }else{
        switch(hidden_activation){
            case TANH : return new LookupLayerTanh(n_input, n_input_features, n_units);
            case CUBE : return new LookupLayerCube(n_input, n_input_features, n_units);
            case RELU : return new LookupLayerRelu(n_input, n_input_features, n_units);
            default : cerr << "NN error, aborting..."<< endl; exit(1);
        }
    }
}
Layer* NeuralNetwork::get_new_output_layer(int input, int output){
    switch(loss_function){
        case CROSS_ENTROPY :  return new SoftmaxLayer(input, output);
        case PERCEPTRON :  return new PerceptronLayer(input, output);
        default : cerr << "NN error, aborting..."<< endl; exit(1);
    }
}
Layer* NeuralNetwork::get_new_hidden_layer(int input, int units){
    switch(hidden_activation){
        case TANH : return new TanhLayer(input, units);
        case CUBE : return new CubeLayer(input, units);
        case RELU : return new ReluLayer(input, units);
        default : cerr << "NN error, aborting..."<< endl; exit(1);
    }
}




void NeuralNetwork::set_hyperparameters(const NeuralNetParams &params){
    n_input_features    = params.n_input_features;
    n_input             = params.n_input;
    n_hidden            = params.n_hidden;
    n_output            = params.n_output;
    asgd                = params.asgd;
    learning_rate       = params.learning_rate;
    decrease_constant   = params.decrease_constant;
    reg_lambda          = params.reg_lambda;
    clipping_threshold  = params.clipping_threshold;
    gradient_clipping   = params.gradient_clipping;
    dropout             = params.dropout;
    ada_grad            = params.ada_grad;
    regularisation      = params.regularization;
    loss_function       = params.loss_function;
    hidden_activation   = params.hidden_activation;

    convolution         = params.convolution;
    initialize();
    input_layer->set_epsilon(params.embedding_epsilon);
    input_layer->set_dimensions(params.embeddings_dimensions);
    input_layer->set_convolution_options(params);

}

void NeuralNetwork::set_learning_rate(double lr){              learning_rate = lr;}
void NeuralNetwork::set_random_init(bool ri){                  input_layer->set_random_init(ri);}
void NeuralNetwork::scale_learning_rate(double d){             learning_rate *= d;}
double NeuralNetwork::get_learning_rate(){                     return learning_rate / (1.0 + n_updates * decrease_constant); }

void NeuralNetwork::allocate_embedding(int i){                 input_layer->check_embedding(i);}

int NeuralNetwork::predict(){
    if (input_layer->is_output_layer())
        return input_layer->predict();
    return layers.back()->predict();
}

void NeuralNetwork::set_pred(int newp){
    if (input_layer->is_output_layer()){    input_layer->set_pred(newp); }
    else                               {    layers.back()->set_pred(newp);}
}

void NeuralNetwork::rescale_weights(bool down){
    if (dropout){
        for (int i = 0; i < layers.size() -1; i++){
            layers[i]->rescale_weights(down);
        }
    }
}

void NeuralNetwork::train_one(const vector<int> &input, int target, const vector<bool> &actions){
    fprop(input,target, actions);
    if (loss_function != PERCEPTRON || predict() != target){
        bprop_std(input, target);
        update();
    }
}

void NeuralNetwork::scores(const vector<int> &input, vector<float> &output, const vector<bool> &actions){
    scores(input, output, 0, actions);
}

double NeuralNetwork::scores(const vector<int> &input, vector<float> &output, int target, const vector<bool> &actions){
    double cost = fprop(input, target, actions);
    scores(output);
    return cost;
}

void NeuralNetwork::scores(vector<float> &output){
    const Vec *output_layer = input_layer->is_output_layer() ? &(input_layer->state()) : & (layers.back()->state());
    if (loss_function == CROSS_ENTROPY){
        for (int i = 0; i < n_output; i++){
            output[i] = log((*output_layer)(i));
        }
    }else{
        for (int i = 0; i < n_output; i++){
            output[i] = (*output_layer)(i);
        }
    }
}



double NeuralNetwork::fprop(const vector<int> & input, int target, const vector<bool> &actions){
    double cost = 0.0;
    if (test_mode){
        input_layer->fast_fprop_std(input, target, actions);
    }else{
        input_layer->fprop_std(input, target, actions);
    }
    if (input_layer->is_output_layer()){
        cost = input_layer->compute_cost(target);
    }else{
        layers[0]->fprop_std(input_layer->state(), target, actions);
        for (int i = 1; i < layers.size(); i++){
            layers[i]->fprop_std(layers[i-1]->state(), target, actions);
        }
        cost = layers.back()->compute_cost(target);
    }

    if (regularisation){
        cost += input_layer->l2_cost(reg_lambda);
        for (int i = 0; i < layers.size(); i++)
            cost += layers[i]->l2_cost(reg_lambda);
    }
    return cost;
}

void NeuralNetwork::bprop_std(const vector<int> & input, int target){
    if (input_layer->is_output_layer()){
        input_layer->bprop_std(input, target);
    }else{
        int i = layers.size()-1;
        if (i > 0){
            layers[i]-> bprop_std(layers[i-1]->state(), target);
            --i;
            while (i > 0){
                layers[i]->bprop_std(layers[i-1]->state(), target);
                --i;
            }
            layers[0]->bprop_std(input_layer->state(), target);
        }else{
            layers[0]->bprop_std(input_layer->state(), target);
        }
        input_layer->bprop_std(input, target);
    }
    if (regularisation){
        input_layer->add_l2_gradient(reg_lambda);
        for (int i = 0; i < layers.size(); i++)
            layers[i]->add_l2_gradient(reg_lambda);
    }
}


void NeuralNetwork::update(){
    n_updates ++;
    double step_size = get_learning_rate();
    if (gradient_clipping){
        clip_gradient(clipping_threshold);
    }
    if (ada_grad){
        input_layer->update_adagrad(step_size, ada_epsilon);
        for (int i = 0; i < layers.size(); i++)
            layers[i]->update_adagrad(step_size, ada_epsilon);
    }else{
        if (asgd){
            input_layer->update_asgd(step_size, n_updates);
            for (int i = 0; i < layers.size(); i++)
                layers[i]->update_asgd(step_size, n_updates);
        }else{
            input_layer->update_sgd(step_size);
            for (int i = 0; i < layers.size(); i++)
                layers[i]->update_sgd(step_size);
       }
    }
}

void NeuralNetwork::clip_gradient(double threshold){
    input_layer->clip_gradient(threshold);
    for (int i = 0; i < layers.size(); i++){
        layers[i]->clip_gradient(threshold);
    }
}


/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////


void NeuralNetwork::gradient_check_layer(const vector<int> &input, int target, Layer *l, Mat & egw, Vec & egb, double epsilon, const vector<bool> &actions){

    double a,c;
    a = c = 0.0;
    for (int i = 0; i < egw.rows(); i ++){
        for (int j = 0; j < egw.cols(); j++){
            l->add_epsilon(i,j,epsilon);
            a = fprop(input, target, actions);
            l->add_epsilon(i,j,-epsilon);
            l->add_epsilon(i,j,-epsilon);
            c = fprop(input, target, actions);
            l->add_epsilon(i,j,epsilon);
            egw(i,j) = (a - c) / (2.0 * epsilon);
        }
        l->add_epsilon_b(i,epsilon);
        a = fprop(input, target, actions);
        l->add_epsilon_b(i,-epsilon);
        l->add_epsilon_b(i,-epsilon);
        c = fprop(input, target, actions);
        l->add_epsilon_b(i,epsilon);
        egb[i] = (a - c) / (2.0 * epsilon);
    }
}

void NeuralNetwork::gradient_check_lookup_layer(const vector<int> &input, int target, LookupToHiddenLayer *l, double epsilon, const vector<bool> &actions){
    int n_input_features = l->get_n_input_features();
    vector<int> unique_features;
    l->get_unique_features(unique_features);
    int n_units = l->get_n_units();

    vector<Mat> egw(input.size());
    vector<Vec> egE(unique_features.size());
    Vec egb(n_units);

    double a,c;
    a = c = 0.0;
    // Weights
    for (int f = 0; f < n_input_features; f++){
        egw[f] = Mat::Zero(n_units, l->get_embedding_size(input[f]));

        for (int i = 0; i < egw[f].rows(); i ++){
            for (int j = 0; j < egw[f].cols(); j++){
                l->add_epsilon(f,i,j,epsilon);
                a = fprop(input, target, actions);
                l->add_epsilon(f,i,j,-epsilon);
                l->add_epsilon(f,i,j,-epsilon);
                c = fprop(input, target, actions);
                l->add_epsilon(f,i,j,epsilon);
                egw[f](i,j) = (a - c) / (2.0 * epsilon);
            }
        }
        cerr << "Lookup layer feature = " << f  << " w : " << (egw[f] - l->gweights(f)).array().abs().sum() / egw[f].size() << endl;
    }
    // Bias
    for (int i = 0; i < n_units; i++){
        l->add_epsilon_b(i,epsilon);
        a = fprop(input, target, actions);
        l->add_epsilon_b(i,-epsilon);
        l->add_epsilon_b(i,-epsilon);
        c = fprop(input, target, actions);
        l->add_epsilon_b(i,epsilon);
        egb[i] = (a - c) / (2.0 * epsilon);
    }
    cerr << "Lookup layer  b : " << (egb - l->gbias()).array().abs().sum() / egb.size() << endl;


    // Embeddings
    for (int i = 0; i < unique_features.size(); i++){
        int size = l->get_embedding_size(unique_features[i]);
        egE[i] = Vec::Zero(size);
        for (int j = 0; j < size; j++){
            l->add_epsilon(unique_features[i], j, epsilon);
            a = fprop(input, target, actions);
            l->add_epsilon(unique_features[i], j, -epsilon);
            l->add_epsilon(unique_features[i], j, -epsilon);
            c = fprop(input, target, actions);
            l->add_epsilon(unique_features[i], j, epsilon);
            egE[i](j) = (a - c) / (2.0 * epsilon);
        }
        cerr << "Lookup layer embedding = " << unique_features[i] << " : " <<(egE[i]- l->gembedding(unique_features[i])).array().abs().sum() / size << endl;
    }
}

void NeuralNetwork::gradient_checking(const vector<int> &input, int target, const vector<bool> &actions){
    double epsilon = 10e-6;

    fprop(input, target, actions);
    bprop_std(input, target);


    gradient_check_lookup_layer(input, target, input_layer, epsilon, actions);

    for (int i = 0; i < layers.size(); i++){
        Layer *l = static_cast<Layer*>(layers[i]);
        Mat egw = Mat::Zero(l->get_n_units(), l->get_n_input());
        Vec egb = Vec::Zero(l->get_n_units());
        gradient_check_layer(input, target, l, egw, egb, epsilon, actions);
        cerr << "Layer " << i << " w : " << (egw - l->gweights()).array().abs().sum() / egw.size() << endl;
        cerr << "Layer " << i << " b : " << (egb - l->gbias()).array().abs().sum() / egb.size() << endl;
    }
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////








// INPUT / OUTPUT
void NeuralNetwork::save(const string dirname){
    mkdir(dirname.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    this->dump_hyperparameters(dirname +"/parameters.conf");
    this->dump_parameters(dirname);
}
void NeuralNetwork::load(const string dirname){                       // loads a saved model from dirname
    load_hyperparameters(dirname + "/parameters.conf");
    initialize();
    load_parameters(dirname);
}
void NeuralNetwork::dump_hyperparameters(const string filename) const{
    std::ofstream outfile(filename);
    outfile << NC << " = " << n_output << endl;
    outfile << NH << " = " << n_hidden.size() << endl;
    outfile << HL << " = "; for (int i = 0; i < n_hidden.size(); i++) outfile << n_hidden[i] << " ";
    outfile << endl;
    outfile << NI << " = " << n_input_features << endl;
    outfile << NE << " = "; for (int i = 0; i < n_input_features; i++) outfile << n_input[i] << " ";
    outfile << endl;
    outfile << LOSS << " = " << loss_function << endl;
    outfile << ACTIVATION << " = " << hidden_activation << endl;
    outfile << CONV_NET << " = " << convolution << endl;
    outfile.close();
}
void NeuralNetwork::load_hyperparameters(const string filename){      // loads hyperparameters from .conf file
    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(filename, pt);

    n_output = pt.get<int>(NC);
    n_hidden = vector<int>(pt.get<int>(NH), 0);
    n_input_features = pt.get<int>(NI);
    n_input.resize(n_input_features);
    hidden_activation = pt.get<int>(ACTIVATION);
    loss_function = pt.get<int>(LOSS);


    string is = pt.get<string>(NE);
    std::stringstream ss(is);
    int tmp;
    for (int i = 0; i < n_input_features; i++){
        ss >> tmp;
        n_input[i] = tmp;
    }

    is = pt.get<string>(HL);
    ss.clear();
    ss.str(is);
    for (int i = 0; i < n_hidden.size(); i++){
        ss >> tmp;
        n_hidden[i] = tmp;
    }
    convolution = pt.get<bool>(CONV_NET);
}
void NeuralNetwork::dump_parameters(const string dirname) const{
    input_layer->dump_parameters(dirname);
    for (int i = 0; i < layers.size(); i++){
        layers[i]->dump_parameters(dirname, i);
    }
}

void NeuralNetwork::load_parameters(const string dirname){
    input_layer->set_convolution(this->convolution);
    input_layer->load_parameters(dirname);
    for (int i = 0; i < layers.size(); i++){
        layers[i]->load_parameters(dirname, i);
    }
}

void NeuralNetwork::load_pretrained_embeddings(const string &lu_filename){
    input_layer->load_and_encode(lu_filename);
}


void NeuralNetwork::run_gradient_checking(){

    NeuralNetParams params;
    params.reg_lambda = 1e-5;
    params.n_input = vector<int>(8, 50);
    params.n_input_features = params.n_input.size();
    params.n_output = 10;
    vector<int> coldims(10, 50);
    params.embeddings_dimensions = coldims;

    string tmp("a");
    for (int i = 0; i < 100; i++){
        tmp += "a";
        IntegerEncoder::get()->encode(tmp, 2);
    }

    vector<int> input(params.n_input_features);
    for (int i = 0; i < params.n_input_features; i++){
        input[i] = rand() % 8;
    }
    int target = rand() % params.n_output;


    vector<bool> actions(params.n_output, true);
    actions[rand()%10] = false;


    for (int loss = 0; loss < 2; loss++){
        for (int activ = 0; activ < 3; activ ++){
            for (int reg = 0; reg < 2; reg ++){
                params.n_hidden.clear();

                for (int H = 0; H < 3; H++){
                    if (H > 0) params.n_hidden.push_back(20);
                    params.loss_function = loss;
                    params.hidden_activation = activ;
                    params.regularization = reg;

                    NeuralNetwork nn;
                    nn.set_hyperparameters(params);

                    cerr << endl << endl;
                    nn.summary(cerr);

                    nn.gradient_checking(input, target, actions);

                }
            }
        }
    }
}










void NeuralNetwork::average(){
   if (asgd){
       input_layer->average(n_updates);
       for (int i = 0; i < layers.size(); i++){
           layers[i]->average(n_updates);
       }
   }
}

void NeuralNetwork::set_test_mode(bool t){
#ifdef MEMORY_EFFICIENT
#else
   test_mode = t;
   input_layer->precompute_hidden_layer();
#endif
}
int NeuralNetwork::get_concatenated_layers_size(){
   int n = 0;
   n += input_layer->get_n_units();
   for (int i = 0; i < layers.size(); i++){
       n += layers[i]->get_n_units();
   }
   return n;
}

void NeuralNetwork::get_concatenated_layers(Vec &perceptron_input){

  //assert(perceptron_input.size()==get_concatenated_layers_size());

  //add hoc implementation...
  switch(layers.size()){
  case 0:
    perceptron_input << input_layer->state();
    break;
  case 1:
    perceptron_input <<  input_layer->state() , layers[0]->state();
    break;
  case 2:
    //cout <<"L1:"<< layers[1]->state().transpose() << endl;
    perceptron_input <<  input_layer->state() , layers[0]->state() , layers[1]->state();
    break;
  case 3:
    perceptron_input << input_layer->state() , layers[0]->state() , layers[1]->state() , layers[2]->state();
    break;
  default:
    throw "Not implemented exception (too many layers)";

  assert(perceptron_input.size()==get_concatenated_layers_size());

  }
  
  /*
   perceptron_input.segment(start_offset, end_offset) = input_layer->state();
   cout << perceptron_input.size() << " : " << get_concatenated_layers_size() << endl;
   cout << end_offset << input_layer->state().transpose() << endl;
   cout << "#"<< layers.size()<<endl;
   for (int i = 0; i < layers.size(); i++){
     start_offset = end_offset;
     end_offset = start_offset+layers[i]->get_n_units();
     cout << "here"<<i<<endl;
     cout << end_offset << ":" << layers[i]->state().transpose() << endl;
     cout << "there"<<endl;
     perceptron_input.segment(start_offset,end_offset) = layers[i]->state();
   }
   cout << endl << endl;
  */

   /*
   perceptron_input.segment(0, offset) = input_layer->state();// / input_layer->state().array().abs2().sum();
   for (int i = 0; i < layers.size(); i++){
       Vec tmp = layers[i]->state();
       for (int i = 0; i < tmp.size(); i++){
           perceptron_input[offset + i] = tmp[i];
       }
       offset += tmp.size();
       assert(offset == perceptron_input.size());
   }
   */
//        cerr << input_layer->state().array().abs().sum();
  
//            tmp /= tmp.array().abs2().sum();
//            cerr << "   "  << tmp.array().abs().sum();
//            perceptron_input.segment(offset, offset + layers[i]->get_n_units()) = layers[i]->state(); // For some reason, this raises segfautls / corrupt memory error
//        cerr << endl;
}




































NeuralPerceptron::NeuralPerceptron(){}
NeuralPerceptron::NeuralPerceptron(int x, int y) : x_dims(x), y_dims(y){
    bias = Vec::Zero(x);
    weights = Mat::Zero(x, y);
}
NeuralPerceptron::NeuralPerceptron(NeuralPerceptron const & other){
    weights = other.weights;
    bias = other.bias;
    x_dims = other.x_dims;
    y_dims = other.y_dims;
}
NeuralPerceptron& NeuralPerceptron::operator=(NeuralPerceptron const & other){
    weights = other.weights;
    bias = other.bias;
    x_dims = other.x_dims;
    y_dims = other.y_dims;
    return *this;
}
NeuralPerceptron::~NeuralPerceptron(){}

NeuralPerceptron& NeuralPerceptron::operator+= (NeuralPerceptron const &other){
    weights += other.weights;
    bias += other.bias;
    return *this;
}
NeuralPerceptron& NeuralPerceptron::operator-= (NeuralPerceptron const &other){
    weights -= other.weights;
    bias -= other.bias;
    return *this;
}
NeuralPerceptron& NeuralPerceptron::operator*= (float scalar){
    weights *= scalar;
    bias *= scalar;
    return *this;
}
NeuralPerceptron& NeuralPerceptron::operator/= (float scalar){
    weights /= scalar;
    bias /= scalar;
    return *this;
}
float NeuralPerceptron::sqL2norm()const{
    return weights.array().abs2().sum();
}
void NeuralPerceptron::save(string const &model_path, int iteration){
    if (iteration == 0){
        ofstream outw(model_path + "/global_perceptron_weights");
        outw << weights << endl;
        outw.close();

        ofstream outb(model_path + "/global_perceptron_bias");
        outb << bias << endl;
        outb.close();
    }else{
        ofstream outw(model_path + "/global_perceptron_weights" + std::to_string(iteration));
        outw << weights << endl;
        outw.close();

        ofstream outb(model_path + "/global_perceptron_bias" + std::to_string(iteration));
        outb << bias << endl;
        outb.close();
    }
}
void NeuralPerceptron::load(string const &model_path){
    load_matrix<Mat>(model_path + "/global_perceptron_weights", weights);
    load_matrix<Vec>(model_path + "/global_perceptron_bias", bias);
    assert(weights.cols() == y_dims && weights.rows() == x_dims);
    assert(bias.size() == x_dims);
}


