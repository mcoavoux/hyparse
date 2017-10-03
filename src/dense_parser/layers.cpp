#include "layers.h"


// static constant
const int LookupToHiddenLayer::MAX_VOCSIZE = 3000000;

void read_dimension_file(const string &filename, vector<int> &colds){
    ifstream instream(filename);
    colds.clear();
    int n;
    int tmp;

    string s;

    getline(instream, s);
    stringstream ss1(s);

    ss1 >> n;
    getline(instream, s);
    stringstream ss2(s);
    for (int i = 0; i < n; i++){
        ss2 >> tmp;
        colds.push_back(tmp);
    }
    instream.close();
}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

NeuralNetParams::NeuralNetParams(){
    learning_rate       = 0.01;
    decrease_constant   = 0;
    reg_lambda          = 0;
    embedding_epsilon   = 0.01;
    clipping_threshold  = 5;

    ada_grad = false;
    dropout = false;
    asgd = false;
    regularization = false;
    gradient_clipping = 0;

    n_input_features    = 0;
    n_hidden            = vector<int>();
    n_input             = vector<int>();
    n_output            = 0;

    embeddings_dimensions = vector<int>(6, 100);

    loss_function       = CROSS_ENTROPY;
    hidden_activation   = TANH;

    char_based_word_embeddings_dimension = 50;
    char_embeddings_dimension = 20;
    filter_size = 5;
    convolution = false;

}
NeuralNetParams::NeuralNetParams(const NeuralNetParams & o) : learning_rate(o.learning_rate),
                                                              decrease_constant(o.decrease_constant),
                                                              reg_lambda(o.reg_lambda),
                                                              embedding_epsilon(o.embedding_epsilon),
                                                              clipping_threshold(o.clipping_threshold),
                                                              ada_grad(o.ada_grad),
                                                              dropout(o.dropout),
                                                              regularization(o.regularization),
                                                              gradient_clipping(o.gradient_clipping),
                                                              n_input_features(o.n_input_features),
                                                              n_output(o.n_output),
                                                              n_input(o.n_input),
                                                              n_hidden(o.n_hidden),
                                                              embeddings_dimensions(o.embeddings_dimensions),
                                                              loss_function(o.loss_function),
                                                              hidden_activation(o.hidden_activation),
                                                              lexical_features(o.lexical_features),
                                                              char_based_word_embeddings_dimension(o.char_based_word_embeddings_dimension),
                                                              char_embeddings_dimension(o.char_embeddings_dimension),
                                                              filter_size(o.filter_size),
                                                              convolution(o.convolution),
                                                              asgd(o.asgd){}

NeuralNetParams& NeuralNetParams::operator=(const NeuralNetParams & o){
    learning_rate       = o.learning_rate;
    decrease_constant   = o.decrease_constant;
    reg_lambda          = o.reg_lambda;
    embedding_epsilon   = o.embedding_epsilon;
    clipping_threshold  = o.clipping_threshold;
    ada_grad            = o.ada_grad;
    dropout             = o.dropout;
    regularization      = o.regularization;
    gradient_clipping   = o.gradient_clipping;
    n_input_features    = o.n_input_features;
    n_hidden            = o.n_hidden;
    n_output            = o.n_output;
    n_input             = o.n_input;
    embeddings_dimensions=o.embeddings_dimensions;
    loss_function       = o.loss_function;
    hidden_activation   = o.hidden_activation;
    lexical_features    = o.lexical_features;
    char_based_word_embeddings_dimension = o.char_based_word_embeddings_dimension;
    char_embeddings_dimension = o.char_embeddings_dimension;
    filter_size         = o.filter_size;
    convolution         = o.convolution;
    asgd                = o.asgd;
    return *this;
}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

AbstractLayer::AbstractLayer(int n_units) : n_units(n_units), next_layer(NULL){}

AbstractLayer::~AbstractLayer(){}
double AbstractLayer::compute_cost(int gold){throw "AbstractLayer::compute_cost: Not implemented error";}
bool AbstractLayer::is_lookup_layer(){ return false;}

void AbstractLayer::add_epsilon_b(int i, double epsilon){ b[i] += epsilon; }

const Vec & AbstractLayer::state(){            return h; }
const Vec & AbstractLayer::gbias(){            return gb;}
void AbstractLayer::generate_dropout_masks(){  dropout_mask = (Vec::Random(n_units).array() < 0).cast<double>(); }
int AbstractLayer::get_pred(){                 return pred; }
int AbstractLayer::get_n_units(){              return n_units;}

int AbstractLayer::predict(){
    h.maxCoeff(&pred);
    return pred;
}

void AbstractLayer::set_pred(int newp){
    pred = newp;
}

void AbstractLayer::set_next_layer(Layer* next){
    next_layer = next;
}
Layer* AbstractLayer::get_next_layer(){
    return next_layer;
}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////



LookupToHiddenLayer::LookupToHiddenLayer(vector<int> n_input, int n_input_features, int n_units) : AbstractLayer(n_units), n_input(n_input), n_input_features(n_input_features){
    this->E = this->gE = this->dgE = vector<Vec>(MAX_VOCSIZE);
    this->exists = this->activated = vector<bool>(MAX_VOCSIZE, false);
    this->n_vecs = 0;
    this->indexes.clear();
    this->dimensions = vector<int>(20,50);
    this->epsilon = 0.01;
    this->random_init = true;

    // Initialisation :
    h = a = b = gb = dgb = dropout_mask = Vec::Zero(n_units);

    int n_total_input = 0;
    for (int i = 0; i < n_input_features; i++){
        n_total_input += n_input[i];
    }
    h_buff.resize(n_input_features);
    w.resize(n_input_features);
    gw.resize(n_input_features);
    dgw.resize(n_input_features);

    for (int i = 0; i < n_input_features; i++){
        w[i] = Mat::Random(n_units, n_input[i]) * sqrt(6.0 / (n_units + n_total_input));
        gw[i] = dgw[i] = Mat::Zero(n_units, n_input[i]);
    }
    this->convolution = false;
}

LookupToHiddenLayer::~LookupToHiddenLayer(){}

void LookupToHiddenLayer::copy(const LookupToHiddenLayer & other){
    dimensions      = other.dimensions;
    colsizes        = other.colsizes;
    indexes         = other.indexes;
    input_features  = other.input_features;
    unique_features = other.unique_features;
    exists          = other.exists;
    activated       = other.activated;
    E               = other.E;
    gE              = other.gE;
    dgE             = other.dgE;
    n_vecs          = other.n_vecs;
    epsilon         = other.epsilon;
    n_input         = other.n_input;
    n_input_features= other.n_input_features;
    h_buff          = other.h_buff;
    w               = other.w;
    gw              = other.gw;
    dgw             = other.dgw;
    random_init     = other.random_init;
    convolution     = other.convolution;
    conv_layer      = other.conv_layer;
    conv_input      = conv_input;
    lexical_features=other.lexical_features;
    precomputed_hidden=other.precomputed_hidden;

    n_units         = other.n_units;
    pred            = other.pred;
    h               = other.h;
    a               = other.a;
    b               = other.b;
    gb              = other.gb;
    dgb             = other.dgb;
    dropout_mask    = other.dropout_mask;
    next_layer      = NULL;
}




void LookupToHiddenLayer::precompute_hidden_layer(){
    for (int tok = 0; tok < exists.size(); tok ++){
        //cerr << tok << " / " << exists.size() << endl;
        if (exists[tok]){
            for (int i = 0; i < n_input_features; i++){
                if (n_input[i] == E[tok].size()){
                    precomputed_hidden[make_pair(tok, i)] = w[i] * E[tok];
                }
            }
        }
    }
}

void LookupToHiddenLayer::display_summary_convolution(ostream &os){
    if (convolution){
        conv_layer.display_summary(os);
    }
}


void LookupToHiddenLayer::set_random_init(bool ri){                      random_init = ri; }
void LookupToHiddenLayer::set_epsilon(double eps){                       epsilon = eps; }
void LookupToHiddenLayer::get_unique_features(vector<int> &features){    features = unique_features; }
int LookupToHiddenLayer::get_n_input_features(){                         return n_input_features; }
void LookupToHiddenLayer::compute_bias_gradient(int target){ throw "LookupToHiddenLayer::compute_bias_gradient(int target): Not implemented error"; }
const Mat & LookupToHiddenLayer::gweights(int f){ return gw[f];}
const Vec & LookupToHiddenLayer::gembedding(int i){ return gE[i]; }
bool LookupToHiddenLayer::is_lookup_layer(){ return true; }

void LookupToHiddenLayer::rescale_weights(bool down){
    double factor = down ? 0.5 : 2.0;

#pragma omp parallel for default(shared)
    for (int i = 0; i < n_input_features; i++){
        w[i] *= factor;
    }
}

// NOTE : for now, on ignore la norme des embeddings
double LookupToHiddenLayer::l2_cost(double reg_lambda){
    double cost = 0.0;

#pragma omp parallel for default(shared)
    for (int i = 0; i < n_input_features; i++){
        cost += w[i].array().abs2().sum();
    }
    return cost * reg_lambda;
}

// NOTE : for now, on ignore la norme des embeddings
void LookupToHiddenLayer::add_l2_gradient(double reg_lambda){

#pragma omp parallel for default(shared)
    for (int i = 0; i < n_input_features; i++){
        gw[i] += 2 * reg_lambda * w[i];
    }
}



void LookupToHiddenLayer::compute_unique(const vector<int> &input){
    input_features = input;
    std::sort(input_features.begin(), input_features.end());
    unique_features = input_features;
    auto it = std::unique(unique_features.begin(), unique_features.end());
    unique_features.assign(unique_features.begin(), it);
    for (int i = 0; i < input_features.size(); i++){
        int n = 1;
        while (i < input.size()-1 && input_features[i] == input_features[i+1]){
            i++; n++;
        }
    }
}

void LookupToHiddenLayer::update_sgd(double step_size){

#pragma omp parallel for default(shared)
    for (int i = 0; i < n_input_features; i++){
        w[i] -= step_size * gw[i];
    }
    b -= step_size * gb;

#pragma omp parallel for default(shared)
    for (int i = 0; i < unique_features.size(); i++){
        E[unique_features[i]] -= step_size * gE[unique_features[i]];
        gE[unique_features[i]].fill(0.0);
    }
    if (convolution){
        conv_layer.update(step_size);
    }
}


/// TODO : finish here
void LookupToHiddenLayer::update_asgd(double step_size, double T){

    for (int i = 0; i < n_input_features; i++){
        w[i] -= step_size * gw[i];
        dgw[i] -= T * step_size * gw[i];
    }
    b -= step_size * gb;
    dgb -= T * step_size * gb;

    for (int i = 0; i < unique_features.size(); i++){
        E[unique_features[i]] -= step_size * gE[unique_features[i]];
        dgE[unique_features[i]] -= T * step_size * gE[unique_features[i]];
        gE[unique_features[i]].fill(0.0);
    }
    if (convolution){
        conv_layer.update_asgd(step_size, T);
    }
}

void LookupToHiddenLayer::update_adagrad(double step_size, double ada_epsilon){

#pragma omp parallel for default(shared)
    for (int i = 0; i < n_input_features; i++){
        dgw[i] += gw[i].array().abs2().matrix();
        w[i] -= step_size *(gw[i].array() / (ada_epsilon + dgw[i].array().sqrt())).matrix();
    }
    dgb += gb.array().abs2().matrix();
    b -= step_size * (gb.array() / (ada_epsilon + dgb.array().sqrt())).matrix();

#pragma omp parallel for default(shared)
    for (int i = 0; i < unique_features.size(); i++){
        dgE[unique_features[i]] += gE[unique_features[i]].array().abs2().matrix();
        E[unique_features[i]] -= step_size * (gE[unique_features[i]].array() / (ada_epsilon + dgE[unique_features[i]].array().sqrt())).matrix();
        gE[unique_features[i]] *= 0;
    }
    if (convolution){
        conv_layer.update_adagrad(step_size, ada_epsilon);
    }
}

void LookupToHiddenLayer::clip_gradient(double threshold){
    double norm =0;
    for (int i = 0; i < n_input_features; i++){
        norm = gw[i].norm();
        if (norm > threshold)
            gw[i] *= threshold / norm;
    }
    for (int i = 0; i < unique_features.size(); i++){
        norm = gE[unique_features[i]].norm();
        if (norm > threshold)
            gE[unique_features[i]] *= threshold / norm;
    }
}





int LookupToHiddenLayer::get_embedding_size(unsigned int i){                 return E[i].size(); }

void LookupToHiddenLayer::add_epsilon(int i, int j, double epsilon){         E[i](j) += epsilon;}
void LookupToHiddenLayer::add_epsilon(int i, int j, int k, double epsilon){  w[i](j,k) += epsilon;}


void LookupToHiddenLayer::check_embedding(unsigned int i){
    if (! exists[i]){
        exists[i] = true;
        int colidx = IntegerEncoder::get()->get_column(i) + 1;      // get col number. +1 because -1 is column code for Phrase structure symbols
        int dim = dimensions[colidx];
        if (random_init){  E[i] = Vec::Random(dim) * epsilon; }
        else            {  E[i] = Vec::Zero(dim);             }
        gE[i] = dgE[i] = Vec::Zero(dim);                            // Initialize gradient history for adagrad
        n_vecs ++;                                                  // vector count
        colsizes[colidx] ++;                                        // vector count by columns
        indexes.push_back(i);                                       // maintain list of existing vectors
    }
}
void LookupToHiddenLayer::check_embedding(const vector<int> &input){
    for (unsigned int i : input)
        check_embedding(i);
}

void LookupToHiddenLayer::set_dimensions(vector<int> dimensions){
    this->dimensions = dimensions;
    this->colsizes = vector<int>(this->dimensions.size(), 0);
}

bool LookupToHiddenLayer::activate(int i){
    if (activated[i]){ return true; }
    activated[i] = true;
    return false;
}

void LookupToHiddenLayer::deactivate(int i){
    activated[i] = false;
}


double LookupToHiddenLayer::fprop_std(const vector<int> &input, int target, const vector<bool> &actions){
    check_embedding(input);
    compute_unique(input);

#pragma omp parallel for default(shared)
    for (int i = 0; i < n_input_features; i++){
        h_buff[i] = w[i] * E[input[i]];
    }
    a = b;
    for (int i = 0; i < n_input_features; i++){
        a += h_buff[i];
    }
    if (convolution){
        for (int i = 0; i < lexical_features.size(); i++){
            conv_input.at(i) = input[lexical_features.at(i)];
        }
        conv_layer.fprop(conv_input, a);
    }
    compute_activation(actions);
    if (is_output_layer()){ return compute_cost(target);  }
    else             { return 0.0; }
}

double LookupToHiddenLayer::fast_fprop_std(const vector<int> &input, int target, const vector<bool> &actions){
    a = b;
    for (int i = 0; i < n_input_features; i++){
        pair<int,int> p(input[i], i);
        auto it = precomputed_hidden.find(p);
        if (it != precomputed_hidden.end()){
            a += it->second;
        }
    }
    if (convolution){
        for (int i = 0; i < lexical_features.size(); i++){
            conv_input.at(i) = input[lexical_features.at(i)];
        }
        conv_layer.fprop(conv_input, a);
    }
    compute_activation(actions);
    return 0.0;
}

void LookupToHiddenLayer::bprop_std(const vector<int> &input, int target){
    compute_bias_gradient(target);

#pragma omp parallel for default(shared)
    for (int i = 0; i < n_input_features; i++){
        gw[i] = gb * E[input[i]].transpose();
    }
    for (int i = 0; i < n_input_features; i++){
        gE[input[i]] += w[i].transpose() * gb;
    }
    if (convolution){
        conv_layer.bprop(conv_input, gb);
    }
}

double LookupToHiddenLayer::embedding_l2_cost(double reg_lambda) const{   // computes L2 cost on embeddings
    double cost = 0.0;
#pragma omp parallel for default(shared)
    for (int i = 0; i < indexes.size(); i++){
        cost += E[indexes[i]].array().abs2().sum();
    }
    return reg_lambda * cost;
}

void LookupToHiddenLayer::regularize_embeddings_sgd(double step_size, double reg_lambda){                         // regularisation update for non activated features.
    int s = indexes.size();
    int i = 0;
#pragma omp parallel for default(shared) private(i) schedule(static, 100)
    for (i = 0; i < s; i++){
        if (! activated[indexes[i]]){
            E[indexes[i]] *= (1 - step_size * 2 * reg_lambda);
        }
    }
}
void LookupToHiddenLayer::regularize_embeddings_adagrad(double step_size, double reg_lambda, double adaEpsilon){  // idem with adagrad update
    int s = indexes.size();
    int i = 0;
#pragma omp parallel for default(shared) private(i) schedule(static, 100)
    for (i = 0; i < s; i++){
        if (! activated[indexes[i]]){
            gE[indexes[i]] = (2 * reg_lambda) * E[indexes[i]];
            dgE[indexes[i]] += gE[indexes[i]].array().abs2().matrix();
            E[indexes[i]] -= step_size * (gE[indexes[i]].array() / (adaEpsilon + dgE[indexes[i]].array().sqrt())).matrix();
        }
    }
}









void LookupToHiddenLayer::dump_parameters(const string dirname){
    for (int i = 0; i < n_input_features; i++){
        std::ofstream outw(dirname + "/input_to_hidden_weights" + std::to_string(i));
        outw << w[i];
        outw.close();
    }
    std::ofstream outb(dirname + "/input_to_hidden_bias");
    outb << b;
    outb.close();
    dump_embeddings(dirname);

    std::ofstream outlex(dirname + "/lexical_features");
    outlex << lexical_features.size() << endl;
    for (int i = 0; i < lexical_features.size(); i++){
        outlex << lexical_features[i] << " ";
    }
    outlex << endl;
    outlex.close();
    if (convolution){
        conv_layer.save(dirname);
    }
}

void LookupToHiddenLayer::load_parameters(const string dirname){
    load_embeddings(dirname);

    for (int i = 0; i < n_input_features; i++){
        load_matrix<Mat>(dirname + "/input_to_hidden_weights" + std::to_string(i), w[i]);
    }
    load_matrix<Vec>(dirname + "/input_to_hidden_bias", b);

    random_init = false;

    lexical_features.clear();
    std::ifstream inlex(dirname + "/lexical_features");
    string tmp;
    getline(inlex, tmp);
    int tab_size = stoi(tmp);
    lexical_features.resize(tab_size);
    getline(inlex, tmp);
    istringstream buff(tmp);
    for (int i = 0; i < tab_size; i++){
        buff >> lexical_features[i];
    }
    conv_input.resize(tab_size);

    if (convolution){
        conv_layer.load(dirname);
    }
}

void LookupToHiddenLayer::dump_embeddings(const string dirname){
    std::ofstream oute(dirname + "/embeddings");
    for (int i = 0; i < MAX_VOCSIZE; i++){
        if (exists[i]){
            oute << i << " " << E[i].size() << " " << IntegerEncoder::get()->decode8(i) << endl;
            oute << E[i].transpose() << endl;
        }
    }
//    cerr << "Embedding matrix vocabulary size for each column:" << endl;
//    cerr << "\t0 (Non terminal) : " << colsizes.at(0) << endl;
//    for (int i = 1; i < colsizes.size(); i++){
//        if (colsizes.at(i) > 0){
//            cerr << "\t" << i << " : " << colsizes.at(i) << endl;
//        }
//    }
    oute.close();
}
void LookupToHiddenLayer::load_embeddings(const string dirname){
    read_dimension_file(dirname+"/embed_dims", dimensions);

    colsizes = vector<int>(dimensions.size(), 0);

    ifstream instream(dirname+"/embeddings");
    string line;
    int idx;
    int dim;
    double val;
    int maxcol = 0;
    while(getline(instream, line)){
        stringstream buffer(line);
        buffer >> idx;
        buffer >> dim;
        Vec v(dim);
        getline(instream, line);
        if ((line.find("nan") != string::npos) || (line.find("inf") != string::npos)){
            cerr << "Error, encountered 'nan' or 'inf' in parameters" << endl;
            cerr << "Aborting ..." << endl;
            exit(1);
        }

        stringstream vbuf(line);
        for (int i = 0; i < dim; i++){
            vbuf>> val;
            v(i) = val;
        }
        int col = IntegerEncoder::get()->get_column(idx) + 1;
        if (col > maxcol) maxcol = col;
        E[idx] = v;
        gE[idx] = dgE[idx] = Vec::Zero(v.size());
        exists[idx] = true;
        n_vecs ++;
        indexes.push_back(idx);
    }
}

void LookupToHiddenLayer::load_and_encode(const string filename){
    ifstream instream(filename);
    string line;

    double val;
    string word;
    vector<double> tmp;

    getline(instream, line);
    int n_lines, dim;
    stringstream ss(line);
    ss >> n_lines;
    ss >> dim;
    assert(dim == dimensions[1]);

    while(getline(instream, line)){
        stringstream buffer(line);
        tmp.clear();

        buffer >> word;

        while (buffer >> val){
            tmp.push_back(val);
        }
        if ( tmp.size() != dim){
            cerr << "Encountered vector with dimension " << tmp.size() << " while loading lookup table" << endl;
            continue;
        }

        Vec v(tmp.size());
        for (int i = 0; i < tmp.size(); i++){
            v(i) = tmp[i];
        }
        int id = IntegerEncoder::get()->encode(word, 0);        // Encoded on sword col -> 0 : encode on word column instead (for PTB)
        int colidx = IntegerEncoder::get()->get_column(id) + 1;

        E[id] = v;
        exists[id] = true;
        dgE[id] = gE[id] = Vec::Zero(dimensions[colidx]);               // Initialize gradient and gradient history
        n_vecs ++;

        colsizes[colidx] ++;                                        // vector count by columns
        indexes.push_back(id);                                           // maintain list of existing vectors
    }
}










void LookupToHiddenLayer::set_convolution_options(const NeuralNetParams& params){
    this->lexical_features = params.lexical_features;
    this->conv_input = vector<int>(lexical_features.size());
    convolution = params.convolution;
    if (convolution){
        conv_layer.set_dimension(params.char_based_word_embeddings_dimension);
        conv_layer.set_char_dimension(params.char_embeddings_dimension);
        conv_layer.set_filter_size(params.filter_size);
        conv_layer.set_n_features(lexical_features.size());
        conv_layer.set_n_units(n_units);
        conv_layer.initialize();
    }
}

void LookupToHiddenLayer::set_convolution(bool b){
    convolution = b;
}

void LookupToHiddenLayer::average(double T){
    for (int i = 0; i < indexes.size(); i++){
        E[indexes[i]] -= dgE[indexes[i]] / T;
    }
    b -= dgb / T;
    for (int i = 0; i < n_input_features; i++){
        w[i] -= dgw[i] / T;
    }
    if (convolution){
        conv_layer.average(T);
    }
}





















////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
LookupLayerSoftmax::LookupLayerSoftmax(vector<int> n_input, int n_input_features, int n_units) : LookupToHiddenLayer(n_input, n_input_features, n_units){}
void LookupLayerSoftmax::compute_activation(const vector<bool> &actions){
    h = (a.array() - a.maxCoeff()).exp();
    for (int i = 0; i < n_units; i++) if (! actions[i]) h[i] = 0;
    h /= h.sum();
}

void LookupLayerSoftmax::compute_bias_gradient(int target){
    gb = h;
    gb(target) -= 1;
}

double LookupLayerSoftmax::compute_cost(int gold){ return - log(h[gold]); }
bool LookupLayerSoftmax::is_output_layer(){ return true;}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

LookupLayerPerceptron::LookupLayerPerceptron(vector<int> n_input, int n_input_features, int n_units) : LookupToHiddenLayer(n_input, n_input_features, n_units){}

void LookupLayerPerceptron::compute_activation(const vector<bool> &actions){
    h = a;
}

double LookupLayerPerceptron::compute_cost(int gold){
    predict();
    if (pred == gold) return 0;
    return h[pred] - h[gold];
}

void LookupLayerPerceptron::compute_bias_gradient(int target){
    gb.fill(0);
    gb[target] -= 1;
    gb[pred] += 1;
}
bool LookupLayerPerceptron::is_output_layer(){ return true;}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

LookupLayerCube::LookupLayerCube(vector<int> n_input, int n_input_features, int n_units) : LookupToHiddenLayer(n_input, n_input_features, n_units){}
void LookupLayerCube::compute_activation(const vector<bool> &actions){
    h = a.array() * a.array() * a.array();
}
void LookupLayerCube::compute_bias_gradient(int target){
    gb = (next_layer->weights().transpose() * next_layer->gbias()).array() * 3 * a.array() * a.array();
}
bool LookupLayerCube::is_output_layer(){ return false;}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

LookupLayerTanh::LookupLayerTanh(vector<int> n_input, int n_input_features, int n_units) : LookupToHiddenLayer(n_input, n_input_features, n_units){}
void LookupLayerTanh::compute_activation(const vector<bool> &actions){
    h = a.unaryExpr(std::ptr_fun<double, double>(tanh));
}
void LookupLayerTanh::compute_bias_gradient(int target){
    gb = (next_layer->weights().transpose() * next_layer->gbias()).array() * (1 - h.array().abs2());
}
bool LookupLayerTanh::is_output_layer(){ return false;}


////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////



LookupLayerRelu::LookupLayerRelu(vector<int> n_input, int n_input_features, int n_units) : LookupToHiddenLayer(n_input, n_input_features, n_units){
    mask = zero = Vec::Zero(n_units);
}
void LookupLayerRelu::compute_activation(const vector<bool> &actions){
    mask.array() = (a.array() > zero.array()).cast<double>();
    h = a.cwiseProduct(mask);
}
void LookupLayerRelu::compute_bias_gradient(int target){
    gb = (next_layer->weights().transpose() * next_layer->gbias()).array() * mask.array();
}
bool LookupLayerRelu::is_output_layer(){ return false;}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

//////////////// STANDARD LAYERS ///////////////////////


Layer::Layer(int n_input, int n_units) : AbstractLayer(n_units), n_input(n_input){
    h = a = b = gb = dgb = Vec::Zero(n_units);
    w = Mat::Random(n_units, n_input) * sqrt(6.0 / (n_units + n_input));
    gw = dgw = Mat::Zero(n_units, n_input);
}
Layer::~Layer(){}

double Layer::fprop_std(const Vec & input, int target, const vector<bool> &actions){throw "Layer::fprop_std(const Vec & input, int target): Not implemented error";}
void Layer::bprop_std(const Vec & input, int target){throw "Layer::bprop_std(const Vec & input, int target): Not implemented error";}

void Layer::rescale_weights(bool down){
    if (down){ w *= 0.5;}
    else{      w *= 2.0;}
}

double Layer::l2_cost(double reg_lambda){           return reg_lambda * w.array().abs2().sum(); }
void Layer::add_l2_gradient(double reg_lambda){   gw += 2 * reg_lambda * w; }
void Layer::update_sgd(double step_size){
    b -= step_size * gb;
    w -= step_size * gw;
}
void Layer::update_asgd(double step_size, double T){
    b -= step_size * gb;
    dgb -= T * step_size * gb;
    w -= step_size * gw;
    dgw -= T * step_size * gw;
}

void Layer::update_adagrad(double step_size, double ada_epsilon){
    dgb += gb.array().abs2().matrix();
    dgw += gw.array().abs2().matrix();
    b -= step_size * (gb.array() / (ada_epsilon + dgb.array().sqrt())).matrix();
    w -= step_size * (gw.array() / (ada_epsilon + dgw.array().sqrt())).matrix();
}
void Layer::average(double T){
    b -= dgb / T;
    w -= dgw / T;
}

void Layer::clip_gradient(double threshold){
    double norm = gw.norm();
    if (norm > threshold)
        gw *= threshold / norm;
}

const Mat & Layer::weights(){  return w; }
const Mat & Layer::gweights(){ return gw;}
int Layer::get_n_input(){ return n_input;}

void Layer::add_epsilon(int i, int j, double epsilon){
    w(i,j) += epsilon;
}

void Layer::dump_parameters(const string dirname, int i){
    std::ofstream outw2(dirname + "/hidden_weights" + std::to_string(i));
    outw2 << w;
    outw2.close();
    std::ofstream outb2(dirname + "/hidden_bias" + std::to_string(i));
    outb2 << b;
    outb2.close();
}

void Layer::load_parameters(const string dirname, int i){
    load_matrix<Mat>(dirname + "/hidden_weights" + std::to_string(i), w);
    load_matrix<Vec>(dirname + "/hidden_bias" + std::to_string(i), b);
    h = a = Vec::Zero(w.rows());
}




////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////


SoftmaxLayer::SoftmaxLayer(int n_input, int n_units) : Layer(n_input, n_units){}
SoftmaxLayer::~SoftmaxLayer(){}
double SoftmaxLayer::compute_cost(int gold){
    return - log(h[gold]);
}
double SoftmaxLayer::fprop_std(const Vec &input, int target, const vector<bool> &actions){
    //bool allowed = false;
    a = w * input + b;
    h = (a.array() - a.maxCoeff()).exp();
//    if (h.size() != actions.size() || h.size() != n_units){
//        cerr << "HUGE PROBLEM HERE" << endl;
//    }
    assert(actions.size() == n_units);
    for (int i = 0; i < actions.size(); i++){
        //if (actions[i]) allowed = true;
        //else
        if (!actions[i])
            h[i] = 0;
    }
    //if (allowed){
        //DBG("normalisation softmax")
        h /= h.sum();
    //}
//    else{
//        //DBG("pas de normalisation" << actions.size() << "  " << n_units)
//    }
    return compute_cost(target);
}
void SoftmaxLayer::bprop_std(const Vec &input, int target){
    gb = h;
    gb[target] -= 1;
    gw = gb * input.transpose();
}
bool SoftmaxLayer::is_output_layer(){ return true;}



////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////


PerceptronLayer::PerceptronLayer(int n_input, int n_units) : Layer(n_input, n_units){}
PerceptronLayer::~PerceptronLayer(){}

double PerceptronLayer::compute_cost(int gold){
    predict();
    if (pred == gold) return 0;
    return h[pred] - h[gold];
}
double PerceptronLayer::fprop_std(const Vec &input, int target, const vector<bool> &actions){
    h = w * input + b;
    double min = h.minCoeff();
    for (int i = 0; i < n_units; i++) if (! actions[i]) h[i] = min;
    return compute_cost(target);
}
void PerceptronLayer::bprop_std(const Vec &input, int target){
    gb.fill(0);
    gb[target] -= 1;
    gb[pred] += 1;
    gw = gb * input.transpose();
}


TanhLayer::TanhLayer(int n_input, int n_units) : Layer(n_input, n_units){}
TanhLayer::~TanhLayer(){}
double TanhLayer::fprop_std(const Vec & input, int target, const vector<bool> &actions){
    a = w * input + b;
    h = a.unaryExpr(std::ptr_fun<double, double>(tanh));
    return 0;
}
void TanhLayer::bprop_std(const Vec & input, int target){
    gb = (next_layer->weights().transpose() * next_layer->gbias()).array() * (1 - h.array().abs2());
    gw = gb * input.transpose();
}




CubeLayer::CubeLayer(int n_input, int n_units) : Layer(n_input, n_units){}
CubeLayer::~CubeLayer(){}
double CubeLayer::fprop_std(const Vec & input, int target, const vector<bool> &actions){
    a = w * input + b;
    h = a.array() * a.array() * a.array();
    return 0;
}
void CubeLayer::bprop_std(const Vec & input, int target){
    gb = (next_layer->weights().transpose() * next_layer->gbias()).array() * 3 * a.array() * a.array();
    gw = gb * input.transpose();
}



ReluLayer::ReluLayer(int n_input, int n_units) : Layer(n_input, n_units){
    mask = zero = Vec::Zero(n_units);
}
ReluLayer::~ReluLayer(){}
double ReluLayer::fprop_std(const Vec & input, int target, const vector<bool> &actions){
    a = w * input + b;
    mask.array() = (a.array() > zero.array()).cast<double>();
    h = a.cwiseProduct(mask);
    return 0;
}
void ReluLayer::bprop_std(const Vec & input, int target){
    gb = (next_layer->weights().transpose() * next_layer->gbias()).array() * mask.array();
    gw = gb * input.transpose();
}



