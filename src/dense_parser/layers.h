#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <string>
#include <Eigen/Dense>
#include <tuple>
#include <boost/functional/hash.hpp>


#include "globals.h"
#include "character_convolution.h"

#define DBG(x) std::cerr << x << std::endl;

using std::vector;
using std::string;
using std::cerr;
using std::endl;
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;




// Globals
enum {TANH, CUBE, RELU};
enum {CROSS_ENTROPY, PERCEPTRON };
const string LOSS_FUNCTIONS[] = {"cross entropy", "perceptron", "hinge-loss"};
const string ACTIVATIONS[] = {"tanh", "cube", "ReLU"};


void read_dimension_file(const string &filename, vector<int> &colds);


struct NeuralNetParams{
    double learning_rate, decrease_constant, reg_lambda, embedding_epsilon, clipping_threshold;
    bool ada_grad, dropout, regularization, gradient_clipping;

    int n_input_features, n_output;
    vector<int> n_input, n_hidden, embeddings_dimensions;

    int loss_function;          // Cross entropy or perceptron
    int hidden_activation;      //tanh, relu, cube

    vector<int> lexical_features;
    int char_based_word_embeddings_dimension;
    int char_embeddings_dimension;
    int filter_size;
    bool convolution;
    bool asgd;

    NeuralNetParams();
    NeuralNetParams(const NeuralNetParams & o);
    NeuralNetParams& operator=(const NeuralNetParams & o);
};


class Layer;
/**
 * Architecture :
 *      AstractLayer
 *          | Layer
 *              | SoftmaxLayer          // output layer
 *              | PerceptronLayer       // output layer
 *              | ReLuLayer         // hidden
 *              | SoftmaxLayer      // hidden
 *              | TanhLayer         // hidden
 *          | LookupLayer           //hidden or output layer
 */
class AbstractLayer{ // rather semi-abstract

public:
    int n_units;                            // number of units
    int pred;                               // predicted class(useful for output layer only)
    Vec h, a, b, gb, dgb, dropout_mask;     // h : activated layer, a : preactivated layer, b : bias, gb : gradient of bias, dgb : gradient history of bias

    Layer *next_layer;

public:
    AbstractLayer(int n_units);

    virtual ~AbstractLayer();
    virtual double compute_cost(int gold);
    virtual void rescale_weights(bool down) = 0;
    virtual double l2_cost(double reg_lambda) = 0;
    virtual void add_l2_gradient(double reg_lambda) = 0;
    virtual void update_sgd(double step_size) = 0;
    virtual void update_asgd(double step_size, double T) = 0;
    virtual void update_adagrad(double step_size, double ada_epsilon) = 0;
    virtual void clip_gradient(double threshold) = 0;
    virtual bool is_lookup_layer();
    virtual bool is_output_layer() = 0;



    void add_epsilon_b(int i, double epsilon);

    const Vec & state();
    const Vec & gbias();
    void generate_dropout_masks();
    int get_pred();
    int get_n_units();
    int predict();
    void set_pred(int newp);
    void set_next_layer(Layer* next);
    Layer* get_next_layer();
};




class LookupToHiddenLayer : public AbstractLayer{

protected:
    // lookup layer
    vector<int> dimensions, colsizes, indexes, input_features, unique_features;  // dimension of embedding for each col / number of vecs per col / idxes of non-empty vectors
    vector<bool> exists, activated;         // true if vector already exists / true if vector is activated
    vector<Vec> E, gE, dgE;                 // Embeddings / gradient / history of gradient
    int n_vecs;                             // total number of vecs
    double epsilon;                         // embeddings are initialised in [-epsilon, epsilon]

    // first hidden layer
    vector<int> n_input;
    int n_input_features;

    vector<Vec> h_buff;
    vector<Mat> w, gw, dgw;

    bool random_init;

    static const int MAX_VOCSIZE;           // static maximum size of embedding matrix.

    bool convolution;
    CharacterConvolution conv_layer;
    vector<int> conv_input;
    vector<int> lexical_features;


    unordered_map<pair<int, int>, Vec, boost::hash<pair<int,int>>> precomputed_hidden;

public:
    // Initialisation
    LookupToHiddenLayer(vector<int> n_input, int n_input_features, int n_units);
    virtual ~LookupToHiddenLayer();
    void copy(const LookupToHiddenLayer & other);
    virtual LookupToHiddenLayer* deepcopy() const = 0;

    void precompute_hidden_layer();

    void display_summary_convolution(ostream &os);

    void set_random_init(bool ri);
    void set_epsilon(double eps);
    void get_unique_features(vector<int> &features);
    int get_n_input_features();
    virtual void compute_bias_gradient(int target);
    virtual void compute_activation(const vector<bool> &actions) = 0;

    const Mat & gweights(int f);
    const Vec & gembedding(int i);
    bool is_lookup_layer();

    void rescale_weights(bool down);
    double l2_cost(double reg_lambda);// NOTE : for now, on ignore la norme des embeddings
    void add_l2_gradient(double reg_lambda);// NOTE : for now, on ignore la norme des embeddings

    void compute_unique(const vector<int> &input);
    void update_sgd(double step_size);
    void update_asgd(double step_size, double T);
    void update_adagrad(double step_size, double ada_epsilon);
    void clip_gradient(double threshold);

    int get_embedding_size(unsigned int i);
    void add_epsilon(int i, int j, double epsilon);
    void add_epsilon(int i, int j, int k, double epsilon);
    void check_embedding(unsigned int i);
    void check_embedding(const vector<int> &input);
    void set_dimensions(vector<int> dimensions);

    bool activate(int i);
    void deactivate(int i);

    double fprop_std(const vector<int> &input, int target, const vector<bool> &actions);
    void bprop_std(const vector<int> &input, int target);

    double fast_fprop_std(const vector<int> &input, int target, const vector<bool> &actions);


    double embedding_l2_cost(double reg_lambda) const;
    void regularize_embeddings_sgd(double step_size, double reg_lambda);
    void regularize_embeddings_adagrad(double step_size, double reg_lambda, double adaEpsilon);


    void dump_parameters(const string dirname);
    void load_parameters(const string dirname);
    void dump_embeddings(const string dirname);
    void load_embeddings(const string dirname);
    void load_and_encode(const string filename);

    void set_convolution_options(const NeuralNetParams& params);
    void set_convolution(bool b);
    void average(double T);

};


class LookupLayerSoftmax : public LookupToHiddenLayer{
public:
    LookupLayerSoftmax(vector<int> n_input, int n_input_features, int n_units);
    void compute_activation(const vector<bool> &actions);
    void compute_bias_gradient(int target);
    double compute_cost(int gold);
    bool is_output_layer();
    LookupToHiddenLayer * deepcopy() const{
        LookupToHiddenLayer * newl = new LookupLayerSoftmax(vector<int>(), 0,0);
        newl->copy(*this);
        return newl;
    }
};

class LookupLayerPerceptron: public LookupToHiddenLayer{
public:
    LookupLayerPerceptron(vector<int> n_input, int n_input_features, int n_units);
    void compute_activation(const vector<bool> &actions);
    double compute_cost(int gold);
    void compute_bias_gradient(int target);
    bool is_output_layer();
    LookupToHiddenLayer * deepcopy() const{
        LookupToHiddenLayer * newl = new LookupLayerPerceptron(vector<int>(), 0,0);
        newl->copy(*this);
        return newl;
    }

};

class LookupLayerCube : public LookupToHiddenLayer{
public:
    LookupLayerCube(vector<int> n_input, int n_input_features, int n_units);
    void compute_activation(const vector<bool> &actions);
    void compute_bias_gradient(int target);
    bool is_output_layer();
    LookupToHiddenLayer * deepcopy()const {
        LookupToHiddenLayer * newl = new LookupLayerCube(vector<int>(), 0,0);
        newl->copy(*this);
        return newl;
    }
};

class LookupLayerTanh: public LookupToHiddenLayer{
public:
    LookupLayerTanh(vector<int> n_input, int n_input_features, int n_units);
    void compute_activation(const vector<bool> &actions);
    void compute_bias_gradient(int target);
    bool is_output_layer();
    LookupToHiddenLayer * deepcopy()const {
        LookupToHiddenLayer * newl = new LookupLayerTanh(vector<int>(), 0,0);
        newl->copy(*this);
        return newl;
    }

};

class LookupLayerRelu: public LookupToHiddenLayer{
    Vec mask, zero;
public:
    LookupLayerRelu(vector<int> n_input, int n_input_features, int n_units);
    void compute_activation(const vector<bool> &actions);
    void compute_bias_gradient(int target);
    bool is_output_layer();
    LookupToHiddenLayer * deepcopy()const{
        LookupToHiddenLayer * newl = new LookupLayerRelu(vector<int>(), 0,0);
        newl->copy(*this);
        return newl;
    }

};







/****
 * Layer by layer implementation of a feed-forward neural network.
 * Notations :
 *  h : state of layer units
 *  a : state before activation
 *  b : bias
 *  w : weights
 *  gb,gw : gradient for bias and weights
 *  dgb,dgw : history of gradients (for AdaGrad optimisation)
 */
class Layer : public AbstractLayer{
public :
    Mat w, gw, dgw;
    int n_input;
public :
    Layer(int n_input, int n_units);
    virtual ~Layer();

    virtual Layer* deepcopy()=0;

    virtual double fprop_std(const Vec & input, int target, const vector<bool> &actions);
    virtual void bprop_std(const Vec & input, int target);


    void rescale_weights(bool down);

    double l2_cost(double reg_lambda);
    void add_l2_gradient(double reg_lambda);
    void update_sgd(double step_size);
    void update_asgd(double step_size, double T);
    void update_adagrad(double step_size, double ada_epsilon);
    void average(double T);
    void clip_gradient(double threshold);

    const Mat & weights();
    const Mat & gweights();
    int get_n_input();

    void add_epsilon(int i, int j, double epsilon);

    void dump_parameters(const string dirname, int i);
    void load_parameters(const string dirname, int i);

};

class SoftmaxLayer : public Layer{
public:
    SoftmaxLayer(int n_input, int n_units);
    virtual ~SoftmaxLayer();
    double compute_cost(int gold);
    double fprop_std(const Vec &input, int target, const vector<bool> &actions);
    void bprop_std(const Vec &input, int target);
    bool is_output_layer();

    Layer* deepcopy(){
        Layer *newl = new SoftmaxLayer(0,0);
        newl->w = w;
        newl->gw = gw;
        newl->dgw = dgw;
        newl->n_input = n_input;
        newl->n_units = n_units;
        newl->pred = pred;
        newl->h = h;
        newl->a = a;
        newl->b = b;
        newl->gb = gb;
        newl->dgb = dgb;
        newl->dropout_mask = dropout_mask;
        newl->next_layer = NULL;
        return newl;
    }


};

class PerceptronLayer : public Layer{
public:
    PerceptronLayer(int n_input, int n_units);
    virtual ~PerceptronLayer();
    double compute_cost(int gold);
    double fprop_std(const Vec &input, int target, const vector<bool> &actions);
    void bprop_std(const Vec &input, int target);
    bool is_output_layer(){ return true;}
    Layer* deepcopy(){
        Layer *newl = new PerceptronLayer(0,0);
        newl->w = w;
        newl->gw = gw;
        newl->dgw = dgw;
        newl->n_input = n_input;
        newl->n_units = n_units;
        newl->pred = pred;
        newl->h = h;
        newl->a = a;
        newl->b = b;
        newl->gb = gb;
        newl->dgb = dgb;
        newl->dropout_mask = dropout_mask;
        newl->next_layer = NULL;
        return newl;
    }

};


class TanhLayer : public Layer{
public:
    TanhLayer(int n_input, int n_units);
    virtual ~TanhLayer();
    double fprop_std(const Vec & input, int target, const vector<bool> &actions);
    void bprop_std(const Vec & input, int target);
    bool is_output_layer(){ return false;}
    Layer* deepcopy(){
        Layer *newl = new TanhLayer(0,0);
        newl->w = w;
        newl->gw = gw;
        newl->dgw = dgw;
        newl->n_input = n_input;
        newl->n_units = n_units;
        newl->pred = pred;
        newl->h = h;
        newl->a = a;
        newl->b = b;
        newl->gb = gb;
        newl->dgb = dgb;
        newl->dropout_mask = dropout_mask;
        newl->next_layer = NULL;
        return newl;
    }

};

class CubeLayer : public Layer{
public:
    CubeLayer(int n_input, int n_units);
    virtual ~CubeLayer();
    double fprop_std(const Vec & input, int target, const vector<bool> &actions);
    void bprop_std(const Vec & input, int target);
    bool is_output_layer(){ return false;}
    Layer* deepcopy(){
        Layer *newl = new CubeLayer(0,0);
        newl->w = w;
        newl->gw = gw;
        newl->dgw = dgw;
        newl->n_input = n_input;
        newl->n_units = n_units;
        newl->pred = pred;
        newl->h = h;
        newl->a = a;
        newl->b = b;
        newl->gb = gb;
        newl->dgb = dgb;
        newl->dropout_mask = dropout_mask;
        newl->next_layer = NULL;
        return newl;
    }

};


class ReluLayer : public Layer{
public :
    Vec mask;
    Vec zero;
    ReluLayer(int n_input, int n_units);
    virtual ~ReluLayer();
    double fprop_std(const Vec & input, int target, const vector<bool> &actions);
    void bprop_std(const Vec & input, int target);
    bool is_output_layer(){ return false;}
    Layer* deepcopy(){
        Layer *newl = new ReluLayer(0,0);
        newl->w = w;
        newl->gw = gw;
        newl->dgw = dgw;
        newl->n_input = n_input;
        newl->n_units = n_units;
        newl->pred = pred;
        newl->h = h;
        newl->a = a;
        newl->b = b;
        newl->gb = gb;
        newl->dgb = dgb;
        newl->dropout_mask = dropout_mask;
        newl->next_layer = NULL;
        return newl;
    }

};

#endif // LAYERS_H
