
#ifndef NEURAL_NET_H
#define NEURAL_NET_H


#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <string>
#include <Eigen/Dense>
#include <sys/stat.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <tuple>
#include <boost/functional/hash.hpp>
#include <boost/algorithm/string.hpp>
#include <assert.h>


#include "layers.h"
#include "globals.h"
#define DBG(x) std::cerr << x << std::endl;

using std::vector;
using std::string;
using std::cerr;
using std::endl;
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;

class Classifier {
protected:
    bool test_mode;
public :
    virtual ~Classifier(){}

    virtual Classifier * deepcopy(){
        throw "ERROR not implemented yet";
        return NULL;
    }

    virtual void set_hyperparameters(const NeuralNetParams &params) = 0;
    virtual void set_learning_rate(double lr)  = 0;
    virtual double get_learning_rate()  = 0;
    virtual void rescale_weights(bool down)  = 0;
    virtual void scale_learning_rate(double d)  = 0;

    virtual void summary(ostream &os)  = 0;
    virtual double fprop(const vector<int> & input, int target, const vector<bool> &actions)  = 0;
    virtual void scores(vector<float> &output)  = 0;
    virtual void scores(const vector<int> &input, vector<float> &output, const vector<bool> &actions) = 0;
    virtual double scores(const vector<int> &input, vector<float> &output, int target, const vector<bool> &actions)= 0;

    virtual void set_pred(int newp)  = 0;
    virtual void bprop_std(const vector<int> & input, int target)  = 0;
    virtual void update()  = 0;

    virtual void save(const string dirname)  = 0;
    virtual void load(const string dirname)  = 0;

    virtual void load_pretrained_embeddings(const string &lu_filename)  = 0;
    virtual void allocate_embedding(int i)  = 0;

    virtual void set_test_mode(bool t);
    virtual int get_concatenated_layers_size();
    virtual void get_concatenated_layers(Vec &perceptron_input);
    virtual void average();
};

class LogisticRegression : public Classifier{
    /////
    /// TODO : regularization / adagrad
    ///

    vector<Vec> weights;
    vector<Vec> avg_weights;
    vector<int> features;

    unordered_map<pair<int,int>, int, boost::hash<pair<int,int>>> map;
    Vec output;
    Vec bias;
    Vec gbias;
    Vec avg_bias;
    double learning_rate;
    int n_units;
    int n_features;
    int prediction;
    double n_updates;
    bool asgd;

public:
    void set_hyperparameters(const NeuralNetParams &params);
    void set_learning_rate(double lr);
    double get_learning_rate();
    void rescale_weights(bool down);
    void scale_learning_rate(double d);
    void summary(ostream &os);
    void set_pred(int newp);
    void find_features(const vector<int> & input);
    double fprop(const vector<int> & input, int target, const vector<bool> &actions);
    void scores(vector<float> &score);
    void scores(const vector<int> &input, vector<float> &output, const vector<bool> &actions);
    double scores(const vector<int> &input, vector<float> &output, int target, const vector<bool> &actions);
    void bprop_std(const vector<int> & input, int target);
    void update();
    void average();
    void save(const string dirname);
    void load(const string dirname);
    void load_pretrained_embeddings(const string &lu_filename);
    void allocate_embedding(int i);
};

class NeuralNetwork : public Classifier{
    int n_input_features, n_output;
    vector<int> n_input;
    vector<int> n_hidden;
    double learning_rate, decrease_constant, reg_lambda, ada_epsilon, clipping_threshold;
    bool ada_grad, dropout, regularisation, gradient_clipping, asgd;         // replace dropout by integer to control which layer has it ?
    int n_updates;

    LookupToHiddenLayer *input_layer;
    vector<Layer*> layers;

    int loss_function;
    int hidden_activation;

    bool convolution;

public :
    NeuralNetwork(int n_input_features, vector<int> n_input, vector<int> n_hidden, int n_output);
    NeuralNetwork();
   ~NeuralNetwork();

    Classifier * deepcopy() {
        return new NeuralNetwork(*this);
    }
    NeuralNetwork(const NeuralNetwork &other);
    NeuralNetwork & operator=(const NeuralNetwork &other);


    void summary(ostream &os);
    void initialize();

    LookupToHiddenLayer* get_new_lookup_layer(vector<int> n_input, int n_input_features, int n_units, bool output);
    Layer* get_new_output_layer(int input, int output);
    Layer* get_new_hidden_layer(int input, int units);

    void set_hyperparameters(const NeuralNetParams &params);
    void set_learning_rate(double lr);
    void set_random_init(bool ri);

    void scale_learning_rate(double d);
    double get_learning_rate();
    void allocate_embedding(int i);

    int predict();
    void set_pred(int newp);
    void rescale_weights(bool down);

    void train_one(const vector<int> &input, int target, const vector<bool> &actions);
    void scores(const vector<int> &input, vector<float> &output, const vector<bool> &actions);
    void scores(vector<float> &output);
    double scores(const vector<int> &input, vector<float> &output, int target, const vector<bool> &actions);
    double fprop(const vector<int> & input, int target, const vector<bool> &actions);
    void bprop_std(const vector<int> & input, int target);
    void update();
    void clip_gradient(double threshold);

    void gradient_check_layer(const vector<int> &input, int target, Layer *l, Mat & egw, Vec & egb, double epsilon, const vector<bool> &actions);
    void gradient_check_lookup_layer(const vector<int> &input, int target, LookupToHiddenLayer *l, double epsilon, const vector<bool> &actions);
    void gradient_checking(const vector<int> &input, int target, const vector<bool> &actions);


    void average();
    void set_test_mode(bool t);
    int get_concatenated_layers_size();
    void get_concatenated_layers(Vec &perceptron_input);


    // INPUT / OUTPUT
    void save(const string dirname);
    void load(const string dirname);
    void dump_hyperparameters(const string filename) const;
    void load_hyperparameters(const string filename);
    void dump_parameters(const string dirname) const;
    void load_parameters(const string dirname);
    void load_pretrained_embeddings(const string &lu_filename);


    static void run_gradient_checking();

protected:
    static const string NC;
    static const string NH;
    static const string NI;
    static const string NE;
    static const string HL;
    static const string LOSS;
    static const string ACTIVATION;
    static const string CONV_NET;
};





/*
 * Encapsulates a perceptron parameters
 * No bias
 */
class NeuralPerceptron {
    Mat weights; // also : try a row major matrix to speed up row manageemnt
    Vec bias;
    int x_dims; // output_size
    int y_dims; // input_size
public :
    NeuralPerceptron();
    NeuralPerceptron(int x, int y);
    NeuralPerceptron(NeuralPerceptron const & other);
    NeuralPerceptron& operator=(NeuralPerceptron const & other);
    ~NeuralPerceptron();
    inline void dot(const Vec &input, Vec &output){     output = weights * input;}///+ bias;
    inline void add_row(const Vec &vec, int row){       weights.row(row) += vec;}
    inline void sub_row(const Vec &vec, int row){       weights.row(row) -= vec;}
    void display_row(int row){
      cout << weights.row(row);
    }
    inline void add_bias(double delta, int idx){        bias[idx] += delta;}

    NeuralPerceptron& operator+= (NeuralPerceptron const &other);
    NeuralPerceptron& operator-= (NeuralPerceptron const &other);
    NeuralPerceptron& operator*= (float scalar);
    NeuralPerceptron& operator/= (float scalar);
    float sqL2norm()const;
    void save(string const &model_path, int iteration);
    void load(string const &model_path);
};


#endif // NEURAL_NET_H
