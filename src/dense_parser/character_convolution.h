#ifndef CHARACTER_CONVOLUTION_H
#define CHARACTER_CONVOLUTION_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <cmath>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>


#include "globals.h"
#include "str_utils.h"

#define DBG(x) std::cerr << x << std::endl;


/// TODO
/// padding adapté à la taille du filtre de convolution
/// utiliser plusieurs filtres de convolution
/// dumper les paramètres
/// récupérer un bon paramétrage sur alpaga
///



using std::vector;
using std::unordered_map;
using std::unordered_set;
using std::ifstream;
using std::ofstream;
using std::string;
using std::cerr;
using std::cout;
using std::endl;

typedef Eigen::MatrixXd Mat;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatR;
typedef Eigen::VectorXd Vec;

typedef wchar_t CHAR;

const string SEPARATOR = " ";
const double CHAR_INIT = 100;
const CHAR INITIAL_PADDING = 0x000022a2;
const CHAR FINAL_PADDING =  0x000022a3;

enum {INITIAL_ID, FINAL_ID};

template <class M>
void load_matrix(std::string filename, M &res){

    std::ifstream input(filename);
    std::string line;
    double d;
    std::vector<double> v;
    int rows = 0;
    while(getline(input, line)){
        rows++;
        std::stringstream input_line(line);
        while (!input_line.eof()){
            input_line >> d;
            v.push_back(d);
        }
    }
    input.close();
    int cols = v.size() / rows;
    res = M::Zero(rows, cols);

    for (int i=0; i< rows; i++){
        for (int j=0; j< cols; j++){
            res(i,j) = v[i*cols + j];
        }
    }
}


class CharacterConvolution{
    /*
     *
     *
     *
     * AJOUTER CONSTRUCTEURS PAR COPIES + ASSIGNEMENT OPERATORS
     *
     *
     *
     *
     */

    int charvocsize;                                    // number of chars
    vector<CHAR> character_voc;                         // character vocabulary
    unordered_map<CHAR, int> charmap;                   // char -> idx   (why string ? avoid transformation to UTF32, chars with > 1 bytes are treated as a string
    unordered_map<int, vector<int>> chars_to_ints;      // word -> vector of char idxes (with beginning and end padding chars)
    unordered_set<int> active_chars;                    // set of active char embeddings

    int char_dimension;                                 // dimension of char embeddings
    int dimension;                                      // dimension of word embeddings
    int filter_size;                                    // size of filter (TODO : possibility of using several filters
    int n_features;
    int n_units;

    MatR kernel;
    MatR gkernel;
    MatR dgkernel;

    vector<Vec> char_embeddings;                        // target char embeddings
    vector<Vec> gchar_embeddings;                       // gradient for target char embeddings
    vector<Vec> dgchar_embeddings;                      //

    vector<Vec> chars_concats;                          // stores the concatenation of characters embeddings
    vector<Mat> feature_maps;                           // stores the feature maps
    vector<Vec> max_pool;                               // stores the char based embedding for a word
    vector<Vec> gmax_pool;                              // stores the gradient of char based embedding
    vector<vector<int>> imax;                           // stores the argmax for each feature map row
    vector<Mat> weights;                                // weights to compute hidden layer
    vector<Mat> gweights;                               // gradient of weights
    vector<Mat> dgweights;
    vector<Vec> buffer;                                 // partial computations of hidden layer


public:
    CharacterConvolution(){
        character_voc.push_back(INITIAL_PADDING);     // gotta find better beginning / end of word character
        character_voc.push_back(FINAL_PADDING);
        charmap[INITIAL_PADDING] = INITIAL_ID;
        charmap[FINAL_PADDING] = FINAL_ID;
    }
    void display_summary(ostream &os){
        os << "Convolution over characters layer : " << endl;
        os << "- dimension of character embeddings : "  << char_dimension << endl;
        os << "- dimension of character-based word embeddings : "  << dimension << endl;
        os << "- size of convolutional filter : " << filter_size<< endl;
        os << "- number of convolved features : " << n_features << endl;
        os << "- character vocabulary size : " << character_voc.size() << endl;
        os << "- character vocabulary : ";
        for (int i = 0; i < character_voc.size(); i++){
            PSTRING w;
            w += character_voc[i];
            os << str::encode(w);
        }
        os << endl;
    }

    void set_dimension(int dimension){              this->dimension = dimension;            }
    void set_char_dimension(int char_dimension){    this->char_dimension = char_dimension;  }
    void set_filter_size(int filter_size){          this->filter_size = filter_size;        }
    void set_n_features(int n_features){            this->n_features = n_features;          }
    void set_n_units(int n_units){                  this->n_units = n_units;                }


    void update_char_encoder(){
        charvocsize = 2;

        vector<PSTRING> res;
        vector<int> idx;
        IntegerEncoder::get()->get_token_vocabulary(res, idx);

        for (int i = 0; i < res.size(); i++){
            vector<int> ints;
            ints.push_back(INITIAL_ID);
            for (CHAR &c :res[i]){
                if (charmap.find(c) == charmap.end()){
                    charmap[c] = charvocsize ++;
                    character_voc.push_back(c);
                }
                ints.push_back(charmap[c]);
            }
            ints.push_back(FINAL_ID);
            chars_to_ints[idx[i]] = ints;
        }
    }

    void initialize(){
        update_char_encoder();
        kernel      = MatR::Random(dimension, char_dimension * filter_size) * sqrt(6.0 / (double)(dimension + char_dimension * filter_size ));
        gkernel     = MatR::Zero(dimension, char_dimension * filter_size);
        dgkernel     = MatR::Zero(dimension, char_dimension * filter_size);
        char_embeddings     = vector<Vec>(charvocsize);
        gchar_embeddings    = vector<Vec>(charvocsize);
        dgchar_embeddings   = vector<Vec>(charvocsize);
        for (int i = 0; i < charvocsize; i++){
            char_embeddings[i]   = Vec::Random(char_dimension) / CHAR_INIT;
            gchar_embeddings[i]  = Vec::Zero(char_dimension);
            dgchar_embeddings[i] = Vec::Zero(char_dimension);
        }
        chars_concats   = vector<Vec>(n_features);
        feature_maps    = vector<Mat>(n_features);
        max_pool        = vector<Vec>(n_features);
        gmax_pool       = vector<Vec>(n_features);
        imax            = vector<vector<int>>(n_features);
        weights         = vector<Mat>(n_features);
        gweights        = vector<Mat>(n_features);
        dgweights       = vector<Mat>(n_features);
        buffer          = vector<Vec>(n_features);
        for (int i = 0; i < n_features; i++){
            max_pool[i] = Vec::Zero(dimension);
            gmax_pool[i]= Vec::Zero(dimension);
            imax[i]     = vector<int>(dimension);
            weights[i]  = Mat::Random(n_units, dimension) * sqrt(6 / (dimension + n_units));
            gweights[i] = Mat::Zero(n_units, dimension);
            dgweights[i]= Mat::Zero(n_units, dimension);
            buffer[i]   = Vec::Zero(dimension);
        }
    }

    void get_int_sequence(int tok_code, vector<int> &res){
        auto it = chars_to_ints.find(tok_code);
        if (it != chars_to_ints.end()){
            res = it->second;
            return;
        }
        if ( IntegerEncoder::get()->decode(tok_code) == L"$UNDEF_LEFT$" || IntegerEncoder::get()->decode(tok_code) == L"$UNDEF_RIGHT$"){
            chars_to_ints[tok_code] = vector<int>(0);
            res = chars_to_ints[tok_code];
            return;
        }
        PSTRING ps = IntegerEncoder::get()->decode(tok_code);
        res.clear();
        res.push_back(INITIAL_ID);
        for (const CHAR &c : ps){
            if (charmap.find(c) != charmap.end()){
                res.push_back(charmap[c]);
            }else{
                cerr << "ERROR, unknown char " << c << endl;
                exit(1);
            }
        }
        res.push_back(FINAL_ID);
    }

    void fprop(vector<int> &input, Vec &accumulator){
        for (int i = 0; i < input.size(); i++){
            vector<int> ints;
            get_int_sequence(input[i], ints);
            if ((int)(ints.size()) - filter_size + 1 > 0){
                for (const int &c : ints){
                    active_chars.insert(c);
                }
                chars_concats[i] = Vec::Zero(ints.size() * char_dimension);
                for (int k = 0; k < ints.size(); k++){
                    chars_concats[i].segment(k * char_dimension, char_dimension) = char_embeddings[ints[k]];
                }
                feature_maps[i] = Mat::Zero(dimension, ints.size() - filter_size + 1);
                for (int k = 0; k < ints.size() - filter_size + 1; k++){
                    feature_maps[i].col(k) = kernel * chars_concats[i].segment(k * char_dimension, char_dimension * filter_size);
                }
                for (int row = 0; row < dimension; row++){
                    max_pool[i][row] = feature_maps[i].row(row).maxCoeff(&(imax[i][row]));
                }
                buffer[i] = weights[i] * max_pool[i];
            }else{
                max_pool[i].fill(0.0);
                buffer[i].fill(0.0);
            }
        }
        for (int i = 0; i < input.size(); i++){
            accumulator += buffer[i];
        }
    }

    void bprop(vector<int> &input, const Vec &output_grad){
        for (int i = 0; i < n_features; i++){
            vector<int> ints;
            get_int_sequence(input[i], ints);
            if ((int)(ints.size()) - filter_size + 1 > 0){
                gweights[i] = output_grad * max_pool[i].transpose();
                gmax_pool[i] = weights[i].transpose() * output_grad;
                for (int row = 0; row < dimension; row++){
                    int max_idx = imax[i][row];
                    gkernel.row(row) = gmax_pool[i][row] * chars_concats[i].segment(max_idx * char_dimension, char_dimension * filter_size);        // Beware : concurrent access possible here
                    for (int f = 0; f < filter_size; f++){
                        gchar_embeddings[ints[max_idx+f]] += gmax_pool[i][row] * kernel.row(row).segment(f * char_dimension, char_dimension);
                    }
                }
            }
        }
    }

    void update(double step_size){
        kernel -= step_size * gkernel;
        for (int i = 0; i < n_features; i++){
            weights[i] -= step_size * gweights[i];
        }
        for (const int &c : active_chars){
            char_embeddings[c] -= step_size * gchar_embeddings[c];
            gchar_embeddings[c].fill(0.0);
        }
        active_chars.clear();
    }
    void update_adagrad(double step_size, double ada_epsilon){
        dgkernel += gkernel.array().abs2().matrix();
        kernel -= step_size * (gkernel.array() / (ada_epsilon + dgkernel.array().sqrt())).matrix();
        for (int i = 0; i < n_features; i++){
            dgweights[i] += gweights[i].array().abs2().matrix();
            weights[i] -= step_size * (gweights[i].array() / (ada_epsilon + dgweights[i].array().sqrt())).matrix();
        }
        for (const int &c : active_chars){
            dgchar_embeddings[c] += gchar_embeddings[c].array().abs2().matrix();
            char_embeddings[c] -= step_size * (gchar_embeddings[c].array() / (ada_epsilon + dgchar_embeddings[c].array().sqrt())).matrix();
            gchar_embeddings[c].fill(0.0);
        }
        active_chars.clear();
    }
    void update_asgd(double step_size, double T){
        kernel -= step_size * gkernel;
        dgkernel -= T * step_size * gkernel;
        for (int i = 0; i < n_features; i++){
            weights[i] -= step_size * gweights[i];
            dgweights[i] -= T * step_size * gweights[i];
        }
        for (const int &c : active_chars){
            char_embeddings[c] -= step_size * gchar_embeddings[c];
            dgchar_embeddings[c] -= T * step_size * gchar_embeddings[c];
            gchar_embeddings[c].fill(0.0);
        }
        active_chars.clear();
    }
    void average(double T){
        kernel -= dgkernel / T;
        for (int i = 0; i < n_features; i++){
            weights[i] -= dgweights[i] / T;
        }
        for (int i = 0; i < char_embeddings.size(); i++){
            char_embeddings[i] -= dgchar_embeddings[i] / T;
        }
    }





    void save(const string dirname){
        save_constants(dirname + "/convolution_parameters.conf");
        save_parameters(dirname);
    }
    void load(const string dirname){
        load_constants(dirname + "/convolution_parameters.conf");
        initialize();
        load_parameters(dirname);
    }

    void save_constants(const string filename){
        ofstream out(filename);
        out << CHAR_EMB_DIM << " = " << char_dimension << endl;
        out << WORD_EMB_DIM << " = " << dimension << endl;
        out << FILTER_SIZE << " = " << filter_size << endl;
        out << NUM_FEATURES << " = " << n_features << endl;
        out << NUM_UNITS << " = " << n_units << endl;
        out << VOC_SIZE << " = " << charvocsize << endl;
        out.close();
    }
    void save_parameters(const string dirname){
        ofstream outfile;
        outfile.open(dirname + "/convolution_charvoc");
        for (int i = 0; i < character_voc.size(); i++){
            PSTRING w;
            w += character_voc[i];
            outfile << str::encode(w) << endl;
        }
        outfile.close();
        outfile.open(dirname + "/convolution_char_embeddings");
        for (int i = 0; i < char_embeddings.size(); i++){
            outfile << char_embeddings[i].transpose() << endl;
        }
        outfile.close();

        outfile.open(dirname + "/convolution_w2v_char_embeddings");
        outfile << char_embeddings.size() << " " << char_dimension << endl;
        for (int i = 0; i < char_embeddings.size(); i++){
            PSTRING w;
            w += character_voc[i];
            outfile << str::encode(w) << " " << char_embeddings[i].transpose() << endl;
        }
        outfile.close();

        for (int i = 0; i < n_features; i++){
            outfile.open(dirname + "/convolution_weights" + std::to_string(i));
            outfile << weights[i] << endl;
            outfile.close();
        }
    }
    void load_parameters(const string dirname){
        Mat tmp;
        load_matrix<Mat>(dirname + "/convolution_char_embeddings", tmp);
        assert(tmp.rows() == charvocsize);
        assert(tmp.cols() == char_dimension);
        for (int i = 0; i < charvocsize; i++){
            char_embeddings[i] = tmp.row(i);
        }
        for (int i = 0; i < n_features; i++){
            load_matrix<Mat>(dirname + "/convolution_weights" + std::to_string(i),  weights[i]);
            assert(weights[i].cols() == n_units);
            assert(weights[i].rows() == dimension);
        }
    }

    void load_constants(const string filename){
        boost::property_tree::ptree pt;
        boost::property_tree::ini_parser::read_ini(filename, pt);

        char_dimension  = pt.get<int>(CHAR_EMB_DIM);
        dimension       = pt.get<int>(WORD_EMB_DIM);
        filter_size     = pt.get<int>(FILTER_SIZE);
        n_features      = pt.get<int>(NUM_FEATURES);
        n_units         = pt.get<int>(NUM_UNITS);
        charvocsize     = pt.get<int>(VOC_SIZE);
    }

private :
    static const string CHAR_EMB_DIM;
    static const string WORD_EMB_DIM;
    static const string FILTER_SIZE;
    static const string NUM_FEATURES;
    static const string NUM_UNITS;
    static const string VOC_SIZE;

};

#endif // CHARACTER_CONVOLUTION_H
