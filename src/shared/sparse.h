
#ifndef SPARSE_H
#define SPARSE_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using namespace std;

/**
 * This module performs some sparse matrix algebra ops suitable for efficient parsing in a large margin framework (and NLP).
 * That is, matrix vector mult is made to be very efficient and vector/matrix addition and scalar mult reasonably efficient too.
 * The module uses a partial hash trick: in other words, feature hash collisions are not resolved and computations are partly approximative.
 * Although the module implements standard linear algebra operators, the inplace operators should be preferred for efficiency reasons.
 * Algebraic expressions should therefore remain simple, for full intensive general linear algebraic stuff use a full linalg library.
 */

//Constant controlling Feature X vector size (Kernel Size)
//primes < 10^6
static const size_t PRIME_MICRO = 499979;
static const size_t PRIME_MINI = 999983;
//10^7 closest prime
static const size_t PRIME_SMALL = 1999993;
//10^8 closest prime
static const size_t PRIME_LARGE = 19999999;

//Max Table size (primes, use with modulus op)
static const size_t KERNEL_SIZE  = PRIME_MINI;

//Type used for feature valuation
typedef unsigned int FVALUE ;


/**
 * This class stores an abstract feature.
 * An abstract feature is a tuple made of a templateID and its related values for each sensor
 */
class AbstractFeature{
    
public:
    
    AbstractFeature();
    AbstractFeature(AbstractFeature const &feat);
    AbstractFeature(FVALUE template_val);
    
    inline FVALUE get_value(unsigned int idx)const{return values[idx];}
    inline FVALUE get_template()const{return values[0];}
    
    inline void set_value(unsigned int idx,FVALUE val){values[idx] = val;}
    inline void set_template(FVALUE val){values[0] = val;}
    
    friend ostream& operator<<(ostream &os,AbstractFeature const &f);
    
    void refresh();
    
private:
    FVALUE values[4];//0 = template idx, 1 = val 1, 2 = val 2, 3 = val 3
    
};

/**
 * This function class returns a feature hashcode
 */
class FeatureHasher{
public:
    unsigned int operator()(AbstractFeature const &feature)const;
};

/**
 * This function class returns the feature index in the dense feature vector.
 */
class FeatureIndexer{
public:
    unsigned int operator()(AbstractFeature const &feature)const;
private:
    FeatureHasher h;
};


//Data structure for encoding scattered 2D values to be inserted in a matrix
struct Triplet{

  Triplet(unsigned int row_idx,unsigned int col_idx,float value);
  unsigned int row_idx;
  unsigned int col_idx;
  float value;

};

class SparseFloatVector{

 public:

  friend class SparseFloatMatrix;

  SparseFloatVector(unsigned int dense_n);
    
  SparseFloatVector(SparseFloatVector const &other);
  SparseFloatVector& operator=(SparseFloatVector const &other);

  //builds a sparse vector from a list of integer x coords; each coord is supposed to be 1-valued
  SparseFloatVector(vector<unsigned int> x_vec,unsigned int dense_n);
  SparseFloatVector(vector<unsigned int> x_vec,unsigned int dense_n, vector<unsigned int> &sparse_locs);

    
    
  float& coeffRef(unsigned int dense_idx); //provides a reference to a dense index 
  
  //Ops...
  const SparseFloatVector operator+(SparseFloatVector const &other) const;
  const SparseFloatVector operator-(SparseFloatVector const &other) const;
  const SparseFloatVector operator*(float scalar) const;
  const SparseFloatVector operator/(float scalar) const;

  SparseFloatVector& operator+=(SparseFloatVector const &other);
  SparseFloatVector& operator-=(SparseFloatVector const &other);
  SparseFloatVector& operator*=(float scalar);
  SparseFloatVector& operator/=(float scalar);

  float sqL2norm()const;  //squared L2 norm

  bool operator==(SparseFloatVector const &other)const; 
  bool operator!=(SparseFloatVector const &other)const; 

  void clear();
  void compress();//this compresses the matrix (may increase performance later on)

  void resize(unsigned int dense_n){size = dense_n;}
  size_t get_size()const{return size;};

  //I/O
  ostream& write(ostream &outstream);
  istream& read(istream &instream);
  friend ostream &operator<<( ostream &output, const SparseFloatVector &vec);

protected:
  
  SparseFloatVector(){};
  
  static vector<int> loc_register;//dense auxiliary vector used as an auxiliary struct for additions etc...
  void init_loc_register(unsigned int dense_size)const;

  float& add_entry(unsigned int dense_index,float value);
  float operator[](unsigned int dense_index) const;
  float& operator[](unsigned int dense_index);

  //adds up the content of this sparse vector into the dense vector (used for matrix ops)
  void accumulate(vector<float> &dense_vector,float coef = 1.0)const;

private:
  size_t size;
  //Two equally sized vectors.
  vector<unsigned int> locations; //stores dense index
  vector<float> values;           //stores corresponding value

};


/**
 * The SparseFloatMatrix class assumes an Y x X matrix where dims(Y) << dims(X). 
 */
class SparseFloatMatrix{

//To be viewed as an Y x X matrix where X are columns and Y rows.
//although implemented the other way round for efficiency..

public:

  SparseFloatMatrix(unsigned int x_dims,unsigned int y_dims);//empty matrix
  SparseFloatMatrix(unsigned int x_dims,unsigned int y_dims, vector<Triplet> const &triples);//init from triplets assumed to be 1 valued
  SparseFloatMatrix(unsigned int x_dims,unsigned int y_dims,string const &filename);//load from file
    
  SparseFloatMatrix(SparseFloatMatrix const &other);
  SparseFloatMatrix& operator= (SparseFloatMatrix const &other);
    
  //Dot product with a column vector (for prediction) ; result stored in dense vector yvec.
  void dot(SparseFloatVector const &xvec,vector<float> &yvec) const;
  void dot(vector<unsigned int> const &xvec,vector<float> &yvec,vector<unsigned int> &sparse_locs);
  void dot(vector<unsigned int> const &xvec,vector<float> &yvec) const; //assumed to be a vector of one valued hashed x coords
  float dot(vector<unsigned int> const &xvec,unsigned int row_index) const; //assumed to be a vector of one valued hashed x coords, and y a row index.
  
  //Mods and ops
  void add_rowY (SparseFloatVector const &x_delta,unsigned int ydense_index);
  void subs_rowY(SparseFloatVector const &x_delta,unsigned int ydense_index);
  void add_rowY (vector<unsigned int> const &x_idxes,unsigned int ydense_index);  //x_idxes is assumed to be a vector of 1 valued indexes
  void subs_rowY(vector<unsigned int> const &x_idxes,unsigned int ydense_index);  //x_idxes is assumed to be a vector of 1 valued indexes

  SparseFloatMatrix& operator+= (SparseFloatMatrix const &other);
  SparseFloatMatrix& operator-= (SparseFloatMatrix const &other);
  SparseFloatMatrix& operator*= (float scalar);
  SparseFloatMatrix& operator/= (float scalar);

  //Sets inplace this matrix to M <- M - lambda M 
  float sqL2norm()const;  //(matrix viewed as a vector-> elementwise squared L2 norm)


  bool operator==( SparseFloatMatrix const &other)const;
  bool operator!=( SparseFloatMatrix const &other)const;

  void clear();   //clears the content of this matrix to a ready to use empty matrix
  void compress();//time consuming op that compresses the matrix for better efficiency (may increase performance later on)

  //Observers
  inline float col_density()const{return xlocations.size() / x_dims; }
  void row_means(vector<float> &means_vec);
  void row_variances(vector<float> &means_vec,bool stddev=false);

  //I/O
  void load(string const &filename,size_t y_size);
  void save(string const &filename);
    
  //to be used with care (reasonable size matrices)
  friend ostream &operator<<( ostream &output, const SparseFloatMatrix &mat);

protected:
  unsigned int x_dims,y_dims;
  vector<unsigned int> xlocations;
  vector<SparseFloatVector> values;

};

///////////////////////////////////////////////////
//LEARNERS
// This section contains code that wraps several large margin learning algorithms and their specific params.
// It allows them to be used in a parser for efficient training.

class AbstractLearner{
public:
    
    virtual ~AbstractLearner();
    
    virtual float get_loss(float hyp_weight,float ref_weight,float margin = 1.0)const = 0;
    
    //PASSIVE update
    virtual void update_model(float T,float T0,float batch_loss,size_t batch_size) = 0;
    
    //ACTIVE update
    virtual void update_model(SparseFloatMatrix &delta,float T,float T0,float batch_loss,size_t batch_size) = 0;
    
    //DOT with X
    virtual void dot(vector<unsigned int> &xvec,vector<float> &yscores) = 0;
    virtual void avg_dot(vector<unsigned int> &xvec,vector<float> &yscores) = 0;

    //This allocates a weight matrix to be used by a parser (typically returns the averaged weights)
    virtual SparseFloatMatrix* get_final_model() = 0 ;
    
};

class PegasosLearner{
public:
    
    PegasosLearner(size_t ydims, float lambda=0.001);
    ~PegasosLearner();
    
    float get_loss(float hyp_weight,float ref_weight,float margin = 1.0) const{
        return std::max<float>(0.0, (hyp_weight)+ margin - ref_weight);
    }
    //PASSIVE update
    void update_model(float T,float T0,float batch_loss,size_t batch_size){
        (*weights) *= (1-(1/T));
        float delta_t = T-T0;
        if( delta_t == 1){ //Averaged SGD
            *averaged_weights = *weights;
        }else if(delta_t > 1){
            scale_factor *= delta_t / (delta_t+1);
            (*averaged_weights) += (*weights);
        }
    }
    //ACTIVE update
    void update_model(SparseFloatMatrix &delta,float T,float T0,float batch_loss,size_t batch_size);
    
    //DOT with X
    void dot(vector<unsigned int> &xvec,vector<float> &yscores);
    void avg_dot(vector<unsigned int> &xvec,vector<float> &yscores);
    
    //This allocates a weight matrix to be used by a parser (typically returns the averaged weights)
    SparseFloatMatrix* get_final_model();

private:
    SparseFloatMatrix * weights;
    SparseFloatMatrix * averaged_weights;
    float scale_factor;
    float lambda;
    float alpha;
    float beta;
};






#endif
