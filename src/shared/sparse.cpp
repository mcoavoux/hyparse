#include "sparse.h"
#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>

// 32 bit hashing (Jenkins)

#define __rot__(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

#define __mix__(a,b,c) \
{ \
a -= c;  a ^= __rot__(c, 4);  c += b; \
b -= a;  b ^= __rot__(a, 6);  a += c; \
c -= b;  c ^= __rot__(b, 8);  b += a; \
a -= c;  a ^= __rot__(c,16);  c += b; \
b -= a;  b ^= __rot__(a,19);  a += c; \
c -= b;  c ^= __rot__(b, 4);  b += a; \
}

#define __final__(a,b,c) \
{ \
c ^= b; c -= __rot__(b,14); \
a ^= c; a -= __rot__(c,11); \
b ^= a; b -= __rot__(a,25); \
c ^= b; c -= __rot__(b,16); \
a ^= c; a -= __rot__(c,4);  \
b ^= a; b -= __rot__(a,14); \
c ^= b; c -= __rot__(b,24); \
}



AbstractFeature::AbstractFeature(){
    std::fill(values,values+4,0);
}

AbstractFeature::AbstractFeature(AbstractFeature const &other){
    std::copy(other.values,other.values+4,this->values);
}

AbstractFeature::AbstractFeature(FVALUE template_val){
    std::fill(values,values+4,0);
    set_template(template_val);
}

void AbstractFeature::refresh(){
    std::fill(values+1,values+4,0);
}


ostream& operator<<(ostream &os,AbstractFeature const &f){
    for(int i = 0; i < 4;++i){
        os << f.values[i]<<" ";
    }
    return os;
}


unsigned int FeatureHasher::operator()(AbstractFeature const &feature)const{
    
    FVALUE a = feature.get_template();
    FVALUE b = feature.get_value(1);
    FVALUE c = feature.get_value(2);
    
    __mix__(a,b,c);
    
    a += feature.get_value(3);
    
    __final__(a,b,c);
    
    return c;
}

unsigned int FeatureIndexer::operator()(AbstractFeature const &feature)const{
    return h(feature) % KERNEL_SIZE;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

Triplet::Triplet(unsigned int row_idx,unsigned int col_idx,float value){
  this->row_idx = row_idx;
  this->col_idx = col_idx;
  this->value   = value;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////


vector<int> SparseFloatVector::loc_register = vector<int>();

SparseFloatVector::SparseFloatVector(unsigned int dense_n){
  this->size = dense_n;
}

void SparseFloatVector::clear(){
  locations.clear();
  values.clear();
}


SparseFloatVector::SparseFloatVector(vector<unsigned int> x_vec,unsigned int dense_n){
  this->size = dense_n;
  for(int i = 0; i < x_vec.size();++i){
    coeffRef(x_vec[i]) += 1.0;
  }
}

SparseFloatVector::SparseFloatVector(vector<unsigned int> x_vec,unsigned int dense_n, vector<unsigned int> &sparse_locs){
    this->size = dense_n;
    for(int i = 0; i < sparse_locs.size();++i){
        coeffRef(x_vec[sparse_locs[i]]) += 1.0;
    }
}

SparseFloatVector::SparseFloatVector(SparseFloatVector const &other){
  this->size = other.size;
  assert(other.locations.size()==other.values.size());
  locations.resize(other.locations.size());
  values.resize(other.values.size());
  std::copy(other.locations.begin(),other.locations.end(),locations.begin());
  std::copy(other.values.begin(),other.values.end(),values.begin());
}

SparseFloatVector& SparseFloatVector::operator=(SparseFloatVector const &other){
    
    this->size = other.size;
    assert(other.locations.size()==other.values.size());
    locations.resize(other.locations.size());
    values.resize(other.values.size());
    std::copy(other.locations.begin(),other.locations.end(),locations.begin());
    std::copy(other.values.begin(),other.values.end(),values.begin());
    return *this;
}


float& SparseFloatVector::add_entry(unsigned int dense_index,float value){
  locations.push_back(dense_index);
  values.push_back(value);
  return values.back();
}

float& SparseFloatVector::coeffRef(unsigned int dense_idx){
  for(int i = 0; i < locations.size();++i){
    if(locations[i] == dense_idx){return values[i];}
  }
  return add_entry(dense_idx,0);
}

void SparseFloatVector::compress(){//this compresses the matrix (may increase performance later on)
  vector<unsigned int> tmp_loc;
  vector<float> tmp_val;
  for(int i = 0; i < locations.size();++i){
    if (values[i] != 0){
      tmp_loc.push_back(locations[i]);
      tmp_val.push_back(values[i]);
    }
  }
  locations = tmp_loc;
  values = tmp_val;
}

float SparseFloatVector::operator[](unsigned int dense_index) const{
 for(int i = 0; i < locations.size();++i){
    if(locations[i] == dense_index){return values[i];}
  }
  return 0;
}

float& SparseFloatVector::operator[](unsigned int dense_index){
  for(int i = 0; i < locations.size();++i){
    if(locations[i] == dense_index){return values[i];}
  }
  return add_entry(dense_index,0);
}

void SparseFloatVector::init_loc_register(unsigned int dense_size)const{
  if (loc_register.size() != dense_size){
    loc_register.clear();
    loc_register.resize(dense_size,-1);
  }
}

SparseFloatVector& SparseFloatVector::operator*=(float scalar){
  for(int i = 0; i < values.size();++i){
    values[i] *= scalar;
  }
  return *this;
}

SparseFloatVector& SparseFloatVector::operator/=(float scalar){
  for(int i = 0; i < values.size();++i){
    values[i] /= scalar;
  }
  return *this;
}

float SparseFloatVector::sqL2norm()const{
  float dotv = 0;
  for(int i = 0; i < values.size();++i){
    dotv += values[i] * values[i];
  }
  return dotv;
}

const SparseFloatVector SparseFloatVector::operator+(SparseFloatVector const &other) const{
  SparseFloatVector tmp(*this);
  tmp += other;
  return tmp;
}

const SparseFloatVector SparseFloatVector::operator-(SparseFloatVector const &other) const{
  SparseFloatVector tmp(*this);
  tmp -= other;
  return tmp;
}

const SparseFloatVector SparseFloatVector::operator*(float scalar) const{
  SparseFloatVector tmp(*this);
  tmp *= scalar;
  return tmp;
}

const SparseFloatVector SparseFloatVector::operator/(float scalar) const{
  SparseFloatVector tmp(*this);
  tmp /= scalar;
  return tmp;
}

ostream& SparseFloatVector::write(ostream &outstream){
    
    unsigned int sparse_size = locations.size();
    outstream.write((char*)&sparse_size,sizeof(unsigned int));
    
    for(unsigned int i = 0; i < sparse_size;++i){
        outstream.write((char*)&locations[i],sizeof(unsigned int));
    }
    for(unsigned int i = 0; i < sparse_size;++i){
        outstream.write((char*)&values[i],sizeof(float));
    }
    return outstream;
}

//binary read
istream& SparseFloatVector::read(istream &instream){
    unsigned int sparse_size;
    instream.read( (char*) &sparse_size,sizeof(unsigned int));
    locations.resize(sparse_size,0);
    values.resize(sparse_size,0.0);
    for(unsigned int i = 0; i < sparse_size;++i){
        instream.read((char*)&locations[i],sizeof(unsigned int));
    }
    for(unsigned int i = 0; i < sparse_size;++i){
        instream.read((char*)&values[i],sizeof(float));
    }
    return instream;
}

ostream& operator<<( ostream &os, const SparseFloatVector &vec){
  if (vec.locations.size()>0){os << vec.locations[0] << ":"<< vec.values[0] ;}
  for (int i = 1; i < vec.locations.size();++i){
    os << " " << vec.locations[i] << ":"<< vec.values[i];
  }
  return os;
}



SparseFloatVector& SparseFloatVector::operator+=(SparseFloatVector const &other){
  assert(this->get_size() == other.get_size());
  init_loc_register(this->get_size());
  //y values in loc register
  for(int i = 0; i < other.locations.size();++i){
    loc_register[other.locations[i]] = i;
  }
  //x values in loc register and update x
  for(int i = 0; i < this->locations.size();++i){
    if( loc_register[this->locations[i]] != -1 ){
      this->values[i] += other.values[loc_register[this->locations[i]]];
      loc_register[this->locations[i]] = -1;
    } 
  }
  //remaining y values
  for(int i = 0; i < other.locations.size();++i){
    if (loc_register[other.locations[i]] != -1){
      this->add_entry(other.locations[i],other.values[i]);
      loc_register[other.locations[i]] = -1;
    }
  }
  assert(std::all_of(loc_register.begin(),loc_register.end(),[](int i){return i == -1;}));
  return *this;
}

SparseFloatVector& SparseFloatVector::operator-=(SparseFloatVector const &other){
  assert(this->get_size() == other.get_size());
  init_loc_register(this->get_size());

  //y values in loc register
  for(int i = 0; i < other.locations.size();++i){
    loc_register[other.locations[i]] = i;
  }
  //x values in loc register and update x
  for(int i = 0; i < this->locations.size();++i){
    if (loc_register[this->locations[i]] != -1 ){
      this->values[i] -= other.values[loc_register[this->locations[i]]];
      loc_register[this->locations[i]] = -1;
    } 
  }
  //remaining y values
  for(int i = 0; i < other.locations.size();++i){
    if (loc_register[other.locations[i]] != -1){
      this->add_entry(other.locations[i],-other.values[i]);
      loc_register[other.locations[i]] = -1;
    }
  }
  assert(std::all_of(loc_register.begin(),loc_register.end(),[](int i){return i == -1;}));
  return *this;
}

//this implementation is motivated by spurious zeros potentially remaining in the sparse vectors
bool SparseFloatVector::operator==(SparseFloatVector const &other)const{
  assert(this->get_size() == other.get_size());
  init_loc_register(this->get_size());
  
  bool res = true;
  //y values in loc register
  for(int i = 0; i < other.locations.size();++i){
    loc_register[other.locations[i]] = i;
  }  
  //x values in loc register
  for(int i = 0; i < this->locations.size();++i){
    if (loc_register[this->locations[i]] == -1 && this->values[i] != 0){
      loc_register[this->locations[i]] = -1;
      res = false;
    }
    if (loc_register[this->locations[i]] != -1){
      if(this->values[i] != other.values[loc_register[this->locations[i]]]){res = false;} 
      loc_register[this->locations[i]] = -1;
    }
  }
  //remaining y values
  for(int i = 0; i < other.locations.size();++i){
    if (loc_register[other.locations[i]] != -1){
      res = false;
      loc_register[other.locations[i]] = -1;
    }
  }  
  assert(std::all_of(loc_register.begin(),loc_register.end(),[](int i){return i == -1;}));
  return res;
}

bool SparseFloatVector::operator!=(SparseFloatVector const &other)const{
  return ! ( *this == other );
}

void SparseFloatVector::accumulate(vector<float> &dense_vector,float coef)const{
  assert(this->get_size() == dense_vector.size());
  for(int i = 0; i < locations.size();++i){
    dense_vector[locations[i]] += values[i] * coef;
  }
}

/////////////MATRIX//////////////
// To be viewed as an Y x X matrix where X are columns and Y rows.

SparseFloatMatrix::SparseFloatMatrix(unsigned int x_dims,unsigned int y_dims){
  this->x_dims = x_dims;
  this->y_dims = y_dims;
  values.resize(x_dims,SparseFloatVector(y_dims));
}

SparseFloatMatrix::SparseFloatMatrix (SparseFloatMatrix const &other){
    this->x_dims = other.x_dims;
    this->y_dims = other.y_dims;
    this->xlocations = other.xlocations;
    this->values = other.values;
}
SparseFloatMatrix& SparseFloatMatrix::operator= (SparseFloatMatrix const &other){
    this->x_dims = other.x_dims;
    this->y_dims = other.y_dims;
    this->xlocations = other.xlocations;
    this->values = other.values;
    return *this;
}


SparseFloatMatrix::SparseFloatMatrix(unsigned int x_dims,unsigned int y_dims, vector<Triplet> const &triples){

  this->x_dims = x_dims;
  this->y_dims = y_dims;
  values.resize(x_dims,SparseFloatVector(y_dims));

  for(int i = 0; i < triples.size();++i){

    assert(triples[i].row_idx < x_dims && triples[i].col_idx < y_dims);
    if (values[triples[i].row_idx].locations.size() == 0){xlocations.push_back(triples[i].row_idx);}
    values[triples[i].row_idx].coeffRef(triples[i].col_idx) += triples[i].value;
  }
}

SparseFloatMatrix::SparseFloatMatrix(unsigned int x_dims,unsigned int y_dims,string const &filename){//load from file
    this->x_dims = x_dims;
    this->y_dims = y_dims;
    values.resize(x_dims,SparseFloatVector(y_dims));
    load(filename,y_dims);
}

void SparseFloatMatrix::dot(SparseFloatVector const &xvec,vector<float> &yvec) const{
  assert(xvec.get_size() == x_dims && yvec.size()==y_dims);
  std::fill(yvec.begin(),yvec.end(),0);//init
  for(int i = 0; i < xvec.locations.size();++i){
    values[xvec.locations[i]].accumulate(yvec,xvec.values[i]);
  }
}

float SparseFloatMatrix::dot(vector<unsigned int> const &xvec,unsigned int row_index) const{ //assumed to be a vector of one valued hashed x coords, and y a row index.
    float score = 0;
    for(int i = 0; i < xvec.size();++i){
        score += values[xvec[i]][row_index];
    }
    return score;
}

void SparseFloatMatrix::dot(vector<unsigned int> const &xvec,vector<float> &yvec,vector<unsigned int> &sparse_locs){
    
    assert(yvec.size()==y_dims);
    std::fill(yvec.begin(),yvec.end(),0);//init
    for(int j = 0; j < sparse_locs.size();++j){
        //assert(xvec[sparse_locs[j]] < x_dims);
        values[xvec[sparse_locs[j]]].accumulate(yvec,1.0);
    }
}


void SparseFloatMatrix::dot(vector<unsigned int> const &xvec,vector<float> &yvec) const{ //assumed to be a vector of one valued hashed x coords
  assert(yvec.size()==y_dims);
  std::fill(yvec.begin(),yvec.end(),0);//init
  for(int i = 0; i < xvec.size();++i){
    assert(xvec[i] < x_dims);
    values[xvec[i]].accumulate(yvec,1.0);
  }
}

void SparseFloatMatrix::add_rowY(SparseFloatVector const &x_delta,unsigned int ydense_index){
  assert(x_delta.get_size() == x_dims && ydense_index < y_dims);
  for(int i = 0; i < x_delta.locations.size();++i){
    if(values[x_delta.locations[i]].locations.size() == 0){xlocations.push_back(x_delta.locations[i]);}
    values[x_delta.locations[i]].coeffRef(ydense_index) += x_delta.values[i];
  } 
}

void SparseFloatMatrix::add_rowY(vector<unsigned int> const &x_idxes,unsigned int ydense_index){
  assert(ydense_index < y_dims);
  for(int i = 0; i < x_idxes.size();++i){
    assert(x_idxes[i] < x_dims);
    if(values[x_idxes[i]].locations.size() == 0){xlocations.push_back(x_idxes[i]);}
    values[x_idxes[i]].coeffRef(ydense_index) += 1.0;
  } 
}

void SparseFloatMatrix::subs_rowY(SparseFloatVector const &x_delta,unsigned int ydense_index){
    assert(x_delta.get_size() == x_dims);
    assert(ydense_index < y_dims);
  for(int i = 0; i < x_delta.locations.size();++i){
    if(values[x_delta.locations[i]].locations.size() == 0){xlocations.push_back(x_delta.locations[i]);}
    values[x_delta.locations[i]].coeffRef(ydense_index) -= x_delta.values[i];
  } 
}

void SparseFloatMatrix::subs_rowY(vector<unsigned int> const &x_idxes,unsigned int ydense_index){
  assert(ydense_index < y_dims);
  for(int i = 0; i < x_idxes.size();++i){
    assert(x_idxes[i] < x_dims);
    if(values[x_idxes[i]].locations.size() == 0){xlocations.push_back(x_idxes[i]);}
    values[x_idxes[i]].coeffRef(ydense_index) -= 1.0;
  } 
}


SparseFloatMatrix& SparseFloatMatrix::operator+= (SparseFloatMatrix const &other){
  for(int i = 0; i < other.xlocations.size();++i){
    if(values[other.xlocations[i]].locations.size() == 0){xlocations.push_back(other.xlocations[i]);}
    values[other.xlocations[i]] += other.values[other.xlocations[i]];
  }
  return *this;
}

SparseFloatMatrix& SparseFloatMatrix::operator-= (SparseFloatMatrix const &other){
  for(int i = 0; i < other.xlocations.size();++i){
    if(values[other.xlocations[i]].locations.size() == 0){xlocations.push_back(other.xlocations[i]);}
    values[other.xlocations[i]] -= other.values[other.xlocations[i]];
  }
  return *this;
}

SparseFloatMatrix& SparseFloatMatrix::operator*= (float scalar){
  #pragma omp parallel for schedule(dynamic)
  for(int i = 0; i < xlocations.size();++i){
    values[xlocations[i]] *= scalar;
  }
  return *this;
}

SparseFloatMatrix& SparseFloatMatrix::operator/= (float scalar){
  #pragma omp parallel for schedule(dynamic)
  for(int i = 0; i < xlocations.size();++i){
    values[xlocations[i]] /= scalar;
  }
  return *this;
}


float SparseFloatMatrix::sqL2norm()const{
    float norm = 0;
    for(int i = 0; i < xlocations.size();++i){
        norm += values[xlocations[i]].sqL2norm();
    }
    return norm;
}



void  SparseFloatMatrix::clear(){
  for(int i = 0; i < xlocations.size();++i){
    values[xlocations[i]].clear();
  }
  xlocations.clear();
}

void SparseFloatMatrix::row_means(vector<float> &means_vec){
  means_vec.resize(y_dims,0);
  for(int i = 0; i < xlocations.size();++i){
    values[xlocations[i]].accumulate(means_vec,1.0);
  }
  for(int i = 0; i < means_vec.size();++i){
    means_vec[i] /= xlocations.size();
  }
}

void SparseFloatMatrix::row_variances(vector<float> &var_vec,bool stddev){
  vector<float> means_vec;
  row_means(means_vec);
  var_vec.resize(y_dims,0);
  for(int i = 0; i < xlocations.size();++i){
    vector<float> diff_vec(y_dims,0);
    for(unsigned int j = 0; j < y_dims ;++j){
        diff_vec[j] = std::pow(means_vec[j] - values[xlocations[i]].coeffRef(j),2);
    }
    values[xlocations[i]].accumulate(diff_vec,1.0);
  }
  for(int i = 0; i < var_vec.size();++i){
    if (stddev){var_vec[i] /= sqrt(xlocations.size());}
    else{var_vec[i] /= xlocations.size();}
  }
}

void SparseFloatMatrix::compress(){
  
  vector<unsigned int> tmp_xlocations;

  for(int i = 0; i < xlocations.size();++i){

    values[xlocations[i]].compress();
    if(values[xlocations[i]].locations.size() > 0){tmp_xlocations.push_back(xlocations[i]);}

  }
  xlocations = tmp_xlocations;
}

bool SparseFloatMatrix::operator==(SparseFloatMatrix const &other)const{//moderately inefficient
  for(int i = 0; i < xlocations.size();++i){
    if (values[xlocations[i]] != other.values[xlocations[i]]){return false;}
  }
 
  for(int i = 0; i < other.xlocations.size();++i){
    if (values[other.xlocations[i]] != other.values[other.xlocations[i]]){return false;}
  }
  return true;
}


bool SparseFloatMatrix::operator!=( SparseFloatMatrix const &other)const{
  return ! (*this == other);
}

void SparseFloatMatrix::load(string const &filename,size_t y_size){
    
    ifstream infile;
    infile.open(filename,ios::binary);

    //Reset
    this->x_dims = KERNEL_SIZE;
    this->y_dims = y_size;
    clear();
    unsigned int bidx=0;
    
    while(infile.read((char*)(&bidx),sizeof(unsigned int))){//Reads records
        xlocations.push_back(bidx);
        SparseFloatVector sp(y_dims);
        sp.read(infile);
        values[bidx] = sp;
    }
    infile.close();
}

void SparseFloatMatrix::save(string const &filename){
    
    this->compress();               //compresses the sparse matrix before saving
    
    ofstream outfile;
    outfile.open(filename,ios::binary);
    //Write actual record, bucket_id+Nactions floats
    for (unsigned int i = 0; i < xlocations.size();++i){
        outfile.write((char*) &xlocations[i],sizeof(unsigned int));//bucket_id
        values[xlocations[i]].write(outfile);
    }
    outfile.close();
}

ostream &operator<<( ostream &os, const SparseFloatMatrix &mat){
  for(int i = 0;  i < mat.xlocations.size();++i){
    os << "row"<< mat.xlocations[i]<<": "<< mat.values[mat.xlocations[i]] << endl;
  }
  return os;
}

/*
int main(){

  SparseFloatVector vecA(1000);
  vecA.coeffRef(10) = 3;
  vecA.coeffRef(12) = 1;
  vecA.coeffRef(100) = 7;
  vecA.coeffRef(759) = 2;
  cout << vecA << endl;
  
  SparseFloatVector vecB(1000);
  vecB.coeffRef(12) = -3;
  vecB.coeffRef(101) = 8;
  vecB.coeffRef(795) = -2;
  vecB.coeffRef(759) = 4;
  cout << vecB << endl;

  cout << vecA+vecB << endl;
  cout << vecA-vecB << endl;
  cout << (vecA-vecB)/2 <<endl;
  cout << (vecA-vecB)*3 <<endl;
  cout << ((vecB*3)/3)-vecB<<endl;


  //Test for parsing dot product
  vector<Triplet> dyn_mat;
  for(int i = 1; i < 70;++i){
    dyn_mat.push_back(Triplet(i*100,i*3,-3));
  }
  SparseFloatMatrix sp_matA(10000,300,dyn_mat);
  vector<unsigned int> x_idxes;
  vector<float> result;
  result.resize(300,0);
  for(int i = 1; i < 70;++i){x_idxes.push_back(i*100);}
  SparseFloatVector x_idxesB(10000);
  for(int i = 1; i < 70;++i){x_idxesB.coeffRef(i*100) += 1.0;}
  cout << x_idxesB<< endl;
  sp_matA.dot(x_idxes,result);
  for(int i = 0; i < result.size();++i){
    cout << result[i] << " ";
  }
  cout << endl;
  sp_matA.dot(x_idxesB,result);
  for(int i = 0; i < result.size();++i){
    cout << result[i] << " ";
  }
  cout << endl;

  SparseFloatMatrix sp_matB(10000,300,dyn_mat);
  //SparseFloatMatrix sp_matC(10000,300,dyn_matC);

  //Tests for update 
  //Simple perceptron update
  sp_matB = sp_matA;
  assert(sp_matA == sp_matB);
  sp_matA.add_rowY(x_idxes,19);
  sp_matA.add_rowY(x_idxesB,152);
  sp_matA.subs_rowY(x_idxesB,19);
  sp_matA.subs_rowY(x_idxes,152);
  assert(sp_matA == sp_matB);

  //Mini batch Mira and Pegasos updates
  SparseFloatMatrix deltaA(10000,300);
  deltaA.add_rowY(x_idxes,19);
  deltaA.add_rowY(x_idxes,21);
  deltaA.subs_rowY(x_idxes,37);
  deltaA.subs_rowY(x_idxes,28);

  SparseFloatMatrix deltaB(10000,300);
  deltaB.add_rowY(x_idxes,78);
  deltaB.add_rowY(x_idxes,90);
  deltaB.subs_rowY(x_idxes,12);
  deltaB.subs_rowY(x_idxes,15);

  SparseFloatMatrix delta(10000,300);
  delta += deltaA;
  delta += deltaB;
  delta *= 0.5;
  sp_matA += delta;
  sp_matA -= delta;
  assert(sp_matA == sp_matB);

  cout << "***"<<endl;
  //sp_matB *= 2.0; 
  //sp_matB *= 1/4;
  // sp_matC /= 2.0;
  // sp_matB += sp_matC;
  sp_matA -= sp_matB;
  sp_matA.compress();
  cout << "***"<<endl;
  cout <<  sp_matA << endl;
  sp_matB.clear();
  cout << "***"<<endl;
  cout << sp_matB << endl;
}
 */
