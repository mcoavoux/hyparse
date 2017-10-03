#include "sparse_encoder.h"
#include "dense_encoder.h"
#include "globals.h"
#include "str_utils.h"

#include <vector>
#include <string>


DenseEncoderGlobal::DenseEncoderGlobal(string const &filename,TemplateTypeDefinition const &ttd){
    this->T = 0;
    wifstream is(filename);
    FeatureTemplate tpl(0);
    while(FeatureTemplate::get_template(is,tpl,ttd)){
        add_template(tpl);
    }
    is.close();
}

DenseEncoderGlobal::DenseEncoderGlobal(const char *filename,TemplateTypeDefinition const &ttd){
    this->T = 0;
    wifstream is(filename);
    FeatureTemplate tpl(0);
    while(FeatureTemplate::get_template(is,tpl,ttd)){
        add_template(tpl);
    }
    is.close();
}


DenseEncoderGlobal::DenseEncoderGlobal(){
    this->T = 0;
}

DenseEncoderGlobal::DenseEncoderGlobal(DenseEncoderGlobal const &other){
    this->templates.clear();
    this->T = other.T;
    this->templates.resize(other.templates.size());
    std::copy(other.templates.begin(),other.templates.end(),this->templates.begin());
    this->morpho = other.morpho;
}

DenseEncoderGlobal& DenseEncoderGlobal::operator=(const DenseEncoderGlobal &other){
    this->templates.clear();
    this->T = other.T;
    this->templates.resize(other.templates.size());
    std::copy(other.templates.begin(),other.templates.end(),this->templates.begin());
    this->morpho = other.morpho;
    return *this;
}

void DenseEncoderGlobal::add_template(FeatureTemplate const &ntemplate){
    ++T;
    templates.push_back(ntemplate);
    if(ntemplate.has_morpho()){morpho=true;}
}

void DenseEncoderGlobal::encode(vector<int> &encoding, StateSignature const &signature, bool clear){
    if (clear) {if (encoding.size() != T) encoding.clear();}
    encoding.resize(T);
    for (int i = 0; i < T; i++){
        encoding[i] = templates[i].get_index_in_lookup(signature);
    }
}

size_t DenseEncoderGlobal::ntemplates()const{
    return templates.size();
}


int DenseEncoderGlobal::get_colidx(int tpl_idx){
    return templates[tpl_idx].get_colidx();
}

