#include "sparse_encoder.h"
#include "globals.h"
#include "str_utils.h"
#include <vector>
#include <string>
#include <algorithm>

TemplateTypeDefinition::TemplateTypeDefinition(vector<PSTRING> const &colnames){
    this->colnames = colnames;
}

TemplateTypeDefinition::TemplateTypeDefinition(TemplateTypeDefinition const &other){
    this->colnames = other.colnames;
}

TemplateTypeDefinition& TemplateTypeDefinition::operator=(TemplateTypeDefinition const &other){
    this->colnames = other.colnames;
    return *this;
}

PSTRING TemplateTypeDefinition::get_colname(int idx)const{
    return colnames[idx];
}


int TemplateTypeDefinition::operator()(PSTRING const &colname)const{
    for(int idx = 0; idx < colnames.size();++idx){
        if(colnames[idx] == colname){return idx;}
    }
    cerr << "Error : a template is trying to address an unknown column '" << str::encode(colname)<< "' in the data set.\naborting." << endl;
    exit(1);
}

bool TemplateTypeDefinition::operator==(TemplateTypeDefinition const &other)const{
    return std::equal(colnames.begin(),colnames.end(),other.colnames.begin());
}

void TemplateTypeDefinition::save(string const &filename)const{
    wofstream outFile(filename);
    for(int i = 0 ; i < colnames.size();++i){
        outFile << colnames[i]<<endl;
    }
    outFile.close();
}

void TemplateTypeDefinition::load(string const &filename){
    
    colnames.clear();
    wifstream inFile(filename);
    wstring bfr;
    while(getline(inFile,bfr)){
        colnames.push_back(bfr);
    }
    inFile.close();
}

TOK_CODE FeatureSensor::get_lex_value(int tok_idx,int token_field,StateSignature const &signature)const{
  if (tok_idx < 0){return undefined_left;}
  if (tok_idx >= signature.N){return undefined_right;}
  return (*(*(signature.input_sequence))[tok_idx])[token_field];
}

void FeatureSensor::set_undefined_values(int colidx){
    undefined_left = IntegerEncoder::get()->encode(L"$UNDEF_LEFT$",colidx);
    undefined_right = IntegerEncoder::get()->encode(L"$UNDEF_RIGHT$",colidx);
}

StackFeatureSensor::StackFeatureSensor(unsigned int stack_idx,ITEM_POS posn){
  this->stack_idx   = stack_idx;
  this->posn        = posn;
  this->field       = CATEGORY;
  this->token_field = -1;
  set_undefined_values(token_field);
}

StackFeatureSensor::StackFeatureSensor(unsigned int stack_idx,ITEM_POS posn, int token_field){
  this->stack_idx   = stack_idx;
  this->posn        = posn;
  this->field       = HEAD_WORD;
  this->token_field = token_field;
  set_undefined_values(token_field);
}

StackFeatureSensor::StackFeatureSensor(const StackFeatureSensor &other){
  this->stack_idx   = other.stack_idx;
  this->posn        = other.posn;
  this->field       = other.field;
  this->token_field = other.token_field;  
  set_undefined_values(token_field);
}

StackFeatureSensor::StackFeatureSensor(){
    set_undefined_values(IntegerEncoder::PS_COLCODE);
}

StackFeatureSensor& StackFeatureSensor::operator=(const StackFeatureSensor &other){
  this->stack_idx   = other.stack_idx;
  this->posn        = other.posn;
  this->field       = other.field;
  this->token_field = other.token_field;  
  return *this;
}

bool StackFeatureSensor::do_sanity_check()const{
    if (stack_idx >= StateSignature::STACK_LOCALITY){
        cerr << "Template reader error :: trying to address a value too deeply nested in the stack"<<endl;
        return false;
    }
    if (stack_idx == 2 and posn != TOP){
        cerr << "Template reader error :: trying to address a value too deeply nested in the stack"<<endl;
        return false;
    }
    return true;
}

PSTRING StackFeatureSensor::toString(TemplateTypeDefinition const &ttd)const{
    wchar_t pos;
    switch (posn){
        case StackFeatureSensor::TOP:
            pos = L't';
            break;
        case StackFeatureSensor::LEFT:
            pos = L'l';
            break;
        case StackFeatureSensor::RIGHT:
            pos = L'r';
            break;
        case StackFeatureSensor::LC:
            pos = L'L';
            break;
        case StackFeatureSensor::RC:
            pos = L'R';
            break;
    }
    wchar_t field = L'c';
    if (this->field == StackFeatureSensor::HEAD_WORD) {field = L'h';}
    PSTRING s = L"_";
    if (this->field == StackFeatureSensor::HEAD_WORD){s = ttd.get_colname(token_field);}
    
    PSTRING res = L"s"+ to_wstring(stack_idx)+L"("+pos+L","+field+L","+ s + L")";
    return res;
}


TOK_CODE StackFeatureSensor::get_value(StateSignature const &signature)const{
    
  //check for stack overflow
  if (! (this->stack_idx < signature.stack_size)){return undefined_left;}
  if(this->field == CATEGORY){
        switch(this->posn){
	    case (TOP):
	      return signature.stack[stack_idx].top;
        case (LEFT):
	      return signature.stack[stack_idx].left;
        case (RIGHT):
	      return signature.stack[stack_idx].right;
        default:
                cerr << "Critical design error : aborting." << endl;
                exit(1);
        }
  }else{//this->field == HEAD_WORD
    switch(this->posn){
	    case (TOP):
        {
            if (NULL_TOKEN == *signature.stack[stack_idx].htop){return undefined_left;}
            return (*signature.stack[stack_idx].htop)[token_field];
        }
        case (LEFT):
        {
            if (NULL_TOKEN == *signature.stack[stack_idx].hleft){return undefined_left;}
            return (*signature.stack[stack_idx].hleft)[token_field];
        }
        case (RIGHT):
        {
            if (NULL_TOKEN == *signature.stack[stack_idx].hright){return undefined_left;}
            return (*signature.stack[stack_idx].hright)[token_field];
        }
        case (LC):
        {
            if (NULL_TOKEN == *signature.stack[stack_idx].left_corner){return undefined_left;}
            return (*signature.stack[stack_idx].left_corner)[token_field];
        }
        case (RC):
        {
            if (NULL_TOKEN == *signature.stack[stack_idx].right_corner){return undefined_left;}
            return (*signature.stack[stack_idx].right_corner)[token_field];
        }
        default:
            cerr << "Critical design error : aborting." << endl;
            exit(1);
    }
  }
}

StackFeatureSensor* StackFeatureSensor::clone()const{
  return new StackFeatureSensor(*this);
}

QueueFeatureSensor::QueueFeatureSensor(){
    set_undefined_values(IntegerEncoder::PS_COLCODE);
}

QueueFeatureSensor::QueueFeatureSensor(unsigned int queue_shift,int token_field){
  this->queue_idx   = queue_shift+1;
  this->token_field = token_field;
  set_undefined_values(token_field);
}

QueueFeatureSensor::QueueFeatureSensor(const QueueFeatureSensor &other){
  this->queue_idx   = other.queue_idx;
  this->token_field = other.token_field;
  set_undefined_values(token_field);
}

QueueFeatureSensor& QueueFeatureSensor::operator=(const QueueFeatureSensor &other){
  this->queue_idx   = other.queue_idx;
  this->token_field = other.token_field;
  return *this;
}

QueueFeatureSensor* QueueFeatureSensor::clone()const{
  return new QueueFeatureSensor(*this);
}

TOK_CODE QueueFeatureSensor::get_value(StateSignature const &signature)const{
  return get_lex_value(queue_idx+signature.J,token_field,signature);
}

PSTRING QueueFeatureSensor::toString(TemplateTypeDefinition const &ttd)const{
    PSTRING res = L"q"+ to_wstring(queue_idx-1)+L"("+ttd.get_colname(token_field)+ L")";
    return res;
}



FeatureTemplate::~FeatureTemplate(){
  for(int i = 0; i < sensors.size();++i){
    delete sensors[i];
  }
}

FeatureTemplate::FeatureTemplate(){
    this->template_idx = 0;
    this->S = 0;
    this->feature_signature.set_template(0);
}


FeatureTemplate::FeatureTemplate(unsigned int tpl_idx){
  this->template_idx = tpl_idx; 
  this->S = 0;
  this->feature_signature.set_template(tpl_idx);
}


int get_template_index(wstring const &tpl_lhs){
    str::SimpleTokenizer sp(L" ,()");
    vector<wstring> fields;
    sp.llex(tpl_lhs,fields);
    return std::stoi(fields[1]);
}

StackFeatureSensor* read_stack_sensor(vector<wstring> const &sensor_fields,TemplateTypeDefinition const &ttd){
    
    //find position
    wchar_t p = sensor_fields[1][0];
    //wcout << L"pos: "<<p<<endl;
    StackFeatureSensor::ITEM_POS item_pos = StackFeatureSensor::TOP;
    
    switch(p){
        case L't':break;
        case L'l':item_pos = StackFeatureSensor::LEFT;
            break;
        case L'r':
            item_pos = StackFeatureSensor::RIGHT;
            break;
        case L'L':item_pos = StackFeatureSensor::LC;
            break;
        case L'R':
            item_pos = StackFeatureSensor::RC;
            break;
        default:
            cerr << "error invalid position"<<endl;
            exit(1);
    }
    wstring sindex(&sensor_fields[0][1]);
    if(item_pos == StackFeatureSensor::LC || item_pos == StackFeatureSensor::RC){
        return new StackFeatureSensor(std::stoi(sindex),item_pos,ttd(sensor_fields[3]));
    }else if (sensor_fields[2] == L"c"){
        return new StackFeatureSensor(std::stoi(sindex),item_pos);
    }else if (sensor_fields[2] == L"h"){
        return new StackFeatureSensor(std::stoi(sindex),item_pos,ttd(sensor_fields[3]));
    }else{
        cerr<< "invalid template"<<endl;
        exit(1);
    }
}

QueueFeatureSensor* read_queue_sensor(vector<wstring> const &sensor_fields,TemplateTypeDefinition const &ttd){
    wstring sindex(&sensor_fields[0][1]);
    return new QueueFeatureSensor(stoi(sindex),ttd(sensor_fields[1]));
}



FeatureSensor* read_sensor(wstring const &sensors_desc,TemplateTypeDefinition const &ttd){
    str::SimpleTokenizer sp(L" ,()");
    vector<wstring> fields;
    sp.llex(sensors_desc, fields);
    
    if (fields[0][0] == L's'){
        
        return read_stack_sensor(fields,ttd);
        
    }else if(fields[0][0] == L'q'){
        
        return read_queue_sensor(fields,ttd);
        
    }else{
        
        cerr << "error reading sensor: " << str::encode(fields[0]) << endl;
        exit(1);
    }
}


FeatureTemplate::FeatureTemplate(wstring const &template_line,
                                 TemplateTypeDefinition const &ttd){
    
    vector<wstring> fields;
    str::SimpleTokenizer sp(L"=&");
    sp.llex(template_line,fields);
    this->clear_sensors();
    this->morpho = this->get_morpho(fields[0]);
    this->template_idx = get_template_index(fields[0]);
    this->feature_signature.set_template(this->template_idx);
    this->S = 0;
    for(int i = 1; i < fields.size();++i){
        this->sensors.push_back(read_sensor(fields[i],ttd));
        ++(this->S);
    }
    
}



FeatureTemplate::FeatureTemplate(const FeatureTemplate &other){
   for(int i = 0; i < sensors.size();++i){
        delete sensors[i];
   }
  this->sensors.clear();
  for(int i = 0; i < other.sensors.size();++i){
    this->add_sensor(other.sensors[i]->clone());
  }
  this->S = other.S;
  this->template_idx = other.template_idx; 
  this->feature_signature = other.feature_signature;
  this->morpho = other.morpho;
}
 
FeatureTemplate& FeatureTemplate::operator=(const FeatureTemplate &other){
  for(int i = 0; i < sensors.size();++i){
    delete sensors[i];
  }
  this->sensors.clear();
  for(int i = 0; i < other.sensors.size();++i){
    this->add_sensor(other.sensors[i]->clone());
  }
  this->template_idx = other.template_idx;
  this->S = other.S;
  this->feature_signature = other.feature_signature;
  this->morpho = other.morpho;
  return *this;
}

void FeatureTemplate::add_sensor(FeatureSensor *sensor){
  this->sensors.push_back(sensor);
  ++S;
}

unsigned int FeatureTemplate::get_index(StateSignature const &signature)const{

  AbstractFeature feature_signature(template_idx);
  for(int i = 0; i < S && i < 3;++i){
    feature_signature.set_value(i+1,sensors[i]->get_value(signature));
  }
  return indexer(feature_signature);
}



bool FeatureTemplate::get_morpho(wstring const &tpl_lhs)const{
    size_t idx = tpl_lhs.find_first_not_of(L' ');
    if(idx != wstring::npos && tpl_lhs[idx] == 'T'){
        return true;
    }
    return false;
}


void FeatureTemplate::clear_sensors(){
    for(int i = 0; i < sensors.size();++i){
        delete sensors[i];
    }
    sensors.clear();
}

PSTRING FeatureTemplate::toString(TemplateTypeDefinition const &ttd)const{
    PSTRING res;
    if (has_morpho()){res += L"T";}else{res += L"P";}
    res += L"("+to_wstring(template_idx)+ L") = "+sensors[0]->toString(ttd);
    for(int i = 1; i < sensors.size();++i){
        res+= L" & "+sensors[i]->toString(ttd);
    }
    return res;
}



bool FeatureTemplate::do_sanity_check()const{
    if (sensors.size() >= 4){
        cerr << "Error :: too many terms in template " << template_idx<<endl;
        cerr << "aborting."<<endl;
        exit(1);
    }
    for(int i = 0; i < sensors.size();++i){
        if (! sensors[i]->do_sanity_check()){
            cerr << "at template "<< this->template_idx << endl << "aborting."<<endl;
            exit(1);
        }
    }
    return true;
}




bool FeatureTemplate::get_template(wistream &instream,FeatureTemplate &tpl, TemplateTypeDefinition const &ttd){
    
    bool tpl_read = false;
    str::SimpleTokenizer sp(L"=&");
    wstring bfr;
    while(!tpl_read && getline(instream,bfr)){
        vector<wstring> fields;
        sp.llex(bfr,fields);
        if (fields.size() > 0){
            tpl.clear_sensors();
            tpl.morpho = tpl.get_morpho(fields[0]);
            tpl.template_idx = get_template_index(fields[0]);
            tpl.feature_signature.set_template(tpl.template_idx);
            tpl.S = 0;
            for(int i = 1; i < fields.size();++i){
                tpl.sensors.push_back(read_sensor(fields[i],ttd));
                ++tpl.S;
            }
            tpl_read = true;
        }
    }
    return tpl_read;
}


SparseEncoder::SparseEncoder(string const &filename,TemplateTypeDefinition const &ttd){
    this->T = 0;
    wifstream is(filename);
    FeatureTemplate tpl(0);
    while(FeatureTemplate::get_template(is,tpl,ttd)){
        add_template(tpl);
    }
    is.close();
}

SparseEncoder::SparseEncoder(const char *filename,TemplateTypeDefinition const &ttd){
    this->T = 0;
    wifstream is(filename);
    FeatureTemplate tpl(0);
    while(FeatureTemplate::get_template(is,tpl,ttd)){
        add_template(tpl);
    }
    is.close();
}


SparseEncoder::SparseEncoder(){
  this->T = 0;
}

SparseEncoder::SparseEncoder(SparseEncoder const &other){
  this->templates.clear();
  this->T = other.T;
  this->morpho = other.morpho;
  std::copy(other.templates.begin(),other.templates.end(),this->templates.begin());
}
SparseEncoder& SparseEncoder::operator=(const SparseEncoder &other){
  this->templates.clear();
  this->T = other.T;
  this->morpho = other.morpho;
  this->templates.resize(other.templates.size());
  std::copy(other.templates.begin(),other.templates.end(),this->templates.begin());
  return *this;
}

void SparseEncoder::add_template(FeatureTemplate const &ntemplate){
  ++T;
  templates.push_back(ntemplate);
  if(ntemplate.has_morpho()){morpho=true;}
}
  
//Encodes a configuration on a sparse Phi(x) vector (encoding param)
void SparseEncoder::encode(vector<unsigned int> &encoding,StateSignature const &signature,bool clear)const{
  if(clear){encoding.clear();}
  for(int i = 0; i < T;++i){
    encoding.push_back(templates[i].get_index(signature));
  }
}

//Encodes a configuration on a sparse Phi(x) vector (encoding param)
void SparseEncoder::encode(SparseFloatVector &encoding,StateSignature const &signature,bool clear)const{
    if(clear){encoding.clear();}
    for(int i = 0; i < T;++i){
        encoding.coeffRef(templates[i].get_index(signature)) += 1;
    }
}

PSTRING SparseEncoder::toString(TemplateTypeDefinition const &ttd)const{
    PSTRING res;
    for(int i = 0; i < templates.size();++i){
        res+= templates[i].toString(ttd)+L"\n";
    }
    return res;
}

void SparseEncoder::save(const char *filename,TemplateTypeDefinition const &ttd)const{
    wofstream out(filename);
    out << toString(ttd);
    out.close();
}


bool SparseEncoder::do_sanity_check()const{
    //check for duplicate ids
    vector<int> id_vec;
    for(int t = 0; t < templates.size();++t){id_vec.push_back(templates[t].get_template_idx());}
    std::sort(id_vec.begin(),id_vec.end());
    int x = -1;
    for(int t = 0; t < templates.size();++t){
        int y = id_vec[t];
        if( x == y ){
            cerr << "Error : found two templates with the same id ("<< x << ")\naborting."<<endl;
            exit(1);
        }
        x = y;
    }
    for(int  t= 0 ; t < templates.size();++t){
        templates[t].do_sanity_check();
    }
    return true;
}




