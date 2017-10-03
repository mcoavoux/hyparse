#include "lexer.h"


InputToken::InputToken(){
    idx = -1;
    jdx = -1;
    gram_code = 0;
}

InputToken::InputToken(const vector<PSTRING> &str_fields,int idx, int jdx){
  
  IntegerEncoder *enc = IntegerEncoder::get();
  this->wordform = str_fields[0];
  fields.push_back(enc->get_code(str_fields[0],0));
  fields.push_back(enc->get_code(str_fields[1],1));//assigns lexical code
  gram_code = enc->get_code(str_fields[1],IntegerEncoder::PS_COLCODE);//assigns gram code

  for(int i = 2; i < str_fields.size();++i){
    this->fields.push_back(enc->get_code(str_fields[i],i));
  }
  this->idx = idx;
  this->jdx = jdx;
};


InputToken::InputToken(const InputToken &other){
    
    this->idx = other.idx;
    this->jdx = other.jdx;
    this->wordform  = other.wordform;
    this->gram_code = other.gram_code;
    this->fields = other.fields;
}


ostream& operator<< (ostream &os,const InputToken &token){
    IntegerEncoder *enc = IntegerEncoder::get();
    if(token.is_null_token()){return os << "NULL";}
    os << "("<< token.idx <<","<< token.jdx << ") " <<  str::encode(token.wordform)<< " ("<< enc->decode8(token.fields[0]) <<"), "<< enc->decode8(token.fields[1]);
    for(int i = 2; i < token.fields.size();++i){
        os << "\t" << enc->decode8(token.fields[i]);
    }
    return os;
}

bool InputToken::operator==(InputToken const &other)const{
    return this->idx == other.idx && this->jdx == other.jdx && this->gram_code == other.gram_code;
}

const InputToken* InputDag::merge_left(TOK_CODE pred_pos,InputToken const *left,InputToken const *right){//merges two tokens and sets them a new joint pos,left token is head
    input_dag.push_back(new InputToken(*left));
    input_dag.back()->jdx = right->jdx;
    input_dag.back()->gram_code = pred_pos;
    input_dag.back()->wordform = left->wordform+InputToken::CPD_SEP+right->wordform;
    return input_dag.back();
}

const InputToken* InputDag::merge_right(TOK_CODE pred_pos,InputToken const *left,InputToken const *right){//merges two tokens and sets them a new joint pos, right token is head
    input_dag.push_back(new InputToken(*right));
    input_dag.back()->idx = left->idx;
    input_dag.back()->gram_code = pred_pos;
    input_dag.back()->wordform = left->wordform+InputToken::CPD_SEP+right->wordform;
    return input_dag.back();
}

const InputToken* InputDag::shift_token(TOK_CODE pred_pos,int J){//shifts the first token in the queue into the dag, sets it a tag and returns a pointer to it.
    input_dag.push_back(new InputToken(*input_queue[J]));
    input_dag.back()->gram_code = pred_pos;
    input_dag.back()->fields[1] = pred_pos;
    return input_dag.back();
}

const InputToken* InputDag::shift_token(int J){//shifts the first token in the queue into the dag without modifying it
    input_dag.push_back(new InputToken(*input_queue[J]));
    //return input_dag.back();
    return input_queue[J];
}

InputDag::~InputDag(){
    this->clear();
}
InputDag::InputDag(InputDag const &other){
    for(list<InputToken*>::const_iterator it = other.input_dag.begin();it != other.input_dag.end();++it){
        this->input_dag.push_back(new InputToken(**it));
    }
    for(vector<InputToken*>::const_iterator it = other.input_queue.begin();it != other.input_queue.end();++it){
        this->input_queue.push_back(new InputToken(**it));
    }
}



const InputToken* InputDag::operator[](unsigned int idx)const{return input_queue.at(idx);}//returns a pointer to input token at index idx in the queue
const InputToken* InputDag::at(unsigned int idx)const{return input_queue.at(idx);}//returns a pointer to input token at index idx in the queue

void InputDag::push_back(InputToken  *token){input_queue.push_back(token);}//adds a token to the queue
size_t InputDag::size()const{return input_queue.size();}

void InputDag::reset(){//empties the dag and resets the structure for a fresh new parse
    for(list<InputToken*>::iterator it = input_dag.begin();it != input_dag.end();++it){delete *it;}
    input_dag.clear();
}
void InputDag::clear(){ //this empties everything in dag and queue
    for(list<InputToken*>::iterator it = input_dag.begin();it != input_dag.end();++it){delete *it;}
    for(vector<InputToken*>::iterator it = input_queue.begin();it != input_queue.end();++it){delete *it;}
    input_dag.clear();
    input_queue.clear();
}

bool InputDag::empty(){//returns true if input_queue is empty
    return input_queue.empty();
}

ostream& operator<< (ostream &os,const InputDag &input_sequence){
    os << "Dag: "<<endl;
    for(list<InputToken*>::const_iterator it = input_sequence.input_dag.begin(); it  != input_sequence.input_dag.end();++it){
        os << **it <<" ";
    }
    os << endl << "Queue: "<<endl;
    for(int i = 0; i < input_sequence.size();++i){
        os << *(input_sequence.input_queue[i]) << " ";
    }
    os << endl;
    return os;
}

bool TbkLexer::skip_header(istream &inFile){

    string u8bfr;
    vector<wstring> fields;
    
    while (getline(inFile,u8bfr)){
        unsigned int arity = stok.llex(u8bfr,fields);
        if (arity > 0){return true;}
    }
    return false;
}

bool MarmotLexer::skip_header(istream &inFile){
  return true;
}

bool TbkLexer::next_sentence(istream &inFile,InputDag &tokens_read){
    
  string u8bfr;
  tokens_read.clear();
  vector<wstring> fields;

  int idx = 0;
  while (getline(inFile,u8bfr)){
    unsigned int arity = stok.llex(u8bfr,fields);
    if (this->arity == 0 && arity > 0){this->arity = arity;}
    if (arity == 0 && tokens_read.size() > 0){return true;}
    if (arity > 0 && arity != this->arity){cerr << "Error badly formatted line ("<< u8bfr << ") : aborting"<<endl;}
    if (fields.size() > 0){tokens_read.push_back(new InputToken(fields,idx,idx+1));}
    fields.clear();
     u8bfr.clear();
    ++idx;
  }
  return !tokens_read.empty();
}

MarmotLexer::MarmotLexer(vector<PSTRING> const &ttd){
  this->arity = ttd.size();
  for(int i = 0; i < this->arity;++i){
    ttd_map[ttd[i]] = i;
  }
  featurestok = str::SimpleTokenizer(L"|");
  attvaltok = str::SimpleTokenizer(L"=");
}

void MarmotLexer::read_ttd(vector<wstring> &ttd,string const &filename){
 
  ttd.clear();
  wifstream infile(filename);
  wstring bfr;
  while(getline(infile,bfr)){
    ttd.push_back(bfr);
  }
}

bool MarmotLexer::next_sentence(istream &inFile,InputDag &tokens_read){
  
  string u8bfr;
  tokens_read.clear();

  vector<wstring> input_fields;
  vector<wstring> ttd_fields(arity,L"Na");

  int idx = 0;
  while (getline(inFile,u8bfr)){

    unsigned loc_arity = linetok.llex(u8bfr,input_fields);
    if (loc_arity == 0 && tokens_read.size() > 0){return true;}    
    if (input_fields.size() > 0){
      ttd_fields[0] = input_fields[1];
      ttd_fields[1] = input_fields[5];
      vector<wstring> features;
      featurestok.llex(input_fields.back(),features);
      for(int j = 0 ; j < features.size() ; ++j){
	vector<wstring> attval;
	attvaltok.llex(features[j],attval);
	ttd_fields[ttd_map[attval[0]]] = attval[1];
      }
      tokens_read.push_back(new InputToken(ttd_fields,idx,idx+1));
    }
    input_fields.clear();
    ttd_fields.clear();
    ttd_fields.resize(arity,L"Na");
    u8bfr.clear();
    ++idx;
  }
  return !tokens_read.empty();
}


bool AbstractLexer::next_batch(size_t batch_size,istream &input_source,vector<InputDag> &input_batch){
    
    int c = 0;
    input_batch.resize(0);
    InputDag input_sequence;
    while(c < batch_size && next_sentence(input_source,input_sequence)){
        input_batch.push_back(input_sequence);
        ++c;
    }
    return !input_batch.empty();
}


void AbstractLexer::process_file(istream &inFile,vector<InputDag> &sentences){
    InputDag cur_sentence;
    while(next_sentence(inFile,cur_sentence)){
        sentences.push_back(cur_sentence);
    }
}
