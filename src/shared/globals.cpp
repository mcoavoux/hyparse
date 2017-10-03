#include "globals.h"
#include "utf8.h"
#include "str_utils.h"
#include <iostream>
#include <sys/time.h>

#include <assert.h>

size_t EncoderHasher::operator()(const pair<PSTRING,int> &val) const   {
        size_t retval = str_hash(val.first);
        hash_combine(retval, val.second);
        return retval;
}

bool IntegerEncoder::exists = false;
IntegerEncoder* IntegerEncoder::instance = NULL;

IntegerEncoder* IntegerEncoder::get(){
  
  if (IntegerEncoder::exists){
    return IntegerEncoder::instance;
  }else{
    IntegerEncoder::instance = new IntegerEncoder();
    IntegerEncoder::exists = true;
    return IntegerEncoder::instance;
  }
}

void IntegerEncoder::kill(){
  
  if (IntegerEncoder::exists){
    delete IntegerEncoder::instance;
    IntegerEncoder::exists = false;
  }
}

IntegerEncoder::IntegerEncoder(){
  this->clear();
}

unsigned int IntegerEncoder::encode(PSTRING const &str,int colidx){
  // @@@m
  // if there is no default code for this column, create it
  int unkn_size = UNKNOWN.size() - 1;
  while (unkn_size <= colidx){
      encoder[make_pair(L"$UNK$",unkn_size)] = current_code;
      decoder.push_back(L"$UNK$");
      column_indexes.push_back(unkn_size);
      UNKNOWN.push_back(current_code++);
      unkn_size ++;
      assert(decoder.size() ==  column_indexes.size());

  }//@@@
  ENCODING_MAP::const_iterator got = encoder.find(make_pair(str,colidx));
  if(got == encoder.end()){
    encoder[make_pair(str,colidx)] = current_code;
    decoder.push_back(str);
    column_indexes.push_back(colidx);
    assert(decoder.size() ==  column_indexes.size());
    current_code++;
    return current_code-1;
  }else{
    return got->second;
  }
}

unsigned int IntegerEncoder::encode(string const &u8str,int colidx){
  PSTRING key;
  utf8::utf8to32(u8str.begin(),u8str.end(), back_inserter(key));
  return this->encode(key,colidx);
}

unsigned int IntegerEncoder::encode(const char *str,int colidx){
  string s(str);
  return encode(s,colidx);
}

void IntegerEncoder::encode_all(vector<PSTRING> const &elements,int colidx){
  for(vector<PSTRING>::const_iterator it = elements.begin();it != elements.end();++it){
    encode(*it,colidx);
  }
}

#ifdef CONVOLUTION
unsigned int IntegerEncoder::get_code(PSTRING const &str,int colidx){
    return encode(str, colidx);
}
unsigned int IntegerEncoder::get_code(string const &u8str,int colidx){
  PSTRING key;
  utf8::utf8to32(u8str.begin(),u8str.end(), back_inserter(key));
  return this->get_code(key,colidx);
}

unsigned int IntegerEncoder::get_code(const char *str,int colidx){
  string s(str);
  return get_code(s,colidx);
}
#else
unsigned int IntegerEncoder::get_code(PSTRING const &str,int colidx)const{

  unordered_map<pair<PSTRING,int>,TOK_CODE,EncoderHasher>::const_iterator got = encoder.find(make_pair(str,colidx));
  if(got == encoder.end()){
    // @@@m unknown words : return default code corresponding to column
    return UNKNOWN[colidx+1];
    //return 0;
  }else{
    return got->second;
  }
}

unsigned int IntegerEncoder::get_code(string const &u8str,int colidx)const{
  PSTRING key;
  utf8::utf8to32(u8str.begin(),u8str.end(), back_inserter(key));
  return this->get_code(key,colidx);
}

unsigned int IntegerEncoder::get_code(const char *str,int colidx) const{
  string s(str);
  return get_code(s,colidx);
}
#endif

PSTRING IntegerEncoder::decode(unsigned int code)const{
  return decoder[code];
} 

string IntegerEncoder::decode8(unsigned int code)const{
  string res;
  PSTRING tmp = decode(code);
  utf8::utf32to8(tmp.begin(),tmp.end(), back_inserter(res));
  return res;
}

void IntegerEncoder::clear(){
  encoder.clear();
  decoder.clear();
  column_indexes.clear();
  PSTRING default_value = L"$UNK$";
  current_code = 0;
  encode(default_value,PS_COLCODE);
  //encode(default_value,0); TODO ?
  UNKNOWN.clear(); // @@@m
}

void IntegerEncoder::load(string const &filename){
  this->clear();
  
  ifstream inputStream(filename);

  string bfr;
  str::SimpleTokenizer stok;
  while(getline(inputStream,bfr)){
      vector<wstring> fields;
      stok.llex(bfr, fields);
      if (fields.size() > 1){
          encode(fields[0],stoi(fields[1]));
      }
  }
  inputStream.close();
}

size_t IntegerEncoder::colsize(int colidx)const{
    size_t c = 0;
    for(int i = 0; i < column_indexes.size();++i){
        if(column_indexes[i]==colidx){++c;}
    }
    return c;
}

void IntegerEncoder::save(string const &filename)const{
  
  ofstream outStream(filename);

  size_t N = decoder.size();
  for (size_t i = 1; i < N;++i){
    outStream << decode8(i) <<"\t"<<column_indexes[i] <<  endl;
  }
  outStream.close();
}


void IntegerEncoder::head(int n){
     IntegerEncoder *enc = IntegerEncoder::get();
    for(int i = 0; i < n && i  < enc->size();++i){
        cout << i << ":" << str::encode(enc->decoder[i]) << ":"<< enc->column_indexes[i] << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
ColumnEncoder::ColumnEncoder(){
    this->clear();
}

//Encoding/decoding methods (encode is not thread safe)
/**
 * Non Thread safe version of encode.
 * If the string is unknown creates an integer code for this string and returns it.
 */
TOK_CODE ColumnEncoder::encode(PSTRING const &u32str){
    
    std::unordered_map<PSTRING,unsigned int>::const_iterator got = encoder.find(u32str);
    if(got == encoder.end()){
        encoder[u32str] = current_code;
        decoder.push_back(u32str);
        current_code++;
        return current_code-1;
    }else{
        return got->second;
    }
}

TOK_CODE ColumnEncoder::encode(string const &u8str){
    PSTRING key;
    utf8::utf8to32(u8str.begin(),u8str.end(), back_inserter(key));
    return this->encode(key);
}

TOK_CODE ColumnEncoder::encode(const char *u8str){
    return this->encode(string(u8str));
}

void ColumnEncoder::encode_all(vector<PSTRING> const &elements){
    
    for(vector<PSTRING>::const_iterator it = elements.begin();it != elements.end();++it){encode(*it);}

}

/**
 * Thread safe version of encode.
 * If the string is unknown returns a default unknown code
 */
TOK_CODE ColumnEncoder::get_code(PSTRING const &u32str) const{
    std::unordered_map<PSTRING,unsigned int>::const_iterator got = encoder.find(u32str);
    if(got == encoder.end()){
        return 0;
    }else{
        return got->second;
    }
}

TOK_CODE ColumnEncoder::get_code(string const &u8str) const{
    PSTRING key;
    utf8::utf8to32(u8str.begin(),u8str.end(), back_inserter(key));
    return this->get_code(key);
}

TOK_CODE ColumnEncoder::get_code(const char *u8str) const{
    return get_code(string(u8str));
}

void ColumnEncoder::clear(){
    
    ColumnEncoder::encoder.clear();
    decoder.clear();
    PSTRING default_value = StructuredEncoder::UNK_CODE;
    current_code = 0;
    encode(default_value);
    
}

PSTRING ColumnEncoder::decode(TOK_CODE code)const{
    return decoder[code];
}

string ColumnEncoder::decode8(TOK_CODE code)const{
    string res;
    PSTRING tmp = decode(code);
    utf8::utf32to8(tmp.begin(),tmp.end(), back_inserter(res));
    return res;
}


bool StructuredEncoder::exists = false;
StructuredEncoder* StructuredEncoder::instance = NULL;

const PSTRING  IntegerEncoder::UNK_SYMBOL = L"$UNK$";
const PSTRING  StructuredEncoder::UNK_CODE = L"$UNK$";

StructuredEncoder::StructuredEncoder(){
    
    encoder_list.push_back(new ColumnEncoder());
    encoder_list.push_back(new ColumnEncoder());
    
}

void StructuredEncoder::push_encoder(){
    
    if(!exists){
        instance = new StructuredEncoder();
        exists = true;
    }
    instance->encoder_list.push_back(new ColumnEncoder());
    
}

void StructuredEncoder::kill(){
    if(exists){
        exists = false;
        delete instance;
    }
}

StructuredEncoder::~StructuredEncoder(){
    clear();
}

void StructuredEncoder::clear(){
    for(vector<ColumnEncoder*>::const_iterator it = encoder_list.begin(); it != encoder_list.end();++it){
        delete *it;
    }
    encoder_list.clear();
}

ColumnEncoder* StructuredEncoder::get(int col_idx){
    if (!exists){
        instance = new StructuredEncoder();
        exists = true;
    }
    return instance->encoder_list[col_idx+1];
}


PSTRING StructuredEncoder::decode(TOK_CODE code,int colidx)const{
    if (colidx < StructuredEncoder::size()){return StructuredEncoder::get(colidx)->decode(code);}
    else{return UNK_CODE;}
}

string StructuredEncoder::decode8(TOK_CODE code,int colidx)const{
    return str::encode(decode(code,colidx));
}



TOK_CODE StructuredEncoder::encode(PSTRING const &u32str,int colidx){
    if(colidx >= StructuredEncoder::size()){StructuredEncoder::resize(colidx+1);}
    return StructuredEncoder::get(colidx)->encode(u32str);
}

TOK_CODE StructuredEncoder::encode(string const &u8str,int colidx){
    PSTRING bfr;
    str::decode(bfr,u8str);
    return encode(bfr,colidx);
}

TOK_CODE StructuredEncoder::encode(const char *u8str,int colidx){
    PSTRING bfr;
    str::decode(bfr,string(u8str));
    return encode(bfr,colidx);
}

void StructuredEncoder::encode_all(vector<PSTRING> const &elements,int colidx){
    for(vector<PSTRING>::const_iterator it = elements.begin();it != elements.end();++it){
        encode(*it, colidx);
    }
}

TOK_CODE StructuredEncoder::get_code(PSTRING const &u32str,int colidx) const{
    return StructuredEncoder::get(colidx)->get_code(u32str);
}

TOK_CODE StructuredEncoder::get_code(string const &u8str,int colidx) const{
    PSTRING bfr;
    str::decode(bfr, u8str);
    return get_code(bfr, colidx);
}

TOK_CODE StructuredEncoder::get_code(const char *u8str,int colidx) const{
    PSTRING bfr;
    str::decode(bfr, string(u8str));
    return get_code(bfr, colidx);    
}



StructuredEncoder* StructuredEncoder::get(){
    
    if (StructuredEncoder::exists){
        return StructuredEncoder::instance;
    }else{
        StructuredEncoder::instance = new StructuredEncoder();
        StructuredEncoder::exists = true;
        return StructuredEncoder::instance;
    }
}




void StructuredEncoder::resize(unsigned int k){
    if (!exists){
        instance = new StructuredEncoder();
        exists = true;
    }
    
    if(instance->size() < k){
        for(int i = instance->size(); i < k ; ++i){
            instance->push_encoder();
        }
    }
    
    if(instance->size() > k){
        for(int i = instance->size(); i > k;--i){//careful! size() = num of columns, not size of vec
            delete instance->encoder_list[i];
        }
        instance->encoder_list.resize(k+1);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

//DEBUG LOGGER
const string DebugLogger::PRIORITY_NAMES[] ={"INFO","DEBUG","WARNING","ERROR"};
bool DebugLogger::exists = false;
DebugLogger* DebugLogger::instance = NULL;


DebugLogger::DebugLogger(Priority minPriority,string const &logfile){
  
  this->minPriority = minPriority;  
  if (logfile != ""){
    fileStream.open(logfile.c_str());
  }
}

void DebugLogger::Start(Priority minPriority, const char *logFile){
  string logs(logFile);
  Start(minPriority,logs);
}

void DebugLogger::Start(Priority minPriority, const string &logfile){
  if (!DebugLogger::exists){
    DebugLogger::exists = true;
    DebugLogger::instance = new DebugLogger(minPriority,logfile);
  }
}

void DebugLogger::Stop(){
  if (DebugLogger::exists){
    if (DebugLogger::instance->fileStream.is_open()){DebugLogger::instance->fileStream.close();}
    delete DebugLogger::instance;
    DebugLogger::exists = false;
  }
}

void DebugLogger::Write(Priority priority, const string& message){

  if (DebugLogger::exists && priority >= instance->minPriority){
    if (instance->fileStream.is_open()){ 
      instance->fileStream << PRIORITY_NAMES[priority] << ":" << message << endl;
    }else{
      cout << PRIORITY_NAMES[priority] << ":" << message << endl;
    }
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////

//PARSE LOGGER
const string ParseLogger:: STATISTICS_NAMES[] ={"PREC","RECALL","FSCORE","TIME","LEN","NUNK","NSCORED"};
bool ParseLogger::exists = false;
ParseLogger* ParseLogger::instance = NULL;


ParseLogger::ParseLogger(int N,string const &logfile){
  
  if (N >= 7)
    {this->N = 7;}
  else
    {this->N = N;}

  this->statsline.resize(N,0.0);

  if (logfile != ""){
    fileStream.open(logfile.c_str());
  }
   total_time = 0;
   total_tokens = 0;
}

void ParseLogger::Start(const char *logFile,int N){
  string logs(logFile);
  Start(logs,N);
}

void ParseLogger::Start(const string &logfile,int N){
  
  if (!ParseLogger::exists){
    ParseLogger::exists = true;
    ParseLogger::instance = new ParseLogger(N,logfile);
    //Logs header
    if (instance->fileStream.is_open()){ 
      for (int i = 0; i < N;++i){
	ParseLogger::instance->fileStream << STATISTICS_NAMES[i] << "\t";
      }
      ParseLogger::instance->fileStream << endl;
    }else{
      for (int i = 0; i < N;++i){
	cout << STATISTICS_NAMES[i] << "\t";
      }
      cout << endl;
    }
  }
}

void ParseLogger::Stop(){
  if (ParseLogger::exists){
    //Close the whole thing
    if (ParseLogger::instance->fileStream.is_open()){ParseLogger::instance->fileStream.close();}
    if(instance->total_time > 0){//final message.
        cerr << "parsed "<<instance->total_tokens<<" tokens in "
                         << instance->total_time << " seconds ("
                         << instance->total_tokens/instance->total_time << " tokens/second)"<<endl;
    }
    delete ParseLogger::instance;
    ParseLogger::exists = false;
  }
}

void ParseLogger::Set(Statistics statname, float value){
 if (ParseLogger::exists){
     ParseLogger::instance->statsline[statname] = value;
     if (statname == Statistics::LEN){instance->total_tokens += value;}
 }
}


void ParseLogger::NextParse(){
  //Logs stat line
  if (ParseLogger::exists){
    if (instance->fileStream.is_open()){ 
      for (int i = 0; i < ParseLogger::instance->N;++i){
	ParseLogger::instance->fileStream << ParseLogger::instance->statsline[i] << "\t";
      }
      ParseLogger::instance->fileStream << endl;
    }else{
      for (int i = 0; i < ParseLogger::instance->N;++i){
	cout << ParseLogger::instance->statsline[i] << "\t";
      }
      cout << endl;
    }
    ParseLogger::instance->reset();
  }
}


void ParseLogger::StartTimer(){
    if (ParseLogger::exists){ParseLogger::instance->btime = ParseLogger::instance->get_time();}
}

void ParseLogger::StopTimer(){
    if(ParseLogger::exists){
        double end = ParseLogger::instance->get_time();
        double ctime = end - ParseLogger::instance->btime;
        Set(ParseLogger::TIME,ctime);
        instance->total_time += ctime;
    }
}

void ParseLogger::reset(){
  if (ParseLogger::exists){
    ParseLogger::instance->statsline.assign(ParseLogger::instance->N,0.0);
  }
}

double ParseLogger::get_time(){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1000000.0;
}

void ParseLogger::AddOneScored(){
    if(ParseLogger::exists){ParseLogger::instance->statsline[Statistics::NSCORED]++;}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
//LEARN LOGGER

const string LearnLogger::STATISTICS_NAMES[] = {"EPOCH","TRAINP","TRAINR","TRAINF","DEVP","DEVR","DEVF","MODEL_SIZE","LOSS"};
bool LearnLogger::exists = false;
LearnLogger* LearnLogger::instance = NULL;



LearnLogger::LearnLogger(int N,string const &logfile){
  
  this->N = N;
  this->epoch = 0;
  this->statsline.resize(N,0.0);
  if (logfile != ""){
    fileStream.open(logfile.c_str());
  }
}


void LearnLogger::Start(const char *logFile,int N){
  string logs(logFile);
  Start(logs,N);
}

void LearnLogger::Start(const string &logfile,int N){
  
  if (!LearnLogger::exists){
    LearnLogger::exists = true;
    LearnLogger::instance = new LearnLogger(N,logfile);

    //Logs header
    if (instance->fileStream.is_open()){ 
      for (int i = 0; i < N;++i){
	LearnLogger::instance->fileStream << STATISTICS_NAMES[i] << "\t";
      }
      LearnLogger::instance->fileStream << endl;
    }else{
      for (int i = 0; i < N;++i){
	cout << STATISTICS_NAMES[i] << "\t";
      }
      cout << endl;
    }
  }
}

void LearnLogger::Stop(){

  if (LearnLogger::exists){
    //Close the whole thing
    if (LearnLogger::instance->fileStream.is_open()){LearnLogger::instance->fileStream.close();}
    delete LearnLogger::instance;
    LearnLogger::exists = false;
  }

}


void LearnLogger::Set(Statistics statname, float value){
 if (LearnLogger::exists){LearnLogger::instance->statsline[statname] = value;}
}

void LearnLogger::NextEpoch(){
  
  if (LearnLogger::exists){
      //Logs stat line
      LearnLogger::instance->epoch+=1;
      LearnLogger::Set(LearnLogger::EPOCH, LearnLogger::instance->epoch);
      if (instance->fileStream.is_open()){
          for (int i = 0; i < LearnLogger::instance->N;++i){
              LearnLogger::instance->fileStream << LearnLogger::instance->statsline[i] << "\t";
          }
          LearnLogger::instance->fileStream << endl;
      }else{
          for (int i = 0; i < LearnLogger::instance->N;++i){
              cout << LearnLogger::instance->statsline[i] << "\t";
          }
          cout << endl;
      }
      LearnLogger::instance->reset();
  }
}

void LearnLogger::reset(){
  if (LearnLogger::exists){
    LearnLogger::instance->statsline.assign(LearnLogger::instance->N,0.0);
  }
}



