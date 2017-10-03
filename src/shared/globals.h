#ifndef GLOBALS_H
#define GLOBALS_H

#include <string>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <functional>

using namespace std;

/*
 *  This header defines some global constants and objects to be shared across a single execution
 *  such as encoders and loggers.
 *   
 *  These classes are not thread safe and may cause issues in multithreaded contexts.
 */

/**
 * The type to be used for string representation.
 * We encode strings most of the time as UTF-32 wide strings.
 */
typedef wstring PSTRING;

/**
 * The type used to encode tokens
 */
typedef unsigned int TOK_CODE;



//Utility function class for hashing encoder entries.
class EncoderHasher{
public:
    size_t operator()(const pair<PSTRING,int> &val) const;
protected:
    //this is Boost hash_combine function
    void hash_combine(size_t & seed, int const& val)const {seed ^= int_hash(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);}
private:
    std::hash<PSTRING> str_hash;
    std::hash<int> int_hash;
};


/**
 * The type to be used for encoding input strings as numbers (encoder)
 * It encodes strings as unsigned integers.
 * Note that 
 */
class IntegerEncoder{

public:

  static const int PS_COLCODE = -1; //abstract column for phrase struct symbols
  static const PSTRING UNK_SYMBOL;// '$UNK$' a string for encoding all symbols not seen at trainining time
    
    
  //Singleton methods
  static IntegerEncoder* get();
  static void kill();

  //Encoding/decoding methods (encode is not thread safe)
  //Non Thread safe version of encode.
  //If the string is unknown creates an integer code for this string and returns it.

  TOK_CODE encode(PSTRING const &u32str,int colidx);
  TOK_CODE encode(string const &u8str,int colidx);
  TOK_CODE encode(const char *u8str,int colidx);
  void encode_all(vector<PSTRING> const &elements,int colidx);
    
   // Thread safe version of encode.
   // If the string is unknown returns a default unknown code
   //
#ifdef CONVOLUTION
  TOK_CODE get_code(PSTRING const &u32str,int colidx) ;
  TOK_CODE get_code(string const &u8str,int colidx)   ;
  TOK_CODE get_code(const char *u8str,int colidx)     ;
#else
  TOK_CODE get_code(PSTRING const &u32str,int colidx) const;
  TOK_CODE get_code(string const &u8str,int colidx)   const;
  TOK_CODE get_code(const char *u8str,int colidx)     const;
#endif
  //
  //Decode int to string
  //
  PSTRING decode(TOK_CODE code) const;
  string decode8(TOK_CODE code) const;

  //
  //Gets columns indexes
  //
  inline int get_column(TOK_CODE code)const{return column_indexes[code];};
  size_t colsize(int colidx)const;//this one is slow
    
  //
  // Save the current state of this encoder to disk
  //
  void save(string const &filename)const;

  //
  //Clears the current state of this encoder and load it from disk
  //
  void load(string const &filename);

   //size of the encoder
  size_t size()const{return decoder.size();}

  //displays the n first elements encoded on cout
  static void head(int n);
    
  // @@m added September 2015  for char-based convolutional network
  void get_token_vocabulary(vector<PSTRING> &res, vector<int> &idx){
    res.clear();
    idx.clear();
    size_t N = decoder.size();
    for (size_t i = 1; i < N;++i){
      if (column_indexes[i] == 0){
        res.push_back(decode(i));
        idx.push_back(i);
      }
    }
  }

protected:
  
  IntegerEncoder();
  
  //clears the contents of this encoder
  void clear();

private:

  static bool exists;
  static IntegerEncoder *instance;

  unsigned int current_code;
    
  typedef unordered_map<pair<PSTRING,int>,TOK_CODE,EncoderHasher>  ENCODING_MAP ;
  ENCODING_MAP encoder;
    
  vector<PSTRING> decoder;
  vector<int> column_indexes; //this vector stores the column indexes in which the value has been found.
  vector<int> UNKNOWN; // @@@m this vector stores $UNK$ code for each column
};




//////////////////////////////////////////////////////////////////////////////////////////
//ENCODING FOR DENSE MODELS

/**
 * This encoder manages the encoding column by column in the data set.
 * This is a component of the StructuredEncoder
 * (used by the dense model)
 * 
 */
class ColumnEncoder{
    
public:
    
    ColumnEncoder();
    
    //Encoding/decoding methods (encode is not thread safe)
    /**
     * Non Thread safe version of encode.
     * If the string is unknown creates an integer code for this string and returns it.
     */
    TOK_CODE encode(PSTRING const &u32str);
    TOK_CODE encode(string const &u8str);
    TOK_CODE encode(const char *u8str);
    void encode_all(vector<PSTRING> const &elements);
    
    /**
     * Thread safe version of encode.
     * If the string is unknown returns a default unknown code
     */
    TOK_CODE get_code(PSTRING const &u32str) const;
    TOK_CODE get_code(string const &u8str) const;
    TOK_CODE get_code(const char *u8str) const;

    TOK_CODE get_defaultcode()const{return 0;};
    
    /**
     * Decode int to string
     */
    PSTRING decode(TOK_CODE code) const;
    string decode8(TOK_CODE code) const;

    
    void clear();

    size_t size()const{return decoder.size();}

private:
    
    unsigned int current_code;
    unordered_map<PSTRING,TOK_CODE> encoder;
    vector<PSTRING> decoder;
    
};


//Top level class for managing the encoding for dense models
/*
 * There are separate column encoders for each column and for constituents.
 * Each column encoder guarantees codes of a column to be contiguous, starting at 0
 *
 *
 */
 
class StructuredEncoder{
    
public:

    //Phrase structure codes
    static const int PS_COLCODE = -1;
    static const PSTRING UNK_CODE; //L"$UNK$"

    static StructuredEncoder* get();
    
    static void kill();
       
    //returns the number of columns actually encoded
    static int size(){return instance->encoder_list.size()-1;}
    
    TOK_CODE encode(PSTRING const &u32str,int colidx);
    TOK_CODE encode(string const &u8str,int colidx);
    TOK_CODE encode(const char *u8str,int colidx);
    void encode_all(vector<PSTRING> const &elements,int colidx);
    
    //
    //If the string is unknown or the colidx is unknown
    //returns a default unknown code
    //
    //
    TOK_CODE get_code(PSTRING const &u32str,int colidx) const;
    TOK_CODE get_code(string const &u8str,int colidx) const;
    TOK_CODE get_code(const char *u8str,int colidx) const;
    TOK_CODE get_defaultcode(int colidx) const{return get(colidx)->get_defaultcode();};

    //
    // Decode int to string
    //
    PSTRING decode(TOK_CODE code,int colidx)const;
    string decode8(TOK_CODE code,int colidx)const;

    //
    //Column size
    //
    size_t get_colsize(int colidx)const{return StructuredEncoder::get(colidx)->size();}
    
    
protected:
    
    //gets the encoder for column idx;
    static ColumnEncoder* get(int col_idx);
   
    static void push_encoder();    //pushes a new column encoder
    static void resize(unsigned int k);//sets the number of columns to k (erasing additional columns or allocating missing ones)
    
    
    
    //Creates a new structured encoders for with a phrase structure symbols+ a word column encoder.
    StructuredEncoder();
    ~StructuredEncoder();
    void clear();
    
    
private:
    
    static bool exists;
    static StructuredEncoder *instance;
    
    vector<ColumnEncoder*> encoder_list;
};


/////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * This is a class for logging/debugging
 *
 */
class DebugLogger{

 public:

  // log priorities
  enum Priority{INFO,DDEBUG,WARNING,ERROR};

  static void Start(Priority minPriority, const string &logFile);
  static void Start(Priority minPriority, const char *logFile);

  static void Stop();

  static void Write(Priority priority, const string& utf8_message);
  static void Write(Priority priority, const wstring& utf32_message);

private:

    DebugLogger(Priority minPriority,string const &logfile);
  
    static bool exists;
    static DebugLogger *instance;
    
    ofstream    fileStream;
    Priority    minPriority;
 
    // names describing the items in enum Priority
    static const string PRIORITY_NAMES[];
};

#ifdef ENABLE_DEBUG_LOGGER
#define DLOGGER_START(MIN_PRIORITY, FILE) DebugLogger::Start(MIN_PRIORITY, FILE);
#define DLOGGER_STOP() DebugLogger::Stop();
#define DLOGGER_WRITE(PRIORITY, MESSAGE) DebugLogger::Write(PRIORITY, MESSAGE); 
#else 
#define DLOGGER_START(MIN_PRIORITY, FILE)
#define DLOGGER_STOP()
#define DLOGGER_WRITE(PRIORITY, MESSAGE)
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * This is a class for logging parsing statistics etc..
 *
 */
class ParseLogger{

 public:

  // log priorities
  enum Statistics{PREC,RECALL,FSCORE,TIME,LEN,NUNK,NSCORED};

  static void Start(const string &logFile,int N=7);
  static void Start(const char   *logFile,int N=7);

  static void Stop();

  static void Set(Statistics statname, float value);
  static void NextParse();

  static void StartTimer();
  static void StopTimer();
  
  static void AddOneScored();


private:
  
    static bool exists;
    static ParseLogger *instance;
    
    ofstream    fileStream;
    vector<float> statsline;
    size_t      N;
    double      btime;

    ParseLogger(int N,string const &logfile);
    double      get_time();
    void        reset();
    
    double total_tokens;
    double total_time;

    // names describing the items in enum Statistics
    static const string STATISTICS_NAMES[];

};


#ifdef ENABLE_PARSE_LOGGER
#define PLOGGER_START(FILE) ParseLogger::Start(FILE,7);
#define PLOGGER_STOP() ParseLogger::Stop();
#define PLOGGER_SET(VARNAME,VALUE) ParseLogger::Set(VARNAME,VALUE);
#define PLOGGER_WRITE() ParseLogger::NextParse(); 
#define PLOGGER_START_TIMER() ParseLogger::StartTimer(); 
#define PLOGGER_END_TIMER() ParseLogger::StopTimer(); 
#define PLOGGER_ADD_ITEM() ParseLogger::AddOneScored();
#else 
#define PLOGGER_START(FILE)
#define PLOGGER_STOP()
#define PLOGGER_SET(VARNAME,VALUE)
#define PLOGGER_WRITE()
#define PLOGGER_START_TIMER() 
#define PLOGGER_END_TIMER()
#define PLOGGER_ADD_ITEM()
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * This is a class for logging learning statistics of an epoch based learner
 *
 */
class LearnLogger{

 public:

  //log priorities
  enum Statistics{EPOCH,TRAINP,TRAINR,TRAINF,DEVP,DEVR,DEVF,MODEL_SIZE,LOSS};

  static void Start(const string &logFile,int N=9);
  static void Start(const char   *logFile,int N=9);

  static void Stop();

  static void Set(Statistics statname, float value);
  static void NextEpoch();
  
private:
  
    static bool exists;
    static LearnLogger *instance;
    
    ofstream    fileStream;
    vector<float> statsline;
    size_t      N;

    LearnLogger(int N,string const &logfile);
    void        reset();
    int epoch;

    // names describing the items in enum Statistics
    static const string STATISTICS_NAMES[];

};


#ifdef ENABLE_LEARN_LOGGER
#define LLOGGER_START(FILE) LearnLogger::Start(FILE);
#define LLOGGER_STOP() LearnLogger::Stop();
#define LLOGGER_SET(VARNAME,VALUE) LearnLogger::Set(VARNAME,VALUE);
#define LLOGGER_SET_TRAIN(P,R,F) LearnLogger::Set(LearnLogger::TRAINP,P);LearnLogger::Set(LearnLogger::TRAINR,R);LearnLogger::Set(LearnLogger::TRAINF,F);
#define LLOGGER_SET_DEV(P,R,F) LearnLogger::Set(LearnLogger::DEVP,P);LearnLogger::Set(LearnLogger::DEVR,R);LearnLogger::Set(LearnLogger::DEVF,F);
#define LLOGGER_WRITE() LearnLogger::NextEpoch(); 
#else 
#define LLOGGER_START(FILE)
#define LLOGGER_STOP()
#define LLOGGER_SET(VARNAME,VALUE)
#define LLOGGER_WRITE()
#define LLOGGER_SET_TRAIN(P,R,F) 
#define LLOGGER_SET_DEV(P,R,F)
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////


#endif 

