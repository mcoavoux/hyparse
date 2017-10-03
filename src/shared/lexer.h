#ifndef LEXER_H
#define LEXER_H

#include "globals.h"
#include "str_utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <list>

/*
 * This module implements several string manipulation procedures and lexing utilities
 * The lexing module requires the encoder to be loaded in order to yield meaningful operations
 */

using namespace std;

/*******************************************
 * Module dedicated to lexing at parse time
 ******************************************/

/**
 * The input token class encodes a token output by the lexer as a vector of integers
 */
class InputToken{
  
public:
    static const wchar_t CPD_SEP = L'_';
    
    friend class InputDag;
    
    /**
     * Builds an input token from a vector of strings and an encoder
     */
     InputToken();//NULL (dummy word) token
     InputToken(const vector<PSTRING> &str_fields,int i, int j);
     InputToken(const InputToken &other);
    
    /**
     * Returns the non encoded string for this token wordform 
     * (may differ from the encoded one in case of unknowns) 
     */
    const PSTRING& get_wordform()const{return wordform;}
    
    /**
     * Gets the integer value of field idx in the token
     */
    TOK_CODE value_at(unsigned int field_idx)const{return fields[field_idx];}
    TOK_CODE operator[](unsigned int field_idx)const{return fields[field_idx];}

    /**
     * Gets the integer value of the pos category in the token
     */
    TOK_CODE get_catcode()const{return gram_code;}
    
    /**
     * Gets the integer value of the word field in the token
     */
    TOK_CODE get_wordcode()const{return fields[0];}

    /**
     * The number of fields of this token
     */
    size_t size()const{return fields.size();}
    
    /**
     * Pretty prints the input token
     */
    friend ostream& operator<< (ostream &os,const InputToken &token);

    bool operator==(InputToken const &other)const;
    
    bool is_null_token()const{return idx == -1;}
private:
    
    int idx, jdx;
    PSTRING wordform;
    TOK_CODE gram_code;
    vector<TOK_CODE> fields;
    
};

/**
 * This class manages an input sequence for parsing purposes.
 * It can dynamically reassign the tag field value to tokens (e.g. for tagging) or the word field token (for segmentation)
 * It manages a limited form of DAG representation convenient for transition based parsing.
 * It acts as a memory manager for dynamically allocated tokens
 */

//IMPL NOTE, this data structure does not implement an usual DAG -> edges are copied instead of being retrieved and hence may be duplicated
// under hyp. that this policy is more efficient given the parsing alg. the data structure acts as a garbage collector...
class InputDag{
public:
    
    InputDag(){};
    InputDag(InputDag const &other);
    ~InputDag();
    
    const InputToken* merge_left(TOK_CODE pred_pos,InputToken const *left,InputToken const *right);//merges two tokens and sets them a new joint pos,left token is head
    const InputToken* merge_right(TOK_CODE pred_pos,InputToken const *left,InputToken const *right);//merges two tokens and sets them a new joint pos, right token is head
    
    const InputToken* shift_token(TOK_CODE pred_pos,int J);//shifts the token at index J into the dag, sets it a tag and returns a pointer to it.
    const InputToken* shift_token(int J);//shifts the token at index J into the dag, copies its tag and returns a pointer to it.
    
    const InputToken* operator[](unsigned int idx)const;//returns a pointer to input token at index idx in the queue
    const InputToken* at(unsigned int idx)const;//returns a pointer to input token at index idx in the queue (with bounds checking)

    void push_back(InputToken *token);//adds a token to the queue
    
    size_t size()const;                     //the size of the input queue
    void reset(); //empties the dag and resets the structure for a fresh new parse
    void clear(); //this empties everything dag and queue
    bool empty();//returns true if input_queue is empty
    friend ostream& operator<< (ostream &os,const InputDag &input_sequence);
    
private:
    
    vector<InputToken*>   input_queue;
    list<InputToken*>   input_dag;
};



/**
 * The lexer class is a default lexer that processes input prior to parsing
 * one line = one token with tab separated fields (left to free interpreation)
 * first column = actual token
 * white line = end of sentence
 * Note that the lexer returns input tokens, -> integer coded tokens
 */
class AbstractLexer{

 public:
  virtual ~AbstractLexer(){};
  virtual bool next_sentence(istream &inFile,InputDag &tokens_read) = 0;
  virtual bool next_batch(size_t batch_size,istream &input_source,vector<InputDag> &input_batch);
  virtual void process_file(istream &inFile,vector<InputDag> &sentences);
  virtual bool skip_header(istream &inFile)=0;
};


class TbkLexer : public AbstractLexer{
    
public:

    /**
     * Builds a new lexer
     */
    TbkLexer(){
        this->arity = 0;
    }
    
    bool skip_header(istream &inFile);
    

    /**
     * This fills in the tokens_read vector with tokens read.
     * Note any content in tokens_read is destroyed
     * @return true if the stream is still readable, false if end of stream is reached.
     */
    bool next_sentence(istream &inFile,InputDag &tokens_read);

protected:
    size_t arity ;
    str::SimpleTokenizer stok;
};


class MarmotLexer : public AbstractLexer{

public:

  MarmotLexer(vector<wstring> const &ttd);

  bool skip_header(istream &inFile);
  bool next_sentence(istream &inFile,InputDag &tokens_read);
  //bool next_batch(size_t batch_size,istream &input_source,vector<InputDag> &input_batch);
  static void read_ttd(vector<wstring> &ttd,string const &);

private:
  size_t arity;
  str::SimpleTokenizer linetok;
  str::SimpleTokenizer featurestok;
  str::SimpleTokenizer attvaltok;
  unordered_map<wstring,unsigned> ttd_map;//maps features to token positions
};




#endif
