#ifndef TREEBANK_H
#define TREEBANK_H

#include<set>
#include<unordered_set>
#include<tuple>

#include "globals.h"
#include "lexer.h"
#include "utilities.h"

using namespace std;

//RESERVED SYMBOLS
static const PSTRING TMP_CODE   = L":";
static const PSTRING UNARY_CODE = L"@"; 
static const PSTRING SEP_CODE   = L"/";  //decorator code (for heads and tmp annotations)
static const PSTRING UTMP_CODE  = L"TTT:";//unique temporary code

class AbstractParseTree{

 public:
  friend void destroy(AbstractParseTree*); 
  friend void unbinarize(AbstractParseTree*);
  friend void unary_closure(AbstractParseTree*,bool);

  //Tree modifications
  virtual void add_child(AbstractParseTree *child) = 0;
  virtual AbstractParseTree* get_child_at(unsigned int idx)const = 0;

  virtual bool is_leaf()const = 0;
  virtual size_t arity()const = 0;

  virtual bool is_head()const{return this->head;}
  virtual void set_head(){this->head = true;}

  virtual void set_label(PSTRING const &label) = 0;
  virtual const PSTRING& get_label()const = 0;
  virtual const PSTRING& get_clabel_at(unsigned int idx)const = 0;

  friend ostream& operator<<(ostream &ostream,AbstractParseTree const &root);

 protected:
  bool head = false;
  virtual ~AbstractParseTree(){};
};

class ParseNode : public AbstractParseTree{

public:

  //this class is friend of transformation methods
  friend AbstractParseTree* clone(const AbstractParseTree*);
  friend unsigned int flajolet_decorate(AbstractParseTree *);
  friend void undecorate(AbstractParseTree *);
  friend void parent_decorate(AbstractParseTree*,PSTRING const &);
  friend void head_markovize(AbstractParseTree*, PSTRING const &, unsigned int);
  friend void unbinarize(AbstractParseTree *);
  friend void unary_closure(AbstractParseTree*,bool);
  friend void unpack_unaries(AbstractParseTree*);

  
  ParseNode(PSTRING const &label,bool head=false);
 
  void add_child(AbstractParseTree *child){this->children.push_back(child);}
  AbstractParseTree* get_child_at(unsigned int idx)const {return children[idx];}

  virtual bool is_leaf()const{return false;};
  size_t arity()const{return children.size();}
    
  void set_label(PSTRING const &label){this->label = label;}
  const PSTRING& get_label()const{return this->label;}
  const PSTRING& get_clabel_at(unsigned int idx)const{return children[idx]->get_label();}


private:

  PSTRING label;
  vector<AbstractParseTree*> children;

  //copy constructor
  ParseNode(const ParseNode &other);

  //disconnects this node from its children
  void clear_children(){children.clear();}

};

class ParseToken : public AbstractParseTree{
 public:

  friend int tree_yield(AbstractParseTree const *root,InputDag&,int);

  ParseToken(vector<PSTRING> const &tok_fields,bool head=false);
  ParseToken(const InputToken &other,bool head=false);
  ParseToken(const ParseToken &other);


  void add_child(AbstractParseTree *child);

  AbstractParseTree* get_child_at(unsigned int idx)const;

  size_t arity()const{return 0;}
  virtual bool is_leaf()const{return true;}
  
  void set_label(PSTRING const &label){this->tok_fields[1] = label;}

  const PSTRING& get_label()const{return this->tok_fields[1];}
  const PSTRING& get_clabel_at(unsigned int idx)const{return tok_fields[idx];}

  size_t lexfields_size()const{return tok_fields.size();}
    
    
 private:
  vector<PSTRING> tok_fields;
    
};


//says if a symbol is temporary or not
bool is_temporary(PSTRING const &symbol);
bool is_unary_closed(PSTRING const &symbol);


///RECURSIVE TREE ALGORITHMS///

//ENCODING
void encode_tree(AbstractParseTree *root);                                      //updates the encoder by encoding tree leaves
//void struct_encode_tree(AbstractParseTree *root);                                      //updates the encoder by encoding tree leaves
void get_symbols(AbstractParseTree *root,unordered_set<PSTRING> &symbol_set);   // updates non terminal symbol_set w/ the set of symbols from this tree
void get_tags(AbstractParseTree *root,unordered_set<PSTRING> &tag_set);           // appends word occurrences to the wordlist vector
void get_wordlist(AbstractParseTree *root,vector<PSTRING> &wordlist);           // appends word occurrences to the wordlist vector
PSTRING get_axiom(AbstractParseTree *root);

//void smooth(AbstractParseTree *root,unordered_set<PSTRING> const &known_vocab); //performs lexical smoothing
int tree_yield(AbstractParseTree const *root,InputDag &result, int idx=0);

//I/O
bool get_tree(istream &in,AbstractParseTree *&root ,int required_arity = -1);//reads a tree in Pseudo XML format returns true if the stream is still readable
bool head_check(AbstractParseTree *root,bool isroot = true);//checks wether each rule is properly headed
ostream& ptb_flush(ostream &os,AbstractParseTree const *root,InputDag const &input_sequence,int &idx);
ostream& native_flush(ostream &os,AbstractParseTree const *root,InputDag const &input_sequence,int &idx,int padding = 0);

//COUNTING
void count_words(AbstractParseTree *root,Counter<PSTRING> &counts);   //counts occurrences of word symbols
void count_symbols(AbstractParseTree *root,Counter<PSTRING> &counts); //counts occurrences of non terminal symbols

//evalb style comparison function
typedef pair<unsigned int, unsigned int> SPAN_T;
typedef tuple<unsigned int, unsigned int,PSTRING> TRIPLET_T;
SPAN_T extract_triples(AbstractParseTree const *root,set<TRIPLET_T> &triples,unsigned int input_idx=0);
//computes <precision,recall,fscore> for this pair of trees
tuple<float,float,float> compare(AbstractParseTree const *rootA,AbstractParseTree const *rootB,bool evalb=true);
tuple<float,float,float> compare_evalb(AbstractParseTree const *rootA,AbstractParseTree const *rootB);

//Deep Copy algorithm (non destructive op allocating memory)
AbstractParseTree* clone(const AbstractParseTree *root);

//Transformation algorithms (destructive ops not allocating memory !)
void head_markovize(AbstractParseTree *root,PSTRING const &root_sym=L"",unsigned int order=0);
void unbinarize(AbstractParseTree *root);

void unary_closure(AbstractParseTree *root,bool merge_tags=false);
void unpack_unaries(AbstractParseTree *root);

void parent_decorate(AbstractParseTree *root,PSTRING const &parent=L"TOP");
unsigned int flajolet_decorate(AbstractParseTree *root); //P. Flajolet numbering.
void undecorate(AbstractParseTree *root);

void destroy(AbstractParseTree *root);

////////////////////////////////////////////////////////////////////////////////////////////
//Treebank

/**
 * A treebank is a convenience wrapper for managing lists of trees.
 * It is also a tree storage container managing tree allocation.
 */
class Treebank{

 public:

  enum Decoration {NO_DECORATIONS,PARENT_DECORATION,FLAJOLET_DECORATION};

  Treebank();
  Treebank(size_t N);//allocates an empty treebank of size N
  Treebank(string const &filename);
 
  Treebank(Treebank const &treebank);
  Treebank& operator=(Treebank const &treebank);

  ~Treebank();

  void get_allsymbols(vector<PSTRING> &nonterminals,vector<PSTRING> &axioms,vector<PSTRING> &tags)const;
  void get_nonterminals(vector<PSTRING> &nonterminals,vector<PSTRING> &axioms)const;
  void get_axioms(vector<PSTRING> &nonterminals)const;
  void get_tagset(vector<PSTRING> &tags)const;
  void get_word_tokens(vector<PSTRING> &tokens)const;
    
  const AbstractParseTree* operator[](unsigned int idx)const{return tree_list[idx];}

  const AbstractParseTree* sample_tree()const;
  
  void shuffle();
  void sample_trees(Treebank &sampled, unsigned int N)const;   
  void split(Treebank &train,Treebank &dev)const;
  void fold(Treebank &train,Treebank &dev,unsigned k,unsigned Kfolds)const;

  void transform(Treebank &transformed,bool merge_tags=false,unsigned int markov_order=0,Decoration decoration=NO_DECORATIONS);
  void detransform(Treebank &detransformed,Decoration decoration=NO_DECORATIONS);
  void detransform(Decoration decoration=NO_DECORATIONS);//inplace detransformation

  tuple<float,float,float> evalb(Treebank const &other);

  void add_tree(AbstractParseTree *tree);
  void set_tree_at(AbstractParseTree *tree,int idx){tree_list[idx]=tree;};
  vector<PSTRING> update_trees(string const &filename);//returns the column header
  void update_encoder();
  void update_structured_encoder();
  void clear();

  size_t size()const{return tree_list.size();};

  friend ostream& operator<<(ostream &os,const Treebank &treebank);
  
  //names of the columns describing tokens
  vector<PSTRING> colnames;
    
 protected:

  vector<AbstractParseTree*> tree_list;
};

////////////////////////////////////////////////////////////////////////////////////////////
// I/O utilities

/**
 * This class encodes a simplified form of conllx graph for outputting 
 * projective untyped dependencies from a parse tree
 */
class ConllGraph{

public:
    //Creates a graph for a sentence of length N
    //By default it encodes a parse failure (all nodes linked to the root)
    ConllGraph(InputDag &input_sequence);
    
    //Creates a graph by converting tree rooted at root
    ConllGraph(AbstractParseTree const *root, InputDag  &input_sequence);

    void add_edge(int dep,int gov);
    
    friend ostream& operator<<(ostream &os,ConllGraph &graph);
    
protected:
    pair<int,int> tree_as_graph(AbstractParseTree const *root,int startIdx = 0);

    
    InputDag *input_sequence;
    vector<int> govRelation; // a vector of size N+1 -> each dep word is mapped to its unique governor
};


////////////////////////   OUTPUT STREAMS  //////////////////////

class ParserOutStream{
    
public:
    virtual void flush_parse(AbstractParseTree const *root,InputDag &input_sequence) = 0;
    virtual void flush_failure(InputDag &input_sequence) = 0;
    
    virtual void close() = 0;
    
    void set_active(bool flag){active=flag;};
    
protected:
    bool active;
    string filename;
};


class PennTreebankOutStream : public ParserOutStream {
    
public:
    PennTreebankOutStream(bool active = true);//cout stream
    PennTreebankOutStream(string const &filename);
    PennTreebankOutStream(PennTreebankOutStream const &other);
    PennTreebankOutStream& operator=(PennTreebankOutStream const &other);
    
    void flush_parse(AbstractParseTree const *root,InputDag &input_sequence);
    void flush_failure(InputDag &input_sequence);
    void close();
    
protected:
    ofstream os;
    bool stdout;
};

class ConllOutStream : public ParserOutStream{
    
public:
    ConllOutStream(bool active=false);
    ConllOutStream(string const &filename);
    ConllOutStream(ConllOutStream const &other);
    ConllOutStream& operator=(ConllOutStream const &other);
    void open(string const &filename);
    
    void flush_parse(AbstractParseTree const *root,InputDag &input_sequence);
    void flush_failure(InputDag &input_sequence);
    void close();
private:
    ofstream os;
    bool stdout;
};

class NativeOutStream :  public ParserOutStream{

public:
    
    NativeOutStream(bool active=false);//cout stream
    NativeOutStream(string const &filename);
    NativeOutStream(NativeOutStream const &other);//cout stream
    NativeOutStream& operator=(NativeOutStream const &other);

    void flush_parse(AbstractParseTree const *root,InputDag &input_sequence);
    void flush_failure(InputDag &input_sequence);
    void close();

private:
    ofstream os;
    bool stdout;
};



// Wraps a collection of output streams
class MultiOutStream{
    
public:
    
    void flush_parse(AbstractParseTree const *root,InputDag &input_sequence);
    void flush_failure(InputDag &input_sequence);
    
    void setPennTreebankOutfile(string const &filename);
    void setConllOutfile(string const &filename);
    void setNativeOutfile(string const &filename);
    void close();
    
private:
    PennTreebankOutStream ptb_out;
    ConllOutStream conll_out;
    NativeOutStream native_out;
    
};







#endif
