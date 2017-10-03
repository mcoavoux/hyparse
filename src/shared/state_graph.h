#ifndef STATE_GRAPH_H
#define STATE_GRAPH_H

#include "globals.h"
#include "lexer.h"
#include "sparse_encoder.h"
#include "srgrammar.h"
#include <vector>
#include <deque>

class ParseState{

public:
      
  friend class ParseDerivation;
    
  //state allocators & parse actions
    
  static ParseState* init_state();//graph init state
    
  ParseState* shift(SrGrammar const &grammar,InputDag &input_sequence,float prefix_weight);//shift without retagging
  ParseState* shift_tag(TOK_CODE tag_symbol,SrGrammar const &grammar,InputDag &input_sequence,float prefix_weight);//shift and tags
  ParseState* reduce_left(TOK_CODE rsymbol,SrGrammar const &grammar,float prefix_weight);
  ParseState* reduce_right(TOK_CODE rsymbol,SrGrammar const &grammar,float prefix_weight);
  ParseState* reduce_unary(TOK_CODE rsymbol,SrGrammar const &grammar,float prefix_weight);
  ParseState* ghost_reduce(SrGrammar const &grammar,float prefix_weight);

  //state encoding : clears and fills the xvec with the state X encoding using spencoder
  void encode(vector<unsigned int> &xvec,
              SparseEncoder &spencoder,
              StateSignature const &sig);//prediction sparse encoding
    
  void encode(SparseFloatVector &xvec,
              SparseEncoder &spencoder,
              StateSignature const &sig);//update sparse encoding
    
  //generates the best parse tree from this parse state
  AbstractParseTree* make_best_tree(InputDag &input_sequence);
   
  //basic accessors
  bool is_init()const;
  bool is_success(int eta,size_t N)const;
  float weight()const;
  const ParseState* stack_predecessor()const{return stack_prev;}
  unsigned int getJ()const{return J;}
  

    

  void get_signature(StateSignature &sig,InputDag const *input_sequence,unsigned int input_length);
  const ParseAction& get_incoming_action()const{return incoming_action;};
  
  //Equivalence for purpose of hamming distance computation
  bool is_hamming_equivalent(ParseState &other,InputDag const *input_sequence,unsigned int N);
    
  friend ostream& operator<<(ostream &os,ParseState const &state);
  
  //returns the span of an item
  pair<int,int> get_span()const{return make_pair(I,J);}

  TOK_CODE get_top_symbol()const{return top;} //@@m function added
  const ParseState* history_predecessor()const{return history_prev;} //@@m function added

protected:
    
  ParseState();
  ParseState(const ParseState &other);
  ParseState& operator=(const ParseState &other);
    
  int I,J;
  TOK_CODE top,left,right;
  InputToken const *htop;
  InputToken const *hleft;
  InputToken const *hright;
  InputToken const *left_corner;
  InputToken const *right_corner;

  float prefix_weight;

  ParseAction incoming_action;
  ParseState *stack_prev;
  ParseState *history_prev;
    
  void make_signature(StateSignature &signature,InputDag const *input_sequence,unsigned int input_length);  //This generates a state signature from this state

    
private:
  //Extracts a linked list of ParseStates from a tree and returns the last one
  static ParseState* from_tree(AbstractParseTree const *root,SrGrammar const &grammar,InputDag &input_sequence,bool tagger_deriv);
  static ParseState* from_binarytreeR(AbstractParseTree const *root,ParseState *current,SrGrammar const &grammar,InputDag &input_sequence,bool tagger_deriv);
    
};


/**
 * This is a convenience class for manipulating derivations (sequences of parse states) easily
 */
class ParseDerivation{
    
public:
    
    ParseDerivation(ParseState *last);       //generates a derivation from a parse state by backtracking its best-first history
    ParseDerivation(AbstractParseTree const *root,SrGrammar const &grammar,InputDag &input_sequence,bool tagger_deriv=false);//generates a derivation from a parse tree
    
    ParseDerivation(ParseDerivation const &other);
    ~ParseDerivation();
    
    AbstractParseTree* as_tree(InputDag const &input_sequence,bool morph=false)const;

    float weight_at(unsigned int idx)const;
    ParseState* operator[](unsigned int idx)const;

    
    //computes the hamming distance between two derivations up to timestep t.
    //In case derivations have different sizes, the difference in size is added to the hamming distance
    unsigned int hamming(ParseDerivation const &other,unsigned int t,InputDag const *input_sequence);
    
    //This reweights a derivations given a model and an encoder, any existing weights are discarded.
    void reweight_derivation(SparseEncoder &spencoder,
                             SparseFloatMatrix const &model,
                             SrGrammar const &grammar,
                             InputDag const *input_sequence,
                             size_t N);
    
    friend ostream& operator<<(ostream &os,ParseDerivation const &derivation);

    size_t size()const{return derivation.size();};
    
    
private:
    deque<ParseState*> derivation;
};

////////////////////////////////////////////////////////////////////////////

/**
 * This is a Beam suitable for building a TSS search space.
 */
class TSSBeam{
    
public:
    //typedefs a pseudo state being candidate for entering the beam
    typedef tuple<ParseState*,ParseAction,float> CANDIDATE;
    
    
    TSSBeam();
    TSSBeam(size_t beam_size,SrGrammar const &grammar);
    TSSBeam(TSSBeam const &other);
    TSSBeam& operator=(TSSBeam const &other);
    ~TSSBeam();
    
    //Finds out the parses once parsing has completed
    bool has_best_parse()const;
    size_t num_parses()const;

    ParseState* best_success_state();
    ParseState* kth_success_state(int k);

    AbstractParseTree* best_parse(InputDag const &input_sequence);
    AbstractParseTree* kth_best_parse(InputDag const &input_sequence,unsigned int k);//returns the kth best parse from this beam.
    
    //Function for adding new states to the beam
    void push_candidates(ParseState* from_state,vector<float> &y_scores); //push new candidates into the candidate pool
    //Selects beam size K-best candidates from the candidate pool and pushes a new beam top to be used for the next round
    void next_step(SrGrammar const &grammar,InputDag &input_sequence,size_t N);
    
    //reinits the beam for a new parse
    void reset();
    
    //Beam top accessors
    size_t top_size()const;    //Returns the size of the beam top
    ParseState* operator[](unsigned int kth);//returns the kth element from the top of the beam.
    
    //Accessors of the beam at step t.
    bool sig_in_kth_beam(StateSignature const &sig,unsigned int t)const;
    ParseState* best_at(unsigned int t);
    ParseState* kth_at(unsigned int t,unsigned int k);

    size_t size_at(unsigned int t)const;//Size ot the beam at time t

    inline size_t size()const{return states.size();}    //current number of time steps
    inline size_t maxsize()const{return beam_size;}     //the max number of parse states at some time step
    
    
    //tests if 2 sigs are equivalent for the purpose of checking if a state is in the beam.
    bool weak_equivalence(StateSignature const &sigA,StateSignature const &sigB)const;

    //debug functions
    void display_kbest_derivations();//debug method displaying the k-best derivations
    void dump_beam(); //dumps the whole beam on stdout
    void set_size(size_t size){beam_size = size;} //for debug only, remove later on
    
    
private:
    bool succ_sorted = false;
    size_t beam_size;  //max number of elements in the beam at time t
    size_t A;          //num actions
    int timestep;
    vector<CANDIDATE> candidates_buffer; //potential next states
    bool morph = false;
    
    vector<vector<ParseState*> > states; //all states stored in the beam
    vector<ParseState*> success_states;  //states that are completed
    vector<ParseAction> action_vector;   //an ordered vector of parsing actions
    
    
        
};



#endif
