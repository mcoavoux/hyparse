#ifndef SRGRAMMAR_H
#define SRGRAMMAR_H

#include "treebank.h"
#include "globals.h"
#include "utilities.h"

#include <vector>
#include <limits>
/**
 * The grammar module defines the grammar. The grammar is responsible for providing the transition function
 * as a set of potential parsing actions for a given state signature. 
 *
 * The grammar is supposed to be inferred from a treebank.
 */
////////////////////////////////////////////////////////////////////////////////////////////////////
//State Signature

static InputToken NULL_TOKEN = InputToken();

class StackItem{
public:
    TOK_CODE top,left,right;
    const InputToken *htop;
    const InputToken *hleft;
    const InputToken *hright;
    const InputToken *left_corner;
    const InputToken *right_corner;
    
    bool is_init_state()const{return *htop == NULL_TOKEN;}
    
    bool operator==(StackItem const &other)const{
        return top == other.top && left == other.left && right == other.right
        && *htop == *other.htop && *hleft == *other.hleft && *hright == *other.hright
        && *left_corner == *other.left_corner && *right_corner == *other.right_corner;
    }
    bool operator!=(StackItem const &other)const{return !(*this == other);}
    
    friend ostream& operator<<(ostream &os, StackItem const &stack);
    
};


class StateSignature{
 
 public:
  static const unsigned int STACK_LOCALITY = 3; //max local stack view size

    
  StackItem stack[STACK_LOCALITY];
  size_t stack_size; //actual stack size

  int J;
  InputDag const *input_sequence;
  int N;

  //Equivalence relation for signatures
  bool operator==(StateSignature const &other)const;
  bool operator!=(StateSignature const &other)const;

  friend ostream& operator<<(ostream &os, StateSignature const &sig);

    
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//Parser Actions

//Encodes actual actions of the form S,RL(X),RR(X),RU(X),GR
struct ParseAction{

  enum ActionType{SHIFT,RL,RR,RU,GR,NULL_ACTION};
  
  ActionType action_type;     //the action
  TOK_CODE tok_code;          //X code for RU,RR,RL
  unsigned int action_code;   //index code given by the grammar
    
  ParseAction(){
        action_code = 0;
        tok_code    = 0;
        action_type = NULL_ACTION;
  }
    
  ParseAction(ParseAction const &other);
  ParseAction& operator=(ParseAction const &other);
    
  unsigned int get_action_code()const{return action_code;}
  unsigned int get_code()const{return action_code;}
  
  bool operator == (ParseAction const &other)const{return action_code == other.action_code;}
  bool operator != (ParseAction const &other)const{return (! ( (*this) == other));}

  friend ostream& operator<<(ostream &os,ParseAction const &action);
    
};

string to_string(ParseAction const &action);



//inner class for grammar construction
class ActionComparator{
    
public:
    bool operator()(ParseAction const &a, ParseAction const &b)const{
        if(a.tok_code==b.tok_code){return a.action_type < b.action_type;}
        else{return a.tok_code < b.tok_code;}
    }
};


//This class provides a unique action code to (Abstract Action, Symbol) couples
// in constant time.
class ActionCoder{
    
public:
    
    typedef unsigned int ACTION_INDEX;
    
    ActionCoder(){};
    ActionCoder(vector<ParseAction> &actions);//encodes a vector of actions with action indexes,
    ActionCoder(ActionCoder const &other);
    ActionCoder& operator=(ActionCoder const &other);
    
    //for convenience, assigns actioncodes to the actual actions passed as argument.
    //orders of actions in the vector is irrelevant for the coding.
    void make_encoding(vector<ParseAction> &actions);

    unsigned int get_action_index(ParseAction::ActionType type,TOK_CODE tok)const;
    unsigned int get_action_index(ParseAction const &action)const;
    const ParseAction& get_shift_action()const{return actionmap.get_label(shift_index);};
    const ParseAction& get_ghost_action()const{return actionmap.get_label(ghost_index);};
    
    const ParseAction& get_action(ParseAction::ActionType type,TOK_CODE tok)const;
    const ParseAction& get_action(ACTION_INDEX idx)const;
    size_t nactions()const {return actionmap.nlabels();}
    size_t nsymbols()const {return symbolmap.nlabels();}
    void clear();
    
private:
    size_t get_actioninternalposition(ParseAction::ActionType type,TOK_CODE tok)const;
    
    Bimap<TOK_CODE> symbolmap;
    Bimap<ParseAction> actionmap;
    vector<ParseAction> allactions;
    
    //special cases, record action index
    unsigned int shift_index=1;
    unsigned int ghost_index=0;
    
};





////////////////////////////////////////////////////////////////////////////////////////////////////
//Utility class for the grammar

/**
 * This class encodes which grammatical symbols are valid reduction symbols for temporaries only
 */
class TemporaryReductionTable{
    
public:
    
    TemporaryReductionTable(){};
    
    /**
     * Creates a new reduction table for all temporary categories
     */
    TemporaryReductionTable(vector<PSTRING> const &all_categories);
    TemporaryReductionTable(const TemporaryReductionTable &other);
    TemporaryReductionTable& operator= (const TemporaryReductionTable &other);
    
    /**
     * Tells if this child can be reduced by this root
     * returns false if child is temp but root is an invalid reduction
     * returns false if the child is not temp (!)
     * otherwise returns true.
     */
    inline bool is_valid_temporary_reduction(TOK_CODE child,TOK_CODE root)const{return table[child][root];}
    
    bool operator==(TemporaryReductionTable const &other)const;
    
private:
    void init_table(vector<PSTRING> const &all_categories);
    void add_valid_reduction(TOK_CODE child,TOK_CODE root);

    vector<vector<bool>> table;
};

/* DEPRECATED
class ReverseActionTable{
    
public:
    ReverseActionTable();
    ~ReverseActionTable(){delete reductions;}
    ReverseActionTable(vector<PSTRING> &allnonterminals);
    
    ReverseActionTable (ReverseActionTable const &other);
    ReverseActionTable& operator=(ReverseActionTable const &other);
    
    
    const ParseAction& operator()(ParseAction::ActionType,TOK_CODE code)const;
    void add_action(ParseAction &action);
    
    
protected:
    size_t N;
    ParseAction shift;
    ParseAction ghost;
    vector<ParseAction> *reductions;
};
*/

////////////////////////////////////////////////////////////////////////////////////////////////////
//Grammar

//The grammar fills two roles:
//  1) Provides means to select valid actions while parsing (automaton)
//  2) Manages parse actions and their conversion to vector indexes (weights)


class SrGrammar{

public:
    
    static TOK_CODE NULL_SYMBOL;
 
/*
  Deprecated
  SrGrammar(vector<PSTRING> const &nonterminals,vector<PSTRING> const &axioms, float semiringzero= - std::numeric_limits<float>::infinity());//Builds a grammar automaton from the list of symbols
  SrGrammar(Treebank const &treebank,float semiringzero= -std::numeric_limits<float>::infinity());    //Extracts the grammar automaton from the treebank
 */
  

  SrGrammar();
  SrGrammar(SrGrammar const &other);
  SrGrammar(string const &filename);
  SrGrammar& operator=(SrGrammar const &other);
    
  //Extracts the grammar automaton directly from the treebank
  SrGrammar(Treebank const &train_bank,
            Treebank const &dev_bank,
            bool tagger=false,
            float semiringzero= - std::numeric_limits<float>::infinity());
    
  /**
   * Given a state signature, sets the weights of invalid actions to SEMI_RING_ZERO
   * The weights in the weight vector are supposed to be ordered with the order 
   * returned by the get_ordered_actions method
   */
  void select_actions(vector<float> &weight_vector,ParseAction const &prev_action,StateSignature const &sig) const;
  void select_actions(vector<bool> &action_vector,ParseAction const &prev_action,StateSignature const &sig) const;  // @@m
  size_t num_actions()const{return actionencoder.nactions();}
  bool has_morph()const{return has_morpho;}
    
/* Deprecated
    vector<ParseAction> get_ordered_actions()const;
  size_t num_actions()const{return all_actions.size();}
*/
    
  //returns a reference to the action at position idx in the action_vector
  void get_ordered_actions(vector<ParseAction> &action_vector)const;

  inline const ParseAction& operator[](unsigned int idx)const{return actionencoder.get_action(idx);}
  //returns the index of this action in the action vector
  unsigned int get_action_index(ParseAction const &action)const{return actionencoder.get_action_index(action);};
  //Returns a properly encoded parse action object given the action type and tok code
  const ParseAction& get_action(ParseAction::ActionType atype,TOK_CODE symbol_code)const;
  inline const ParseAction& get_shift_action()const{return actionencoder.get_shift_action();}; //for shift action when there is only one shift
  inline const ParseAction& get_ghost_action()const{return actionencoder.get_ghost_action();}; //for ghost reduce actions
  
  //Loads/saves the grammar from file (and erases any existing content)
  void load(string const &filename);
  void load_grammar(string const &filename);
  void save(string const &filename)const;
    
  //Tests equality
  bool operator==(SrGrammar const &other)const;

  //Prints a quick summary of this grammar
  friend ostream& operator<<(ostream &stream,SrGrammar const &grammar);
  //Prints the action ordering
  ostream& print_action_vec(ostream &os);
    
  bool has_tranformations()const;//says if this grammar contains transformed symbols
    
private:

  void load_actions(string filename);
  void save_actions(string filename)const;

  // Says if this action code is valid given current state
   bool is_valid_action(ParseAction const &prev_action,ParseAction const &action,StateSignature const &sig)const;
   bool is_valid_action_with_tagger(ParseAction const &prev_action,ParseAction const &action,StateSignature const &sig)const;

   //Says if a parse category is temporary
   bool is_temporary(TOK_CODE cat)const;
   bool is_terminal(TOK_CODE cat)const;
   bool is_axiom(TOK_CODE cat)const;
    
   //Says if root parse category is a valid reducer for head child parse category
   bool is_valid_temporary_reduction(TOK_CODE cat_child,TOK_CODE cat_root)const;

   //grammar compilation subroutines
   //Extracts a reduced grammar automaton from the treebank
   void categorize_symbols(vector<PSTRING> const &all_nonterminals,
                           vector<PSTRING> const &all_axioms,
                           vector<PSTRING> const &all_tags);
   void compile_grammar(Treebank const &train_bank,Treebank const &dev_bank,bool tagger);
   ParseAction make_action(ParseAction::ActionType action_type, TOK_CODE tok_code,unsigned int action_code)const;
   void make_actions(Treebank const &train_bank,Treebank const &dev_bank);
   void readoff_actions(AbstractParseTree const *root,std::set<ParseAction,ActionComparator> &all_actions);
    

  /* TO BE REMOVED
   void compile_grammar(Treebank const &treebank);    //Extracts the grammar automaton from the treebank
   void compile_grammar(Treebank const &train_bank,Treebank const &dev_bank,bool tagger);    //Extracts the grammar automaton from the treebank
   void compile_grammar(vector<PSTRING> const &all_nonterminals,vector<PSTRING> const &axioms);    //Extracts the grammar automaton from the list of non terminals
   void compile_grammar_with_tagger(vector<PSTRING> const &all_nonterminals,vector<PSTRING> const &tags,vector<PSTRING> const &axioms);    //Extracts the grammar automaton from the list of non terminals
   void make_actions(vector<PSTRING> const &all_nonterminals);
   void make_actions(vector<PSTRING> const &all_nonterminals,vector<PSTRING> const &ordered_tags);
   ParseAction make_action(ParseAction::ActionType action_type, TOK_CODE tok_code,unsigned int action_code)const;
   void make_reverse_action_vector(vector<ParseAction> const &all_actions);
   */
    
    void clear();

   //fields
    float SEMI_RING_ZERO;  //by default zero for real and tropical semi rings, can be modified for prob semi rings
    
    //symbol encoding;
    vector<bool> tags;
    vector<bool> temporaries;
    vector<bool> axioms;
    vector<bool> allnonterminals; //(includes temporaries)
    bool has_morpho = false;
    TemporaryReductionTable valid_reductions;//Reduction Table (for temporaries, lists all potential reduced categories)
    vector<PSTRING> _nonterminals;
    vector<PSTRING> _axioms;
    vector<PSTRING> _tags;
    //action encoding
    ActionCoder actionencoder;//maps actions to contiguous dense Y indexes
    Bimap<TOK_CODE> symbolencoder;//maps grammatical symbols to contiguous dense Y indexes
    
   /*
    //size_t Nt;     //number of non terminals
    //size_t Nshifts;  //number of tags
    //size_t Na;//size of all_actions
    vector<PSTRING> ordered_nonterminals;         //Keeps an ordered list of nonterminals for serialization purposes.
    vector<PSTRING> ordered_axioms;               //Keeps an ordered list of nonterminals for serialization purposes.
    vector<PSTRING> ordered_tags;                 //Keeps an ordered list of tags for serialization purposes.
    */
    
    /* TO BE REMOVED
    vector<unsigned int> reverse_action_vector;
    ReverseActionTable reverse_action_table;
    vector<ParseAction> all_actions; //Action list
    */

    // @@m : the following is for DynamicOracle class
    friend class DynamicOracle;
    void get_temporaries_codes(vector<TOK_CODE> &tmp){
        tmp.clear();
        for (int i = 0; i < temporaries.size(); i++)
            if (temporaries[i])
                tmp.push_back(symbolencoder.get_label(i));
    }
    void get_nontemporaries_codes(vector<TOK_CODE> &tmp){
        tmp.clear();
        for (int i = 0; i < allnonterminals.size(); i++)
            if (allnonterminals[i] && ! temporaries[i])
                tmp.push_back(symbolencoder.get_label(i));
    } // @@m
  
};



#endif
