#include "srgrammar.h"
#include "globals.h"
#include "str_utils.h"
#include <cassert>
#include <algorithm>

TOK_CODE SrGrammar::NULL_SYMBOL = IntegerEncoder::get()->encode(L"$$UNK$$",IntegerEncoder::PS_COLCODE);


ostream& operator<<(ostream &os,ParseAction const &action){
    
    IntegerEncoder *enc = IntegerEncoder::get();
    switch (action.action_type){
        case ParseAction::SHIFT:
            if (action.tok_code == 0){return os << "S:"<< action.action_code;}
            else{return os << "S("<< enc->decode8(action.tok_code) << "):" << action.action_code;}
        case ParseAction::RL:
            return os << "RL("<< enc->decode8(action.tok_code) <<"):"<<action.action_code;
        case ParseAction::RR:
            return os << "RR("<< enc->decode8(action.tok_code) <<"):"<<action.action_code;
        case ParseAction::RU:
            return os << "RU("<< enc->decode8(action.tok_code) <<"):"<<action.action_code;
        case ParseAction::GR:
            return os << "GR:"<< action.action_code;
        default:
            return os << "Error (undefined action):" << action.action_code;
    }
}

string to_string(ParseAction const &action){
    
    IntegerEncoder *enc = IntegerEncoder::get();
    switch (action.action_type){
        case ParseAction::SHIFT:
            if (action.tok_code == 0){return "S:" + std::to_string(action.action_code);}
            else{return "S(" + enc->decode8(action.tok_code) + "):" + std::to_string(action.action_code);}
        case ParseAction::RL:
            return "RL(" + enc->decode8(action.tok_code) + "):" + std::to_string(action.action_code);
        case ParseAction::RR:
            return "RR(" + enc->decode8(action.tok_code) + "):" + std::to_string(action.action_code);
        case ParseAction::RU:
            return "RU(" + enc->decode8(action.tok_code) + "):" + std::to_string(action.action_code);
        case ParseAction::GR:
            return "GR:" + std::to_string(action.action_code);
        default:
            return "Error (undefined action):" + std::to_string(action.action_code);
    }
}




ostream& operator<<(ostream &os, StackItem const &stack){
    
    IntegerEncoder *enc = IntegerEncoder::get();
    return os   << "(t:"<< enc->decode8(stack.top)     <<"["<< *(stack.htop)
                << "] l:"<< enc->decode8(stack.left)  <<"["<< *(stack.hleft)
                << "] r:"<< enc->decode8(stack.right) <<"["<< *(stack.hright) << "])";
}


ParseAction::ParseAction(ParseAction const &other){
    this->action_code = other.action_code;
    this->tok_code    = other.tok_code;
    this->action_type = other.action_type;
}

ParseAction& ParseAction::operator=(ParseAction const &other){
    this->action_code = other.action_code;
    this->tok_code    = other.tok_code;
    this->action_type = other.action_type;
    return *this;
}


bool StateSignature::operator==(StateSignature const &other)const{
    
    if(this->J != other.J){return false;}
    if(this->stack_size == other.stack_size){
        for(int i = 0; i < this->stack_size-1;++i){
            if(this->stack[i] != other.stack[i]){return false;}
        }
        return    *(this->stack[this->stack_size-1].htop) == *(other.stack[this->stack_size-1].htop)
        && this->stack[this->stack_size-1].top  == other.stack[this->stack_size-1].top;
    }
    return false;
}

bool StateSignature::operator!=(StateSignature const &other)const{
    return !(*this == other);
}


ostream& operator<<(ostream &os, StateSignature const &sig){
    
    for(int i = 0; i < sig.stack_size && i < StateSignature::STACK_LOCALITY;++i){
        os <<"S"<< i << ":" << sig.stack[i]<<endl;
    }
    os << "J:"<< sig.J << endl;
    os << "N:"<< sig.N << endl;
    return os;
}


TemporaryReductionTable::TemporaryReductionTable(vector<PSTRING> const &all_categories){
  init_table(all_categories);
}

TemporaryReductionTable::TemporaryReductionTable(const TemporaryReductionTable &other){
  this->table = other.table;
}

TemporaryReductionTable& TemporaryReductionTable::operator= (const TemporaryReductionTable &other){
  this->table = other.table;
  return *this;
}

void TemporaryReductionTable::init_table(vector<PSTRING> const &all_categories){
  
  //Find table size
  int max_catcode = 0;
  IntegerEncoder *enc = IntegerEncoder::get();
  for(int i = 0; i < all_categories.size();++i){;
    TOK_CODE catcode = enc->get_code(all_categories[i],IntegerEncoder::PS_COLCODE);
    if(catcode > max_catcode){
      max_catcode = catcode;
    }
  }

  //Fills the table with false values
  vector<bool> tmp(max_catcode+1,false);
  for(int j = 0; j < max_catcode+1;++j){
    table.push_back(tmp);
  }

  str::SimpleTokenizer unary_splitter(UNARY_CODE);

  //adds positive entries to the table (valid reductions)
  for(int i = 0; i < all_categories.size();++i){
    size_t pos = all_categories[i].find(TMP_CODE);
    if (pos != PSTRING::npos){//if is a temp symbol
      
      //reflexive addition
      add_valid_reduction(enc->get_code(all_categories[i],IntegerEncoder::PS_COLCODE),enc->get_code(all_categories[i],IntegerEncoder::PS_COLCODE));
      //non reflexive addition
      PSTRING prefix = all_categories[i].substr(0,pos);
        
      for(int j = 0; j < all_categories.size();++j){//handles unaries as well
        vector<PSTRING> strs;
        unary_splitter.llex(all_categories[j],strs);
        
        //RELAXED TEMP REDUCTION: a temporary can reduce to any non temp non terminal too...
        if(strs.back().find(TMP_CODE) == PSTRING::npos){
          add_valid_reduction(enc->get_code(all_categories[i],IntegerEncoder::PS_COLCODE),enc->get_code(all_categories[j],IntegerEncoder::PS_COLCODE));     
        }
        //if(strs.back() == prefix){
        //    add_valid_reduction(enc->get_code(all_categories[i],IntegerEncoder::PS_COLCODE),enc->get_code(all_categories[j],IntegerEncoder::PS_COLCODE));
        //}
      }
    }
  }
}

void TemporaryReductionTable::add_valid_reduction(TOK_CODE child,TOK_CODE root){
    table[child][root] = true;
}

bool TemporaryReductionTable::operator==(TemporaryReductionTable const &other)const{
  assert(table.size()==other.table.size());
  for(int i = 0; i < table.size();++i){
    if (table[i] != other.table[i]){
      return false;
    }
  }
  return true;
}





////////////////////////////////////////////////////////////////////////////////////////////////
SrGrammar::SrGrammar(){}

SrGrammar::SrGrammar(string const &filename){
    load(filename);
    
}

SrGrammar::SrGrammar(Treebank const &train_bank,Treebank const &dev_bank,bool tagger,float semiringzero){
    SEMI_RING_ZERO = semiringzero;
    compile_grammar(train_bank,dev_bank,false);//no tagger
}

SrGrammar::SrGrammar(SrGrammar const &other){
  this->SEMI_RING_ZERO       = other.SEMI_RING_ZERO;
  this->temporaries          = other.temporaries;
  this->tags                 = other.tags;
  this->allnonterminals      = other.allnonterminals;
  this->axioms               = other.axioms;
  this->_nonterminals        = other._nonterminals;
  this->_axioms              = other._axioms;
  this->_tags                = other._tags;
  this->valid_reductions     = other.valid_reductions;
  this->actionencoder        = other.actionencoder;
  this->symbolencoder        = other.symbolencoder;
}

SrGrammar& SrGrammar::operator=(SrGrammar const &other){
  
  this->SEMI_RING_ZERO       = other.SEMI_RING_ZERO;
  this->temporaries          = other.temporaries;
  this->tags                 = other.tags;
  this->allnonterminals      = other.allnonterminals;
  this->axioms               = other.axioms;
  this->_nonterminals        = other._nonterminals;
  this->_axioms              = other._axioms;
  this->_tags                = other._tags;
  this->valid_reductions     = other.valid_reductions;
  this->actionencoder        = other.actionencoder;
  this->symbolencoder        = other.symbolencoder;
  return *this;
}


void SrGrammar::clear(){
    this->temporaries.clear();
    this->axioms.clear();
    this->tags.clear();
    this->allnonterminals.clear();
    this->valid_reductions = TemporaryReductionTable();
    this->actionencoder.clear();
    this->symbolencoder.clear();
    this->_nonterminals.clear();
    this->_axioms.clear();
    this->_tags.clear();
}


//from treebank
void SrGrammar::compile_grammar(Treebank const &train_bank,
                                Treebank const &dev_bank,
                                bool tagger){    //Extracts the grammar automaton from the treebank
    
    clear();
    // (1) categorize symbols
    train_bank.get_allsymbols(_nonterminals,_axioms,_tags);
    dev_bank.get_allsymbols(_nonterminals,_axioms,_tags);
    
    //add unknown grammatical symbols (tags)
    _tags.push_back(IntegerEncoder::get()->decode(NULL_SYMBOL));
    _tags.push_back(IntegerEncoder::UNK_SYMBOL);
    
    categorize_symbols(_nonterminals,_axioms,_tags);
    valid_reductions = TemporaryReductionTable(_nonterminals);
    
    // (2) encode actions
    if (tagger){
        vector<PSTRING> tags;
        train_bank.get_tagset(tags);
        cerr << "Error (grammar option not implemented)" << endl;
        exit(1);
        //compile_grammar_with_tagger(nonterminals,tags,axioms);
    }else{
        make_actions(train_bank,dev_bank);
    }
}

//from treebank
void SrGrammar::make_actions(Treebank const &train_bank,Treebank const &dev_bank){
        
    set<ParseAction,ActionComparator> act_set;
    for(int i = 0;i < train_bank.size();++i){readoff_actions(train_bank[i],act_set);}
    for(int i = 0;i < dev_bank.size();++i){readoff_actions(dev_bank[i],act_set);}
    
    actionencoder.clear();
    vector<ParseAction> all_actions(act_set.begin(),act_set.end());
    actionencoder.make_encoding(all_actions);
}


void SrGrammar::categorize_symbols(vector<PSTRING> const &all_nonterminals,
                                        vector<PSTRING> const &all_axioms,
                                        vector<PSTRING> const &all_tags){
    
    //1) Make symbol encoder
    IntegerEncoder *enc = IntegerEncoder::get();
    vector<TOK_CODE> symbol_codes;
    for(int i = 0; i < all_tags.size();++i){
        TOK_CODE c = enc->get_code(all_tags[i],IntegerEncoder::PS_COLCODE);
        symbol_codes.push_back(c);

    }
    for(int i = 0; i < all_nonterminals.size();++i){
        TOK_CODE c = enc->get_code(all_nonterminals[i],IntegerEncoder::PS_COLCODE);
        symbol_codes.push_back(c);
    }
    symbolencoder.make_map(symbol_codes);
    
    //2) fill in the bit vectors
    this->temporaries     =  vector<bool>(symbolencoder.nlabels(),false);
    this->allnonterminals =  vector<bool>(symbolencoder.nlabels(),false);
    this->axioms          =  vector<bool>(symbolencoder.nlabels(),false);
    this->tags            =  vector<bool>(symbolencoder.nlabels(),false);
    
    for(int i = 0; i < all_nonterminals.size();++i){
        TOK_CODE c = enc->get_code(all_nonterminals[i],IntegerEncoder::PS_COLCODE);
        this->allnonterminals[symbolencoder.get_index(c)] = true;
        if(all_nonterminals[i].find(TMP_CODE) != PSTRING::npos){
            this->temporaries[symbolencoder.get_index(c)] = true;
        }
    }
    for(int i = 0; i < all_axioms.size();++i){
        TOK_CODE c = enc->get_code(all_axioms[i],IntegerEncoder::PS_COLCODE);
        this->axioms[symbolencoder.get_index(c)] = true;
    }
    for(int i = 0; i < all_tags.size();++i){
        TOK_CODE c = enc->get_code(all_tags[i],IntegerEncoder::PS_COLCODE);
        this->tags[symbolencoder.get_index(c)] = true;
    }
}

//TODO : manage shift and GR
const ParseAction& SrGrammar::get_action(ParseAction::ActionType atype,TOK_CODE symbol_code)const{
    return actionencoder.get_action(atype, symbol_code);
}

void SrGrammar::load(const string &filename){
    
    this->clear();
    ifstream grammar_in(filename);
    string bfr;
    getline(grammar_in,bfr);
    SEMI_RING_ZERO = stof(bfr);
    
    while(getline(grammar_in,bfr) && bfr != "#####"){_nonterminals.push_back(str::decode(bfr));}
    while(getline(grammar_in,bfr) && bfr != "#####"){_axioms.push_back(str::decode(bfr));}
    
    while(getline(grammar_in,bfr)){_tags.push_back(str::decode(bfr));}
    grammar_in.close();

    //compiles the grammar
    actionencoder.clear();
    categorize_symbols(_nonterminals,_axioms,_tags);
    valid_reductions = TemporaryReductionTable(_nonterminals);
    //compiles the actions
    load_actions(filename+"-actions");
}

void SrGrammar::save(const string &filename)const{
    
    ofstream grammar_out(filename);
    grammar_out << SEMI_RING_ZERO << endl;
    for(int i = 0; i < _nonterminals.size();++i){
        grammar_out << str::encode(_nonterminals[i]) << endl;
    }
    grammar_out << "#####"<<endl;
    for(int i = 0; i < _axioms.size();++i){
        grammar_out << str::encode(_axioms[i]) << endl;
    }
    grammar_out << "#####"<<endl;
    for(int i = 0; i < _tags.size();++i){
        grammar_out << str::encode(_tags[i]) << endl;
    }
    grammar_out.close();
    save_actions(filename+"-actions");
}


void SrGrammar::load_actions(string filename){
 
    actionencoder.clear();
    vector<ParseAction> all_actions;
    
    ifstream ins(filename);
    string bfr;
    str::SimpleTokenizer lex;
    vector<PSTRING> tokens;
    IntegerEncoder *enc = IntegerEncoder::get();
    while(getline(ins,bfr)){
        tokens.clear();
        lex.llex(bfr,tokens);
        if (tokens[0] == L"S"){all_actions.push_back(make_action(ParseAction::SHIFT,0,0));}
        if (tokens[0] == L"GR"){all_actions.push_back(make_action(ParseAction::GR,0,0));}
        if (tokens[0] == L"RL"){all_actions.push_back(make_action(ParseAction::RL,enc->encode(tokens[1],IntegerEncoder::PS_COLCODE),0));}
        if (tokens[0] == L"RR"){all_actions.push_back(make_action(ParseAction::RR,enc->encode(tokens[1],IntegerEncoder::PS_COLCODE),0));}
        if (tokens[0] == L"RU"){all_actions.push_back(make_action(ParseAction::RU,enc->encode(tokens[1],IntegerEncoder::PS_COLCODE),0));}
    }
    ins.close();
    
    actionencoder.make_encoding(all_actions);
    
}

void SrGrammar::save_actions(string filename)const{
    
    ofstream outs(filename);
    IntegerEncoder *enc = IntegerEncoder::get();
    
    for(int i = 0; i < actionencoder.nactions();++i){
        
        ParseAction::ActionType t = actionencoder.get_action(i).action_type;
        if(t==ParseAction::SHIFT){
            outs << "S" << endl;
        }
        if(t==ParseAction::GR){
            outs << "GR" << endl;
        }
        if(t == ParseAction::RL){
            outs << "RL " << enc->decode8(actionencoder.get_action(i).tok_code) <<endl;
        }
        if(t == ParseAction::RR){
            outs << "RR " << enc->decode8(actionencoder.get_action(i).tok_code) <<endl;
        }
        if(t == ParseAction::RU){
            outs << "RU " << enc->decode8(actionencoder.get_action(i).tok_code) <<endl;
        }
    }
    outs.close();
}




/*
 SrGrammar::SrGrammar(Treebank const &treebank,float semiringzero){
 SEMI_RING_ZERO = semiringzero;
 compile_grammar(treebank);
 }
 */

/*
SrGrammar::SrGrammar(vector<PSTRING> const &nonterminals,vector<PSTRING> const &axioms,float semiringzero){
    SEMI_RING_ZERO = semiringzero;
    compile_grammar(nonterminals,axioms);
}
*/
//Extracts the grammar automaton from the treebank



/*
void SrGrammar::compile_grammar(Treebank const &treebank){    //Extracts the grammar automaton from the treebank

  clear();
  vector<PSTRING> nonterminals;
  vector<PSTRING> axioms;
  treebank.get_nonterminals(nonterminals,axioms);
  compile_grammar(nonterminals,axioms);
}
*/
/*
void SrGrammar::compile_grammar(Treebank const &train_bank,Treebank const &dev_bank){
    vector<PSTRING> nonterminals;
    vector<PSTRING> axioms;
    train_bank.get_nonterminals(nonterminals,axioms);
    dev_bank.get_nonterminals(nonterminals,axioms);
    this->ordered_nonterminals = nonterminals;
    this->ordered_axioms = axioms;
    std::sort(ordered_nonterminals.begin(),ordered_nonterminals.end());
    std::sort(ordered_axioms.begin(),ordered_axioms.end());
    categorize_nonterminals(ordered_nonterminals,ordered_axioms);
    Nshifts = 2;
    valid_reductions = TemporaryReductionTable(ordered_nonterminals);
    make_actions(train_bank,dev_bank);
    //make_reverse_action_vector(all_actions);
    
    for(int i = 0; i < all_actions.size();++i){
        all_actions[i].action_code = i;
        reverse_action_table.add_action(all_actions[i]);
    }
    make_reverse_action_vector(all_actions);

}



void SrGrammar::compile_grammar_with_tagger(vector<PSTRING> const &nonterminals,
                                            vector<PSTRING> const &tags,
                                            vector<PSTRING> const &axioms){
    //Extracts the grammar automaton from the ordered set of non terminals
    clear();
    this->ordered_nonterminals = nonterminals;
    this->ordered_axioms = axioms;
    this->ordered_tags = tags;
    //different orderings -> different behaviour (wrt action encoding)-> sort to normalize
    std::sort(ordered_nonterminals.begin(),ordered_nonterminals.end());
    std::sort(ordered_axioms.begin(),ordered_axioms.end());
    std::sort(ordered_tags.begin(),ordered_tags.end());
    categorize_nonterminals(ordered_nonterminals,ordered_axioms);
    Nshifts = ordered_tags.size();
    valid_reductions = TemporaryReductionTable(ordered_nonterminals);
    make_actions(ordered_nonterminals,ordered_tags);
}
*/


/*
void SrGrammar::make_actions(vector<PSTRING> const &all_nonterminals){

  IntegerEncoder *enc = IntegerEncoder::get();
  all_actions.clear();

  unsigned int C = 0;
  all_actions.push_back(make_action(ParseAction::SHIFT,0,C));
  ++C;
  all_actions.push_back(make_action(ParseAction::GR,0,C));
  ++C;
  for(int i = 0 ; i < Nt ; ++i){
    all_actions.push_back(make_action(ParseAction::RU, enc->get_code(all_nonterminals[i],IntegerEncoder::PS_COLCODE),C));
    ++C;
  }
  for(int i = 0 ; i < Nt ; ++i){
      all_actions.push_back(make_action(ParseAction::RL, enc->get_code(all_nonterminals[i],IntegerEncoder::PS_COLCODE),C));
    ++C;
  }
  for(int i = 0 ; i < Nt ; ++i){
    all_actions.push_back(make_action(ParseAction::RR, enc->get_code(all_nonterminals[i],IntegerEncoder::PS_COLCODE),C));
    ++C;
  }
  this->Na = all_actions.size();
  make_reverse_action_vector(all_actions);
}

void SrGrammar::make_actions(vector<PSTRING> const &all_nonterminals,vector<PSTRING> const &ordered_tags){
    
    IntegerEncoder *enc = IntegerEncoder::get();
    all_actions.clear();
    
    unsigned int C = 0;
    for(int i = 0; i < ordered_tags.size();++i){
        all_actions.push_back(make_action(ParseAction::SHIFT,enc->get_code(ordered_tags[i],IntegerEncoder::PS_COLCODE),C));
        ++C;
    }
    for(int i = 0 ; i < Nt ; ++i){
        all_actions.push_back(make_action(ParseAction::RL, enc->get_code(all_nonterminals[i],IntegerEncoder::PS_COLCODE),C));
        ++C;
    }
    for(int i = 0 ; i < Nt ; ++i){
        all_actions.push_back(make_action(ParseAction::RR, enc->get_code(all_nonterminals[i],IntegerEncoder::PS_COLCODE),C));
        ++C;
    }
    this->Na = all_actions.size();
    make_reverse_action_vector(all_actions);
}

ParseAction SrGrammar::make_action(ParseAction::ActionType action_type, TOK_CODE tok_code,unsigned int action_code)const{

  ParseAction a;
  a.action_type = action_type;     //the action
  a.tok_code = tok_code;          //X code for RU,RR,RL
  a.action_code = action_code;   //index code given by the grammar
  return a;

}
*/
/*
const ParseAction& SrGrammar::get_action(ParseAction::ActionType atype,TOK_CODE symbol_code)const{

    return reverse_action_table(atype,symbol_code);
  /*
    if (Nshifts == 2){
        unsigned int begin = Nshifts;
        if (atype == ParseAction::SHIFT){return all_actions[0];}
        if (atype == ParseAction::GR){return all_actions[1];}
        if (atype == ParseAction::RL){begin += Nt;}
        if (atype == ParseAction::RR){begin += 2*Nt;}
        for(int i = begin; i < begin + Nt; ++i){
            if(all_actions[i].tok_code == symbol_code){return all_actions[i];}
        }
    }else{
        if (atype == ParseAction::SHIFT){
            for(int i = 0; i < Nshifts;++i){
                if(all_actions[i].tok_code == symbol_code){return all_actions[i];}
            }
        }
        unsigned int begin = Nshifts;
        if (atype == ParseAction::RR){begin += Nt;}
        for(int i = begin; i < begin + Nt; ++i){
            if(all_actions[i].tok_code == symbol_code){return all_actions[i];}
        }
    }
   
  cerr << "Programming error detected in the grammar module"<<endl;
  cerr << IntegerEncoder::get()->decode8(symbol_code)<<endl;
    if (atype==ParseAction::RR){
        cerr <<"RR"<<endl;
    }
    if (atype==ParseAction::RL){
        cerr <<"RL"<<endl;
    }
    if (atype==ParseAction::RU){
        cerr <<"RU"<<endl;
    }
    if (atype==ParseAction::SHIFT){
        cerr <<"S"<<endl;
    }

  exit(1);
  return all_actions[0];
}
*/

/*
const ParseAction&  SrGrammar::get_shift_action()const{
    assert(Nshifts == 2);
    return get_action(ParseAction::SHIFT,0);
}
const ParseAction&  SrGrammar::get_ghost_action()const{
    assert(Nshifts == 2);
    return get_action(ParseAction::GR,0);
}
*/

void SrGrammar::get_ordered_actions(vector<ParseAction> &action_vector) const{
    
    for(int i = 0; i < actionencoder.nactions();++i){
        action_vector.push_back(actionencoder.get_action(i));
    }
}


/*
vector<ParseAction> SrGrammar::get_ordered_actions()const{
    return all_actions;
}
*/

void SrGrammar::select_actions(vector<float> &weight_vector,ParseAction const &prev_action,StateSignature const &sig)const{
    
    for(int i = 0; i < actionencoder.nactions();++i){
        if (! is_valid_action(prev_action,actionencoder.get_action(i),sig)){
            weight_vector[i] = SEMI_RING_ZERO;
        }
    }
    /*
    }else{
        for(int i = 0; i < Na;++i){
            if (! is_valid_action_with_tagger(prev_action,all_actions[i],sig)){weight_vector[i] = SEMI_RING_ZERO;}
        }
    }
    */
}

// @@m
void SrGrammar::select_actions(vector<bool> &action_vector,ParseAction const &prev_action,StateSignature const &sig) const{
    //if(Nshifts==2){
    
    for(int i = 0; i <actionencoder.nactions();++i){
        action_vector[i] = is_valid_action(prev_action,actionencoder.get_action(i),sig);
    }
    /*
     else{
        for(int i = 0; i < Na;++i) action_vector[i] = is_valid_action_with_tagger(prev_action,all_actions[i],sig);
    }
     */
}

bool SrGrammar::has_tranformations()const{//says if this grammar contains transformed symbols
    for(vector<PSTRING>::const_iterator it = _nonterminals.begin();it != _nonterminals.end();++it){
        if(it->find_first_of(SEP_CODE) != PSTRING::npos){return true;}
    }
    return false;
}

 
bool SrGrammar::is_valid_action(ParseAction const &prev_action,ParseAction const &action,StateSignature const &sig)const{

        switch(action.action_type){
            case ParseAction::SHIFT:
            {
                if(prev_action.action_type == ParseAction::SHIFT){//NO SHIFT if last action was a shift
                    return false;
                    break;
                }
                
                if (sig.J == (sig.N-1)){return false;}                         //End of sentence
                break;
            }
            case ParseAction::RL:
            {
                if(prev_action.action_type == ParseAction::SHIFT){//NO BINARY REDUCE if last action was a shift
                    return false;
                    break;
                }
                
                if (sig.stack_size < 3){                                    //NO REDUCE when stack too small
                    return false;
                    break;
                }
                if (is_temporary(sig.stack[0].top)){//NO REDUCE LEFT if s0 is temporary
                    return false;
                    break;
                }
                if (is_temporary(sig.stack[1].top) && !is_valid_temporary_reduction(sig.stack[1].top,action.tok_code)){   //REDUCE ONLY TO VALID TEMP REDUCTIONS
                    return false;
                    break;
                }
                if ((sig.J == sig.N-1) && sig.stack[2].is_init_state() && is_temporary(action.tok_code)){ //NO TEMP REDUCTION when end of parsing
                    return false;
                    break;
                }
                if ((sig.J == sig.N-1) && is_temporary(sig.stack[2].top) && is_temporary(action.tok_code)){ //NO BINARY TEMP REDUCTION when about to end parsing
                    return false;
                    break;
                }
            
                if(is_axiom(action.tok_code) && ( (sig.J != sig.N-1) || !sig.stack[2].is_init_state() ) ){ //NO AXIOM REDUCTION WHEN NOT FINISHING PARSING
                    return false;
                    break;
                }
                break;
            }
            case ParseAction::RR:
            {
                if(prev_action.action_type == ParseAction::SHIFT){//NO BINARY REDUCE if last action was a shift
                    return false;
                    break;
                }
                if (sig.stack_size < 3){                                    //NO REDUCE when stack too small
                    return  false;
                    break;
                }
                if (is_temporary(sig.stack[1].top)){                                         //NO REDUCE RIGHT if s1 is temporary
                    return false;
                    break;
                }
                if (is_temporary(sig.stack[0].top) && !is_valid_temporary_reduction(sig.stack[0].top,action.tok_code)){   //REDUCE ONLY TO VALID TEMP REDUCTIONS
                   return  false;
                    break;
                }
                if ((sig.J == sig.N-1) && sig.stack[2].is_init_state() && is_temporary(action.tok_code)){ //NO TEMP REDUCTION when end of parsing
                  return false;
                    break;
                }
                if ((sig.J == sig.N-1) && is_temporary(sig.stack[2].top) && is_temporary(action.tok_code)){ //NO BINARY TEMP REDUCTION when about to end parsing
                    return  false;
                    break;
                }
            
                if(is_axiom(action.tok_code) && ( (sig.J != sig.N-1) || !sig.stack[2].is_init_state())){ //NO AXIOM REDUCTION WHEN NOT FINISHING PARSING
                    return false;
                    break;
                }
                
                break;
            }
            case ParseAction::RU:
            {
                if(prev_action.action_type != ParseAction::SHIFT){
                    return false;
                }
                if (sig.J == sig.N-1 && sig.stack_size < 2){
                    return false;
                }
                if (sig.stack_size < 2){                                    //NO REDUCE UNARY when s0 is not a terminal
                    return false;
                }
                break;
            }
            case ParseAction::GR:
            {
                if(prev_action.action_type != ParseAction::SHIFT){
                    return false;
                }
                if (sig.J == sig.N-1 && sig.stack_size < 2){
                    return false;
                }
                if (sig.stack_size < 2){                                    //NO SHADOW REDUCE when s0 is not a terminal
                    return false;
                }
                break;
            }
            default:
                cout << "Error (Undefined action) : " << action.action_type << endl;
                exit(1);
        }
	return true;
}

/*
bool SrGrammar::is_valid_action_with_tagger(ParseAction const &prev_action,ParseAction const &action,StateSignature const &sig)const{
    
    
    switch(action.action_type){
        case ParseAction::SHIFT:
        {
            if (sig.J == (sig.N-1)){return false;}                         //End of sentence
            break;
        }
        case ParseAction::RL:
        {
            if (sig.stack_size < 3){                                    //NO REDUCE when stack too small
                return false;
                break;
            }
            if (is_temporary(sig.stack[0].top)){                                         //NO REDUCE LEFT if s0 is temporary
                return false;
                break;
            }
            if (is_temporary(sig.stack[1].top) && !is_valid_temporary_reduction(sig.stack[1].top,action.tok_code)){   //REDUCE ONLY TO VALID TEMP REDUCTIONS
                return false;
                break;
            }
            if ((sig.J == sig.N-1) && sig.stack[2].is_init_state() && is_temporary(action.tok_code)){ //NO TEMP REDUCTION when end of parsing
                return false;
                break;
            }
            if ((sig.J == sig.N-1) && is_temporary(sig.stack[2].top) && is_temporary(action.tok_code)){ //NO BINARY TEMP REDUCTION when about to end parsing
                return false;
                break;
            }
            
            if(is_axiom(action.tok_code) && ( (sig.J != sig.N-1) || !sig.stack[2].is_init_state() ) ){ //NO AXIOM REDUCTION WHEN NOT FINISHING PARSING
                return false;
                break;
            }
            break;
        }
        case ParseAction::RR:
        {
            if (sig.stack_size < 3){                                    //NO REDUCE when stack too small
                return  false;
                break;
            }
            if (is_temporary(sig.stack[1].top)){                                         //NO REDUCE RIGHT if s1 is temporary
                return false;
                break;
            }
            if (is_temporary(sig.stack[0].top) && !is_valid_temporary_reduction(sig.stack[0].top,action.tok_code)){   //REDUCE ONLY TO VALID TEMP REDUCTIONS
                return  false;
                break;
            }
            if ((sig.J == sig.N-1) && sig.stack[2].is_init_state() && is_temporary(action.tok_code)){ //NO TEMP REDUCTION when end of parsing
                return false;
                break;
            }
            if ((sig.J == sig.N-1) && is_temporary(sig.stack[2].top) && is_temporary(action.tok_code)){ //NO BINARY TEMP REDUCTION when about to end parsing
                return  false;
                break;
            }
            
            if(is_axiom(action.tok_code) && ( (sig.J != sig.N-1) || !sig.stack[2].is_init_state())){ //NO AXIOM REDUCTION WHEN NOT FINISHING PARSING
                return false;
            }
            
            break;
        }
        case ParseAction::RU:
        case ParseAction::GR:
                    return false;
        default:
            cout << "Error (Undefined action) : " << action.action_type << endl;
            exit(1);
    }
    return true;
}
*/


bool SrGrammar::is_terminal(TOK_CODE cat)const{
       return tags.at(symbolencoder.get_index(cat));
}

bool SrGrammar::is_axiom(TOK_CODE cat)const{
    return axioms.at(symbolencoder.get_index(cat));
}


bool SrGrammar::is_temporary(TOK_CODE cat)const{
    return temporaries.at(symbolencoder.get_index(cat));
}

bool SrGrammar::is_valid_temporary_reduction(TOK_CODE cat_child,TOK_CODE cat_root)const{
      return valid_reductions.is_valid_temporary_reduction(cat_child,cat_root);
}

/*
ostream& SrGrammar::print_action_vec(ostream &os){
    for(int i = 0; i < all_actions.size();++i){
        os << all_actions[i] << " ";
    }
    return os;
}
 */




/*
void SrGrammar::make_reverse_action_vector(vector<ParseAction> const &all_actions){
    
    unsigned int max_action_code = 0 ;
    for(int i = 0; i < all_actions.size(); ++i){
        uint16_t acode = all_actions[i].get_code();
        if (acode > max_action_code){max_action_code = acode;}
    }
    reverse_action_vector = vector<unsigned int> (max_action_code+1,0);
    for(int i = 0; i < all_actions.size(); ++i){
        reverse_action_vector[ all_actions[i].get_code()] = i;
    }
}
*/
 

ParseAction SrGrammar::make_action(ParseAction::ActionType action_type, TOK_CODE tok_code,unsigned int action_code)const{
    
    ParseAction a;
    a.action_type = action_type;     //the action
    a.tok_code = tok_code;          //X code for RU,RR,RL
    a.action_code = action_code;   //index code given by the grammar
    return a;
}


void SrGrammar::readoff_actions(AbstractParseTree const *root,std::set<ParseAction,ActionComparator> &all_actions){
    
    
    // typedef std::set<ParseAction>::iterator SI;
    
    IntegerEncoder *enc = IntegerEncoder::get();
    if(root->is_leaf()){
        all_actions.insert(make_action(ParseAction::SHIFT,0,0));
    }else if (root->arity()==1){//unary node
        
        ParseNode const *rroot = static_cast<ParseNode const*>(root);
        all_actions.insert(make_action(ParseAction::RU, enc->get_code(rroot->get_label(),IntegerEncoder::PS_COLCODE),0));
        readoff_actions(rroot->get_child_at(0),all_actions);
        
    }else if(root->arity()==2){//binary nodes
        
        ParseNode const *rroot = static_cast<ParseNode const*>(root);
        AbstractParseTree *c0 = rroot->get_child_at(0);
        AbstractParseTree *c1 = rroot->get_child_at(1);
        
        //Ghost reductions
        readoff_actions(c0,all_actions);
        if (c0->is_leaf()){
            all_actions.insert(make_action(ParseAction::GR,0,0));
        }
        readoff_actions(c1,all_actions);
        if (c1->is_leaf()){
            all_actions.insert(make_action(ParseAction::GR,0,0));
        }
        
        if(rroot->get_child_at(0)->is_head()){
            all_actions.insert(make_action(ParseAction::RL, enc->get_code(rroot->get_label(),IntegerEncoder::PS_COLCODE),0));
        }else{
            all_actions.insert(make_action(ParseAction::RR, enc->get_code(rroot->get_label(),IntegerEncoder::PS_COLCODE),0));
        }
    }
}


/*

ReverseActionTable::ReverseActionTable(){
    this->reductions = new vector<ParseAction>();
}
*/
 
/*
ReverseActionTable::ReverseActionTable(vector<PSTRING> &allnonterminals){
    
    N = 0;
    IntegerEncoder *enc = IntegerEncoder::get();
    for(int i = 0; i < allnonterminals.size();++i){
        unsigned int C = enc->get_code(allnonterminals[i],IntegerEncoder::PS_COLCODE);
        if( C > N ) { N = C;}
    }
    this->reductions = new vector<ParseAction>(3 * N ,ParseAction());
}

ReverseActionTable::ReverseActionTable(ReverseActionTable const &other){
    this->N     = other.N;
    this->shift = other.shift;
    this->ghost = other.ghost;
    //this->reductions = other.reductions;
    *(this->reductions) = *other.reductions;
}

ReverseActionTable& ReverseActionTable::operator=(ReverseActionTable const &other){
    this->N     = other.N;
    this->shift = other.shift;
    this->ghost = other.ghost;
    //other.dump(100);
    
    reductions->clear();
    for(int i = 0;i < other.reductions->size();++i){
        reductions->push_back((*other.reductions)[i]);
    }
    //this->reductions = other.reductions;
    return *this;
}


const ParseAction& ReverseActionTable::operator()(ParseAction::ActionType atype,TOK_CODE tok_code)const{
    
    
    switch (atype) {
        case ParseAction::SHIFT:
            return shift;
            break;
        case ParseAction::GR:
            return ghost;
            break;
        case ParseAction::RU:
            return (*reductions)[tok_code];
            break;
        case ParseAction::RL:
            return (*reductions)[N+tok_code];
            break;
        case ParseAction::RR:
            return (*reductions)[2*N+tok_code];
            break;
        default:
            cerr << " RevActionTable Error " << endl;
            exit(1);
    }
    
}


void ReverseActionTable::add_action(ParseAction &action){
    
    switch (action.action_type) {
        case ParseAction::SHIFT:
            shift = action;
            break;
        case ParseAction::GR:
            ghost = action;
            break;
        case ParseAction::RU:
            (*reductions)[action.tok_code] = action;
            break;
        case ParseAction::RL:
            (*reductions)[N+action.tok_code] = action;
            break;
        case ParseAction::RR:
            (*reductions)[2*N+action.tok_code] = action;
            break;
        default:
            cerr << " RevActionTable Error " << endl;
            exit(1);
    }
}
*/

/*
void SrGrammar::load_grammar(string const &filename){
        ifstream grammar_in(filename);
        string bfr;
        getline(grammar_in,bfr);
        SEMI_RING_ZERO = stof(bfr);
        
        vector<PSTRING> all_nonterminals;
        vector<PSTRING> all_axioms;
        vector<PSTRING> all_tags;
        
        while(getline(grammar_in,bfr) && bfr != "#####"){all_nonterminals.push_back(str::decode(bfr));}
        while(getline(grammar_in,bfr) && bfr != "#####"){all_axioms.push_back(str::decode(bfr));}
        while(getline(grammar_in,bfr)){all_tags.push_back(str::decode(bfr));}
        
        grammar_in.close();
    
        this->ordered_nonterminals = all_nonterminals;
        this->ordered_axioms = all_axioms;
        std::sort(ordered_nonterminals.begin(),ordered_nonterminals.end());
        std::sort(ordered_axioms.begin(),ordered_axioms.end());
        categorize_nonterminals(ordered_nonterminals,ordered_axioms);
        Nshifts = 2;
        valid_reductions = TemporaryReductionTable(ordered_nonterminals);
        load_actions(filename+"-actions");

}
*/



size_t ActionCoder::get_actioninternalposition(ParseAction::ActionType type,TOK_CODE tok)const{
    
    switch(type){
        case ParseAction::GR    : return 0;
        case ParseAction::SHIFT : return 1 ;//+ symbolmap.get_index(TokCodeWrap(tok));
        case ParseAction::RL    : return 1 + symbolmap.nlabels()+symbolmap.get_index(tok);
        case ParseAction::RR    : return 1 + 2*symbolmap.nlabels()+symbolmap.get_index(tok);
        case ParseAction::RU    : return 1 + 3*symbolmap.nlabels()+symbolmap.get_index(tok);
        default:
            cerr << "Error: unsupported action " << endl;
            exit(1);
    }
    
}

void ActionCoder::make_encoding(vector<ParseAction> &actions){
    this->clear();
    set<TOK_CODE> symbols;
    for(int i = 0; i < actions.size();++i){
        symbols.insert(actions[i].tok_code);
    }
    vector<TOK_CODE> vsymbols(symbols.begin(),symbols.end());
    symbolmap.make_map(vsymbols);
    
    size_t N = symbolmap.nlabels();
    
    //We assume GR,S,RL,RR,RU as abstract actions
    allactions.resize(4*N+1,ParseAction());
    
    //We encode all actions and assign them a redundant action code
    for(int i = 0; i < actions.size();++i){
        allactions[get_actioninternalposition(actions[i].action_type,actions[i].tok_code)] = actions[i];
    }
    unsigned int c = 0;
    vector<ParseAction> real_actions;
    for(int i = 0; i < allactions.size();++i){
        if(allactions[i].action_type != ParseAction::NULL_ACTION){
            allactions[i].action_code = c;
            if (allactions[i].action_type == ParseAction::SHIFT){shift_index=c;}
            if (allactions[i].action_type == ParseAction::GR){ghost_index=c;}
            ++c;
            real_actions.push_back(allactions[i]);
        }
    }
    actionmap.make_map(real_actions);
}

ActionCoder::ActionCoder(vector<ParseAction> &actions){
    make_encoding(actions);
}

ActionCoder::ActionCoder(ActionCoder const &other){
    symbolmap = other.symbolmap;
    actionmap = other.actionmap;
    allactions = other.allactions;
}

ActionCoder& ActionCoder::operator=(ActionCoder const &other){
    clear();
    symbolmap = other.symbolmap;
    actionmap = other.actionmap;
    allactions = other.allactions;
    return *this;
}

unsigned int ActionCoder::get_action_index(ParseAction const &action)const{
    return actionmap.get_index(action);
}

unsigned int ActionCoder::get_action_index(ParseAction::ActionType type,TOK_CODE tok)const{
    return actionmap.get_index(get_action(type,tok));
}
const ParseAction& ActionCoder::get_action(ParseAction::ActionType type,TOK_CODE tok)const{
    return allactions[get_actioninternalposition(type,tok)];
}
const ParseAction& ActionCoder::get_action(ActionCoder::ACTION_INDEX index)const{
    return actionmap.get_label(index);
}

void ActionCoder::clear(){
    symbolmap.clear();
    actionmap.clear();
    allactions.clear();
}


