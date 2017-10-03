#include "state_graph.h"
#include <stack>
#include <algorithm>



ParseState::ParseState(){}

ParseState* ParseState::init_state(){
    
    ParseState *ps = new ParseState();
    ps->I = -2;
    ps->J = -1;
    ps->top = SrGrammar::NULL_SYMBOL;
    ps->left = SrGrammar::NULL_SYMBOL;
    ps->right = SrGrammar::NULL_SYMBOL;
    ps->htop         = &NULL_TOKEN;
    ps->hleft        = &NULL_TOKEN;
    ps->hright       = &NULL_TOKEN;
    ps->left_corner  = &NULL_TOKEN;
    ps->right_corner = &NULL_TOKEN;
    ps->prefix_weight = 0.0;
    ps->stack_prev = NULL;
    ps->history_prev = NULL;
    ps->incoming_action = ParseAction();
    ps->incoming_action.action_type = ParseAction::NULL_ACTION;
    
    return ps;
}

//shifts without tagging
ParseState* ParseState::shift(SrGrammar const &grammar,InputDag &input_sequence,float prefix_weight){
    
    ParseState *ps = new ParseState();
    ps->I = this->J;
    ps->J = this->J+1;
    ps->top    =  input_sequence[ps->J]->get_catcode();
    ps->left   =  SrGrammar::NULL_SYMBOL;
    ps->right  =  SrGrammar::NULL_SYMBOL;
    ps->htop   =  input_sequence.shift_token(ps->J);
    ps->hleft  =  ps->htop;
    ps->hright =  ps->htop;
    ps->left_corner  = ps->htop;
    ps->right_corner = ps->htop;
    ps->prefix_weight = prefix_weight;
    ps->stack_prev = this;
    ps->history_prev = this;
    ps->incoming_action = grammar.get_shift_action();
    return ps;
}

//shifts and tags
ParseState* ParseState::shift_tag(TOK_CODE tag_symbol,SrGrammar const &grammar,InputDag &input_sequence,float prefix_weight){
    
    ParseState *ps = new ParseState();
    
    ps->I = this->J;
    ps->J = this->J+1;
    ps->top    =  tag_symbol;
    ps->left   =  SrGrammar::NULL_SYMBOL;
    ps->right  =  SrGrammar::NULL_SYMBOL;
    ps->htop   =  input_sequence.shift_token(tag_symbol,ps->J);
    ps->hleft  =  ps->htop;
    ps->hright =  ps->htop;
    ps->left_corner  = ps->htop;
    ps->right_corner = ps->htop;
    ps->prefix_weight = prefix_weight;
    ps->stack_prev = this;
    ps->history_prev = this;
    ps->incoming_action = grammar.get_action(ParseAction::SHIFT,tag_symbol);
    
    return ps;
}


ParseState* ParseState::reduce_left(TOK_CODE rsymbol,SrGrammar const &grammar,float prefix_weight){
    
    ParseState *ps = new ParseState();
    ps->I      =  this->stack_prev->I;
    ps->J      =  this->J;
    ps->top    =  rsymbol;
    ps->left   =  this->stack_prev->top;
    ps->right  =  this->top;
    ps->htop   =  this->stack_prev->htop;
    ps->hleft  =  this->stack_prev->htop;
    ps->hright =  this->htop;
    ps->left_corner  = this->stack_prev->left_corner;
    ps->right_corner = this->right_corner;
    ps->prefix_weight = prefix_weight;
    ps->stack_prev = this->stack_prev->stack_prev;
    ps->history_prev = this;
    ps->incoming_action = grammar.get_action(ParseAction::RL,rsymbol);
    
    return ps;

}


ParseState* ParseState::reduce_right(TOK_CODE rsymbol,SrGrammar const &grammar,float prefix_weight){
    
    ParseState *ps = new ParseState();
    ps->I      =  this->stack_prev->I;
    ps->J      =  this->J;
    ps->top    =  rsymbol;
    ps->left   =  this->stack_prev->top;
    ps->right  =  this->top;
    ps->htop   =  this->htop;
    ps->hleft  =  this->stack_prev->htop;
    ps->hright =  this->htop;
    ps->left_corner  = this->stack_prev->left_corner;
    ps->right_corner = this->right_corner;
    ps->prefix_weight = prefix_weight;
    ps->stack_prev = this->stack_prev->stack_prev;
    ps->history_prev = this;
    ps->incoming_action = grammar.get_action(ParseAction::RR,rsymbol);

    return ps;
    
}

ParseState* ParseState::reduce_unary(TOK_CODE rsymbol,SrGrammar const &grammar,float prefix_weight){
    
    ParseState *ps = new ParseState();
    
    ps->I      =  this->I;
    ps->J      =  this->J;
    ps->top    =  rsymbol;
    ps->left   =  this->left;
    ps->right  =  this->right;
    ps->htop   =  this->htop;
    ps->hleft  =  this->hleft;
    ps->hright =  this->hright;
    ps->left_corner  = this->left_corner;
    ps->right_corner = this->right_corner;
    ps->prefix_weight = prefix_weight;
    ps->stack_prev = this->stack_prev;
    ps->history_prev = this;
    ps->incoming_action = grammar.get_action(ParseAction::RU,rsymbol);

    return ps;
    
}

ParseState* ParseState::ghost_reduce(SrGrammar const &grammar,float prefix_weight){
    
    ParseState *ps = new ParseState();
    
    ps->I      =  this->I;
    ps->J      =  this->J;
    ps->top    =  this->top; 
    ps->left   =  this->left;
    ps->right  =  this->right;
    ps->htop   =  this->htop;
    ps->hleft  =  this->hleft;
    ps->hright =  this->hright;
    ps->left_corner  = this->left_corner;
    ps->right_corner = this->right_corner;
    ps->prefix_weight   = prefix_weight;
    ps->stack_prev      = this->stack_prev;
    ps->history_prev    = this;
    ps->incoming_action = grammar.get_ghost_action();
  
    return ps;
    
}

ParseState::ParseState(const ParseState &other){

    this->I      = other.I;
    this->J      = other.J;
    this->top    =  other.top;
    this->left   =  other.left;
    this->right  =  other.right;
    this->htop   =  other.htop;
    this->hleft  =  other.hleft;
    this->hright =  other.hright;
    this->left_corner = other.left_corner;
    this->right_corner = other.right_corner;
    this->prefix_weight   = other.prefix_weight;
    this->stack_prev      = other.stack_prev;
    this->history_prev    = other.history_prev;
    this->incoming_action = other.incoming_action;
}

ParseState& ParseState::operator=(const ParseState &other){
    
    this->I      = other.I;
    this->J      = other.J;
    this->top    =  other.top;
    this->left   =  other.left;
    this->right  =  other.right;
    this->htop   =  other.htop;
    this->hleft  =  other.hleft;
    this->hright =  other.hright;
    this->left_corner = other.left_corner;
    this->right_corner = other.right_corner;
    this->prefix_weight   = other.prefix_weight;
    this->stack_prev      = other.stack_prev;
    this->history_prev    = other.history_prev;
    this->incoming_action = other.incoming_action;
    return *this;
}

bool ParseState::is_init()const{
    return this->incoming_action.action_type == ParseAction::NULL_ACTION;
}

bool ParseState::is_success(int eta, size_t N)const{return eta == 3*N-1 && this->stack_prev->is_init();}


float ParseState::weight()const{
    return prefix_weight;
}

void ParseState::make_signature(StateSignature &signature,InputDag const *input_sequence,unsigned int input_length){

    ParseState *current = this;
    unsigned int stack_idx = 0;
    while (stack_idx < StateSignature::STACK_LOCALITY){

        signature.stack[stack_idx].top = current->top;
        signature.stack[stack_idx].left = current->left;
        signature.stack[stack_idx].right = current->right;
        signature.stack[stack_idx].htop = current->htop;
        signature.stack[stack_idx].hleft = current->hleft;
        signature.stack[stack_idx].hright = current->hright;
        signature.stack[stack_idx].left_corner = current->left_corner;
        signature.stack[stack_idx].right_corner = current->right_corner;
        ++stack_idx;
        if (current->is_init()){break;}
        current = current->stack_prev;
    }
    signature.stack_size = stack_idx;
    signature.input_sequence = input_sequence;
    signature.N = input_length;
    signature.J = this->J;
}

void ParseState::get_signature(StateSignature &sig,InputDag const *input_sequence,unsigned int input_length){
  make_signature(sig,input_sequence,input_length);
}



void ParseState::encode(vector<unsigned int> &xvec,SparseEncoder &spencoder,StateSignature const &signature){
    spencoder.encode(xvec,signature,true);
}

void ParseState::encode(SparseFloatVector &xvec,SparseEncoder &spencoder,StateSignature const &signature){
    spencoder.encode(xvec,signature,true);
}


AbstractParseTree* ParseState::make_best_tree(InputDag &input_sequence){
    ParseDerivation deriv(this);
    return deriv.as_tree(input_sequence);
    
}

ParseState* ParseState::from_binarytreeR(AbstractParseTree const *root,ParseState *current,SrGrammar const &grammar,InputDag &input_sequence,bool tagger_deriv){
    
    IntegerEncoder *enc = IntegerEncoder::get();

    if(tagger_deriv && root->is_leaf()){

        return current->shift_tag(enc->get_code(root->get_label(),IntegerEncoder::PS_COLCODE),grammar,input_sequence,0.0);

    }else if(!tagger_deriv && root->is_leaf()){

        return current->shift(grammar,input_sequence,0.0);
        
    }else if (root->arity()==1){//unary node

        ParseNode const *rroot = static_cast<ParseNode const*>(root);
        current = from_binarytreeR(rroot->get_child_at(0),current,grammar,input_sequence,tagger_deriv);
        return current->reduce_unary(enc->get_code(rroot->get_label(),IntegerEncoder::PS_COLCODE),grammar,0.0);
        
    }else if(root->arity()==2){//binary nodes

        ParseNode const *rroot = static_cast<ParseNode const*>(root);

        AbstractParseTree *c0 = rroot->get_child_at(0);
        AbstractParseTree *c1 = rroot->get_child_at(1);

        //Ghost reductions
        current = from_binarytreeR(c0,current,grammar,input_sequence,tagger_deriv);
        if (c0->is_leaf() && !tagger_deriv){current = current->ghost_reduce(grammar,0.0);}
        current = from_binarytreeR(c1,current,grammar,input_sequence,tagger_deriv);
        if (c1->is_leaf() && !tagger_deriv){current = current->ghost_reduce(grammar,0.0);}
        
        if(rroot->get_child_at(0)->is_head()){
            return current->reduce_left(enc->get_code(rroot->get_label(),IntegerEncoder::PS_COLCODE),grammar,0.0);
        }else{
            return current->reduce_right(enc->get_code(rroot->get_label(),IntegerEncoder::PS_COLCODE),grammar,0.0);
        }
    }
    //heavy bug if this point is reached.
    return NULL;
}

ParseState* ParseState::from_tree(AbstractParseTree const *root,SrGrammar const &grammar,InputDag &input_sequence,bool tagger_deriv){
    return from_binarytreeR(root,ParseState::init_state(),grammar,input_sequence,tagger_deriv);
}

bool ParseState::is_hamming_equivalent(ParseState &other,InputDag const *input_sequence,unsigned int N){
    
    StateSignature sigA;
    StateSignature sigB;
    this->make_signature(sigA,input_sequence,N);
    other.make_signature(sigB,input_sequence,N);
    
    if(sigA.J != sigB.J){return false;}
    if(sigA.stack_size == sigB.stack_size){
        for(int i = 0; i < sigA.stack_size-1;++i){
            if(sigA.stack[i] != sigB.stack[i]){return false;}
        }
        return    *(sigA.stack[sigA.stack_size-1].htop) == *(sigB.stack[sigA.stack_size-1].htop)
        && sigA.stack[sigA.stack_size-1].top  == sigB.stack[sigA.stack_size-1].top;
    }
    return false;
}


ParseDerivation::ParseDerivation(ParseState *last){
    
    ParseState *current = last;
    while(!current->is_init()){
        derivation.push_front(new ParseState(*current));
        current = current->history_prev;
    }
    derivation.push_front(new ParseState(*current));
}

ParseDerivation::ParseDerivation(AbstractParseTree const *root,SrGrammar const &grammar,InputDag &input_sequence,bool tagger_deriv){
    assert(input_sequence.size() > 0);
    ParseState *current = ParseState::from_tree(root,grammar,input_sequence,tagger_deriv);
    while(!current->is_init()){
        derivation.push_front(current);
        current = current->history_prev;
    }
    derivation.push_front(current);
}

ParseDerivation::ParseDerivation(ParseDerivation const &other){
    for(int i = 0; i < derivation.size();++i){
        derivation.push_back(new ParseState(*(this->derivation[i])));
    }
}

ParseDerivation::~ParseDerivation(){
    for(int i = 0; i < this->size();++i){
        delete derivation[i];
    }
    derivation.clear();
}


AbstractParseTree* ParseDerivation::as_tree(InputDag const &input_sequence,bool morph)const{
    
    stack<AbstractParseTree*> S;
    IntegerEncoder *enc = IntegerEncoder::get();

    for(int i = 1; i < this->size();++i){//skips dummy start state

        ParseState *current = derivation[i];
        
        switch(current->incoming_action.action_type){
            case ParseAction::SHIFT:
            {
                if(morph){
                    ParseToken *p = new ParseToken(*(current->htop));
                    p->set_label(enc->decode(current->top));
                    S.push(p);
                }else{
                    ParseToken *p = new ParseToken(*input_sequence[current->getJ()]);
                    S.push(p);
                }
                break;
            }
            case ParseAction::RL:
            {
                ParseNode *p = new ParseNode(enc->decode(current->top));
                AbstractParseTree *c1 = S.top();
                S.pop();
                AbstractParseTree *c2 = S.top();
                S.pop();
                p->add_child(c2);
                p->add_child(c1);
                c2->set_head();
                S.push(p);
                break;
            }
            case ParseAction::RR:
            {
                ParseNode *p = new ParseNode(enc->decode(current->top));
                AbstractParseTree *c1 = S.top();
                S.pop();
                AbstractParseTree *c2 = S.top();
                S.pop();
                p->add_child(c2);
                p->add_child(c1);
                c1->set_head();
                S.push(p);
                break;
            }
            case ParseAction::RU:
            {
                ParseNode *p = new ParseNode(enc->decode(current->top));
                AbstractParseTree *c1 = S.top();
                S.pop();
                p->add_child(c1);
                c1->set_head();
                S.push(p);
                break;
            }
            case ParseAction::GR:
            {
                break;
            }
            case ParseAction::NULL_ACTION:
            {
                break;
            }
        }
    }
    S.top()->set_head();
    return S.top();
}

float ParseDerivation::weight_at(unsigned int idx)const{
    return derivation[idx]->weight();
}

ParseState* ParseDerivation::operator[](unsigned int idx)const{
    return derivation[idx];
}

ostream& operator<<(ostream &os,ParseState const &state){

    string acode[] = {"S","RL","RR","RU","GR"};

    if(state.incoming_action.action_type == ParseAction::NULL_ACTION){
        return os << "<START>:0";
    }
    if (state.incoming_action.action_type == ParseAction::SHIFT){
        if (state.incoming_action.tok_code != 0){
            return os << "<" << state.J << "," <<  acode[state.incoming_action.action_type] << ","<< IntegerEncoder::get()->decode8(state.incoming_action.tok_code)<< ">:"<<state.weight();
        }else{return os << "<" << state.J << "," <<  acode[state.incoming_action.action_type] << ",**>:"<<state.weight();}
    }
    if (state.incoming_action.action_type == ParseAction::GR){
        return os << "<" << state.J << "," <<  acode[state.incoming_action.action_type] << ",**>:"<<state.weight();
    }
    return os << "<" << state.J << "," <<  acode[state.incoming_action.action_type] << ","<< IntegerEncoder::get()->decode8(state.incoming_action.tok_code)<< ">:"<<state.weight();
}

ostream& operator<<(ostream &os,ParseDerivation const &derivation){
    os <<"<START>:0";
    for(int i = 1; i < derivation.size();++i){
        os << " " <<(*derivation[i]);
    }
    return os;
    //return os << "  >>" << derivation.size()<<"<<"<<endl;
}


void ParseDerivation::reweight_derivation(SparseEncoder &spencoder,
                                          SparseFloatMatrix const &model,
                                          SrGrammar const &grammar,
                                          InputDag const *input_sequence,
                                          size_t N){
    
    vector<unsigned int> xvec(spencoder.ntemplates());
    StateSignature sig;

    for(int i = 0; i < derivation.size()-1;++i){
      derivation[i]->get_signature(sig,input_sequence,N);
      derivation[i]->encode(xvec,spencoder,sig);
      derivation[i+1]->prefix_weight = derivation[i]->prefix_weight + model.dot(xvec,grammar.get_action_index(derivation[i+1]->incoming_action));
    }
}

unsigned int ParseDerivation::hamming(ParseDerivation const &other,unsigned int t,InputDag const *input_sequence){
    
    int N = std::min<int>(this->size(),other.size());
    int d = this->size() > other.size() ? this->size()-other.size() : other.size()-this->size();
    for(int i = 1; i < N;++i){
        if(! derivation[i]->is_hamming_equivalent(*other.derivation[i],input_sequence,input_sequence->size())){++d;}
    }
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TSSBeam::TSSBeam(){}
TSSBeam::TSSBeam(size_t beam_size,SrGrammar const &grammar){
    this->beam_size = beam_size;
    grammar.get_ordered_actions(this->action_vector);
    this->A = action_vector.size();
    ParseState *init_state = ParseState::init_state();
    vector<ParseState*>initv;
    initv.push_back(init_state);
    states.push_back(initv);
    timestep = 0;
    morph = grammar.has_morph();
}

TSSBeam::TSSBeam(TSSBeam const &other){
    this->beam_size = other.beam_size;
    this->action_vector = other.action_vector;
    this->A = other.A;
    ParseState *init_state = ParseState::init_state();
    vector<ParseState*>initv;
    initv.push_back(init_state);
    states.push_back(initv);
    timestep = other.timestep;
    morph = other.morph;
}

TSSBeam& TSSBeam::operator=(TSSBeam const &other){
    this->beam_size = other.beam_size;
    this->action_vector = other.action_vector;
    this->A = other.A;
    ParseState *init_state = ParseState::init_state();
    vector<ParseState*>initv;
    initv.push_back(init_state);
    states.push_back(initv);
    timestep = other.timestep;
    morph = other.morph;
    return *this;
}

TSSBeam::~TSSBeam(){
    reset();
    delete states[0][0];
}

void TSSBeam::reset(){
    this->timestep = 0;
    candidates_buffer.clear();
    success_states.clear();
    for(int i = 1; i < states.size();++i){
        for(int j = 0; j  < states[i].size();++j){
            delete states[i][j];
        }
        states[i].clear();
    }
    states.resize(1);
    succ_sorted = false;
}

ParseState* TSSBeam::operator[](unsigned int idx){
    return states[timestep][idx];
}

size_t TSSBeam::top_size()const{
    return states[timestep].size();
}

bool TSSBeam::has_best_parse()const{return !success_states.empty();}
size_t TSSBeam::num_parses()const{return success_states.size();}

bool success_cmp(ParseState const *lhs, ParseState const *rhs){return lhs->weight() > rhs->weight();}

ParseState* TSSBeam::best_success_state(){
    return kth_success_state(0);
}

ParseState* TSSBeam::kth_success_state(int k){
    if(!succ_sorted){
        std::sort(success_states.begin(),success_states.end(),success_cmp);
        succ_sorted = true;
    }
    return success_states[k];
}



AbstractParseTree* TSSBeam::best_parse(InputDag const &input_sequence){
    return kth_best_parse(input_sequence,0);
}

AbstractParseTree* TSSBeam::kth_best_parse(InputDag const &input_sequence,unsigned int k){
    if(!succ_sorted){
        std::sort(success_states.begin(),success_states.end(),success_cmp);
        succ_sorted = true;
    }
    ParseDerivation d(success_states[k]);
    return ParseDerivation(success_states[k]).as_tree(input_sequence,this->morph);
}


//puts new candidates in the candidates pool
void TSSBeam::push_candidates(ParseState* from_state,vector<float> &y_scores){ //push new candidates into the candidate pool
    for(int i = 0; i < A; ++i){
        candidates_buffer.push_back(make_tuple(from_state,action_vector[i],from_state->weight()+y_scores[i]));
    }
}

//Beam K-argmax related functions
bool cand_cmp(TSSBeam::CANDIDATE const &lhs, TSSBeam::CANDIDATE const &rhs){return std::get<2>(lhs) > std::get<2>(rhs);}

void TSSBeam::next_step(SrGrammar const &grammar,InputDag &input_sequence,size_t N){
    
    ++timestep;
    
    vector<ParseState*> top;
    //std::sort(candidates_buffer.begin(),candidates_buffer.end(),cand_cmp);
     #ifdef FASTS
        //Quickselect style for K-argmaxing
        std::partial_sort(candidates_buffer.begin(),candidates_buffer.begin()+beam_size,candidates_buffer.end(),cand_cmp);
    #else
        std::stable_sort(candidates_buffer.begin(),candidates_buffer.end(),cand_cmp); //@@max : changed to stable sort, ensures consistent choice when several actions have same score.
    #endif
    unsigned int k = 0;
    unsigned int tagger_beam = floor(sqrt(beam_size));//alt :  tagger_beam = beam_size to remove the constraint.
    unsigned int tagger_k = 0;
    unsigned int eta = morph ? 2 * N - 1 : 3 * N - 1;
    ParseState *prev = NULL;
    unsigned int i = 0;
    while(k < beam_size && i < candidates_buffer.size() && std::get<2>(candidates_buffer[i]) != -std::numeric_limits<float>::infinity()){
        
        ParseState *s = std::get<0>(candidates_buffer[i]);
        ParseAction a = std::get<1>(candidates_buffer[i]);
        
        PLOGGER_ADD_ITEM();
        
        if(s != prev){    // (!!) pointer equality test (assumes that states are contiguous in cand buffer)
            tagger_k = 0;
            prev = s;
        }

        switch(a.action_type){
            case ParseAction::SHIFT:
            {
                if (a.tok_code == 0){
                    top.push_back(s->shift(grammar,input_sequence,std::get<2>(candidates_buffer[i])));
                    ++k;
                }
                else{
                    if(tagger_k < tagger_beam){
                        top.push_back(s->shift_tag(a.tok_code,grammar,input_sequence,std::get<2>(candidates_buffer[i])));
                        ++tagger_k;
                        ++k;
                    }
                    //do not add if tagger beam is full for this incoming state
                }
                break;
            }
            case ParseAction::RL:
            {
                top.push_back(s->reduce_left(a.tok_code,grammar,std::get<2>(candidates_buffer[i])));
                ++k;
                break;
            }
            case ParseAction::RR:
            {
                top.push_back(s->reduce_right(a.tok_code,grammar,std::get<2>(candidates_buffer[i])));
                ++k;
                break;
            }
            case ParseAction::RU:
            {
                top.push_back(s->reduce_unary(a.tok_code,grammar,std::get<2>(candidates_buffer[i])));
                ++k;
                break;
            }
            case ParseAction::GR:
            {
                top.push_back(s->ghost_reduce(grammar,std::get<2>(candidates_buffer[i])));
                ++k;
                break;
            }
            case ParseAction::NULL_ACTION:
            default:
            {
                //serious bug if this point is reached
                cout << "bug in beam selection (next step)"<<endl;
                exit(1);
            }
        }
        //if success state, push it to succ vector
        if(timestep == eta && top.back()->stack_predecessor()->is_init()){
            success_states.push_back(top.back());
        }
        ++i;
    }
    states.push_back(top);
    candidates_buffer.clear();
}

bool TSSBeam::sig_in_kth_beam(StateSignature const &sig,unsigned int k)const{
    
    StateSignature beam_sig;
    for(int i = 0; i < states[k].size();++i){
        states[k][i]->get_signature(beam_sig,sig.input_sequence,sig.N);
        if(sig == beam_sig){
            return true;
        }
    }
    return false;
}

ParseState* TSSBeam::best_at(unsigned int t){
    return kth_at(t,0);
}

ParseState* TSSBeam::kth_at(unsigned int t,unsigned int k){
    std::sort(states[t].begin(),states[t].end(),success_cmp);
    return states[t][k];
}

size_t TSSBeam::size_at(unsigned int t)const{
    if (t < states.size()){
        return states[t].size();
    }
    return 0;
}


bool TSSBeam::weak_equivalence(const StateSignature &sigA, const StateSignature &sigB)const{
    return (sigA == sigB);
}

void TSSBeam::display_kbest_derivations(){//debug method displaying the k-best derivations
    for(int i = 0; i < num_parses();++i){
        ParseDerivation d(success_states[i]);
        cout << d << endl;
    }
}

void TSSBeam::dump_beam(){ //dumps the whole beam on stdout
    
    for(int t = 0; t < states.size();++t){
        cout << "@t:";
        for(int j = 0;j < size_at(t);++j){
            cout << *states[t][j] << " ";
        }
        cout << endl;
    }
    cout << "Size >> "<<states.size() << " <<" <<  endl;

}
    
    
