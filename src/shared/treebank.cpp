#include "treebank.h"
#include "str_utils.h"
#include <algorithm>
#include <stack>
#include <random>
#include <string>


ParseNode::ParseNode(PSTRING const &label,bool head){
    this->label = label;
    this->head = head;
}

ParseNode::ParseNode(const ParseNode &other){
    this->head = other.head;
    this->label = other.label;
    this->children = other.children;
}

ParseToken::ParseToken(vector<PSTRING> const &tok_fields,bool head){
    this->tok_fields = tok_fields;
    this->head=head;
}


ParseToken::ParseToken(InputToken const &other,bool head){
    IntegerEncoder *enc = IntegerEncoder::get();
    for(int i = 0; i < other.size();++i){
        tok_fields.push_back(enc->decode(other[i]));
    }
}

ParseToken::ParseToken(const ParseToken &other){
    this->tok_fields = other.tok_fields;
    head = other.head;
}

void ParseToken::add_child(AbstractParseTree *child){
    cerr << "Implementation error (adding child to tree leaf node !):aborting" << endl;
    exit(1);
}

AbstractParseTree* ParseToken::get_child_at(unsigned int idx)const{
    cerr << "Error trying to access child of leaf node : aborting"<<endl;
    exit(1);
}

void get_symbols(AbstractParseTree *root,unordered_set<PSTRING> &symbol_set){
  
  if (!(root->is_leaf())){
    symbol_set.insert(root->get_label());
    for(int i = 0; i < root->arity();++i){
      get_symbols(((ParseNode*)root)->get_child_at(i),symbol_set);
    } 
  }
}

void get_tags(AbstractParseTree *root,unordered_set<PSTRING> &tag_set){           // appends word occurrences to the wordlist vector
  
  if(root->is_leaf()){
    tag_set.insert(root->get_clabel_at(1));
  }else{
    for(int i = 0; i < root->arity();++i){
      get_tags(root->get_child_at(i),tag_set);
    }
  }
}

void get_wordlist(AbstractParseTree *root,vector<PSTRING> &wordlist){
  if(root->is_leaf()){
    wordlist.push_back(root->get_clabel_at(0));
  }else{
    for(int i = 0; i < root->arity();++i){
      get_wordlist(root->get_child_at(i),wordlist);
    }
  }
}

PSTRING get_axiom(AbstractParseTree *root){
  return root->get_label();
}

int tree_yield(AbstractParseTree const *root,InputDag &result,int idx){
  
  if (!(root->is_leaf())){
    int jdx = idx;
    for(int i = 0; i < root->arity();++i){
        jdx = tree_yield(((ParseNode*)root)->get_child_at(i),result,jdx);
    }
    return jdx;
  }else{
    ParseToken *r = (ParseToken*)root;
    result.push_back(new InputToken(r->tok_fields,idx,idx+1));
    return idx+1;
  }
}

void encode_tree(AbstractParseTree *root){
  
  IntegerEncoder *enc =  IntegerEncoder::get();
  if (!root->is_leaf()){//recurrence
    enc->encode(root->get_label(),IntegerEncoder::PS_COLCODE);
    for(int i = 0; i < root->arity();++i){
      encode_tree(((ParseNode*)root)->get_child_at(i));
    }
  }else{//encode leaves
      for(int i = 0; i < ((ParseToken*)root)->lexfields_size();++i){
          if(i == 1){enc->encode(root->get_clabel_at(i),IntegerEncoder::PS_COLCODE);}//also encode TAGS as Non terminal symbols
          enc->encode(root->get_clabel_at(i),i);
      }
  }
}


void destroy(AbstractParseTree *root){
  if (!(root->is_leaf())){
    for(int i = 0; i < root->arity();++i){
      destroy(((ParseNode*)root)->get_child_at(i));
    } 
  }
  delete root;
}

AbstractParseTree* clone(const AbstractParseTree *root){
  if (!(root->is_leaf())){
    ParseNode *rootcp = new ParseNode(*((ParseNode*)root));
    rootcp->clear_children(); 
    for(int i = 0; i < root->arity();++i){
      rootcp->add_child(clone(((ParseNode*)root)->get_child_at(i)));
    }
    return rootcp;
  }else{return new ParseToken((*(ParseToken*)root));}
}

pair<unsigned int,unsigned int> extract_triples(AbstractParseTree const *root,set<TRIPLET_T> &triples,unsigned int input_idx){
  
  if (root->is_leaf()){
    return make_pair(input_idx,input_idx+1);
  }else{
    unsigned int start_idx = input_idx;
    unsigned int tmp_idx   = input_idx;
    for(int i = 0; i < root->arity();++i){
     SPAN_T span =  extract_triples((((ParseNode*)root)->get_child_at(i)),triples,tmp_idx);
     tmp_idx = span.second;
    }
    unsigned int end_idx = tmp_idx;
    triples.insert(make_tuple(start_idx,end_idx,root->get_label()));
    return make_pair(start_idx,end_idx);
  }
}

pair<unsigned int,unsigned int> extract_evalb_triples(AbstractParseTree const *root,set<TRIPLET_T> &triples,unsigned int input_idx,bool top){
    
    if (!root->is_leaf()){
        unsigned int start_idx = input_idx;
        unsigned int tmp_idx   = input_idx;
        for(int i = 0; i < root->arity();++i){
            SPAN_T span =  extract_evalb_triples((((ParseNode*)root)->get_child_at(i)),triples,tmp_idx,false);
            tmp_idx = span.second;
        }
        unsigned int end_idx = tmp_idx;
        if(!top){triples.insert(make_tuple(start_idx,end_idx,root->get_label()));}
        return make_pair(start_idx,end_idx);
    }else{
        return make_pair(input_idx, input_idx+1);
    }
}



tuple<float,float,float> compare(AbstractParseTree const *rootA,AbstractParseTree const *rootB,bool evalb){
  set<TRIPLET_T> t1t;
  if (evalb) {extract_evalb_triples(rootA,t1t,0,true);}else{extract_triples(rootA,t1t,0);}
  set<TRIPLET_T> t2t;
  if(evalb)  {extract_evalb_triples(rootB,t2t,0,true);}else{extract_triples(rootB,t2t,0);}
    
  set<TRIPLET_T> inter_triple;
  set_intersection (t1t.begin(),t1t.end(),t2t.begin(),t2t.end(),inserter(inter_triple, inter_triple.begin()));
  

  float prec = 0;
  float rec = 0;
  float fscore = 0;
  if (t1t.size() == 0 && t2t.size() == 0){
      fscore=1.0;
  }else if(t1t.size() == 0 || t2t.size() == 0){
      fscore = 0.0;
  }else{
    prec = (float)inter_triple.size()/(float)t1t.size();
    rec = (float)inter_triple.size()/(float)t2t.size();
    if((prec+rec) == 0){
      fscore=0;
    }else{
      fscore = 2*prec*rec/(prec+rec);
    }
  }
  return make_tuple(prec,rec,fscore);
}

tuple<float,float,float> compare_evalb(AbstractParseTree const *rootA,AbstractParseTree const *rootB){
  set<TRIPLET_T> t1t;
  extract_evalb_triples(rootA,t1t,0,true);
  set<TRIPLET_T> t2t;
  extract_evalb_triples(rootB,t2t,0,true);


  float gold = t2t.size();
  float pred = t1t.size();

  set<TRIPLET_T> inter_triple;
  set_intersection (t1t.begin(),t1t.end(),t2t.begin(),t2t.end(),inserter(inter_triple, inter_triple.begin()));

  float good = inter_triple.size();

  return make_tuple(good,pred,gold);
}


void count_symbols(AbstractParseTree *root,Counter<PSTRING> &counts){

  counts.add_count(root->get_label());
  if (!(root->is_leaf())){
    for(int i = 0; i < root->arity();++i){
      count_symbols(((ParseNode*)root)->get_child_at(i),counts);
    }
  }
}

void count_words(AbstractParseTree *root,Counter<PSTRING> &counts){

  if (root->is_leaf()){
     counts.add_count(root->get_clabel_at(0));
  }else{  
     for(int i = 0; i < root->arity();++i){
      count_words(((ParseNode*)root)->get_child_at(i),counts);
     }
  }
}

ostream& operator<<(ostream &os,AbstractParseTree const &root){
  if (root.is_leaf()){
    os << "(" << str::encode(root.get_label());
    if (root.is_head()){
      os << "* " << str::encode(root.get_clabel_at(0));
    }else{
      os << " " <<  str::encode(root.get_clabel_at(0));
    }
    return os << ")";
 }else{
        os << "(" << str::encode(root.get_label());
        if (root.is_head()){os << "*";}
        for (int i = 0; i < root.arity();++i){
	  os << " " << *root.get_child_at(i);
	}
        return os << ")";
    }
}

ostream& ptb_flush(ostream &os,AbstractParseTree const *root,InputDag const &input_sequence,int &idx){
    IntegerEncoder *enc = IntegerEncoder::get();
    if (root->is_leaf()){
        os << "(" << str::encode(enc->decode(input_sequence[idx]->value_at(1))) << " " <<  str::encode(input_sequence[idx]->get_wordform()) << ")";
        ++idx;
        return os;
    }else{
        os << "(" << str::encode(root->get_label());
        for (int i = 0; i < root->arity();++i){
            os << " " ;
            ptb_flush(os,root->get_child_at(i),input_sequence,idx);
        }
        return os << ")";
    }
}

ostream& native_flush(ostream &os,AbstractParseTree const *root,InputDag const &input_sequence,int &idx,int padding){
    
    string pad(padding,' ');
    
    if(root->is_leaf()){
        IntegerEncoder *enc = IntegerEncoder::get();
        os << pad;
        for(int j = 0; j < input_sequence[idx]->size();++j){              //   ((ParseToken*)root)->lexfields_size();++j){
            if (j==0){
                os << str::encode(input_sequence[idx]->get_wordform())<<"\t";
            }else{
                if(root->is_head() && j == 1){
                    os << str::encode(enc->decode(input_sequence[idx]->value_at(j)))<< "-head\t";
                }else{os << str::encode(root->get_clabel_at(j))<< "\t";
                    os << str::encode(enc->decode(input_sequence[idx]->value_at(j)))<< "\t";
                }
            }
        }
        ++idx;
        os << endl;
    }else{
        if(root->is_head()){os << pad << "<" << str::encode(root->get_label())<<"-head>"<<endl;}
        else{os << pad << "<" << str::encode(root->get_label())<<">"<<endl;}
        for(int i = 0; i < root->arity();++i){
            native_flush(os, root->get_child_at(i),input_sequence,idx,padding+3);
        }
        os << pad << "</" << str::encode(root->get_label())<<">"<<endl;
    }
    return os;
}

bool get_tree(istream &inStream,AbstractParseTree *&root,int required_arity){

  string u8bfr;
  vector<PSTRING> fields;
  stack<AbstractParseTree*> stck;
  str::SimpleTokenizer sp;
  while(getline(inStream,u8bfr)){
    size_t arity = sp.llex(u8bfr,fields);

    if(arity == 1 && str::xml_beg(fields[0])){//non terminal begin
      ParseNode* n = new ParseNode(str::strip_tag(str::xml_strip(fields[0])),str::is_head(fields[0]));
      if(!stck.empty()){stck.top()->add_child(n);}
      stck.push(n);
    }

    if(arity == 1 && str::xml_end(fields[0])){//non terminal end
      root = stck.top();
      stck.pop();
      if (stck.empty()){return true;}
    }

    if(arity > 1){//terminal

    //error checking
    if(required_arity != -1 && required_arity != arity){
        cerr << "Error : wrong number of fields at line " << u8bfr << endl << "aborting." <<endl;
        exit(1);
    }
    if(str::is_head(fields[1])){
        fields[1] = str::strip_tag(fields[1]);
        ParseToken* n = new ParseToken(fields,true);
        stck.top()->add_child(n);
    }else{
        ParseToken* n = new ParseToken(fields,false);
        stck.top()->add_child(n);
      }
    }
  }
  return false;
}

void parent_decorate(AbstractParseTree *root,PSTRING const &parent){

  if (! root->is_leaf()){
      ParseNode *rroot = static_cast<ParseNode*>(root);
    PSTRING plabel =  rroot->get_label();
    rroot->label += SEP_CODE + parent;
    for(int i = 0; i < root->arity();++i){
      parent_decorate(root->get_child_at(i),plabel); 
    }
  }
}


unsigned int flajolet_decorate(AbstractParseTree *root){
  if(root->is_leaf()){
    return 0;
  }else{
    unsigned int max = 0;
    bool eq = false;
    for(int i = 0; i < root->arity();++i){
      unsigned int F = flajolet_decorate(root->get_child_at(i));
      if (F == max){eq = true;}
      if (F > max){
	max = F;
	eq = false;
      }
    }
    unsigned int H = eq ? max+1 : max;
    ((ParseNode*)root)->label += SEP_CODE + to_wstring(H);
    return H;
  }
}

void undecorate(AbstractParseTree *root){
  if(!root->is_leaf()){
    ParseNode *rroot = static_cast<ParseNode*>(root);
    rroot->label = rroot->label.substr(0,((ParseNode*)root)->label.find_first_of(SEP_CODE));
    for(int i = 0; i < root->arity();++i){
      undecorate(root->get_child_at(i));
    }
  }
}

PSTRING make_symbol(PSTRING const&ancestor,vector<AbstractParseTree*> const&merged,unsigned int order){
  
#ifdef UNIQUE_TEMPORARY
    return UTMP_CODE;
#else
    PSTRING symbol = ancestor;
    for(int i = 0; i < order && i < merged.size() ;++i){
        symbol += TMP_CODE + merged[i]->get_label();
    }
    if(order == 0){symbol += TMP_CODE;}
    return symbol;
#endif

}


void head_markovize(AbstractParseTree *root, PSTRING const &root_sym, unsigned int order){

  if (root->arity() > 2){//binarize this rule
    
    ParseNode *rroot = static_cast<ParseNode*>(root);
    
    bool direction = true;
    AbstractParseTree *unmerged;
    vector<AbstractParseTree*> merged;
    if(rroot->get_child_at(rroot->arity()-1)->is_head()){//last one is head
      unmerged = rroot->get_child_at(0);
      for(int i = 1; i  < rroot->arity();++i){
          merged.push_back(rroot->get_child_at(i));
      }
      rroot->clear_children();
    }else{                                           //head is somewhere in front
      unmerged = rroot->get_child_at(rroot->arity()-1);
      for(int i = 0; i  < rroot->arity()-1;++i){
          merged.push_back(rroot->get_child_at(i));
          direction = false;
      }
      rroot->clear_children();
    }
    ParseNode *p;
    if(root_sym.empty()){
      p = new ParseNode(make_symbol(root->get_label(),merged,order));
    }else{
      p = new ParseNode(make_symbol(root_sym,merged,order));
    }
    p->set_head();
    for(int i = 0; i < merged.size();++i){
      p->add_child(merged[i]);
    }

    //replug unmerged and p into root
    if(direction){
       rroot->add_child(unmerged);
       rroot->add_child(p);
    }else{
       rroot->add_child(p);
       rroot->add_child(unmerged);
    }

    //move on
    head_markovize(unmerged,L"",order);
    if (root_sym.empty()){
      head_markovize(p,rroot->get_label(),order);
    }else{
      head_markovize(p,root_sym,order);
    }
  }else{//basic recursion
    for(int i = 0; i < root->arity();++i){
      head_markovize(root->get_child_at(i),L"",order);
     }
  }
}

bool is_temporary(PSTRING const &symbol){
  return symbol.find_first_of(TMP_CODE) != wstring::npos;
}


void unbinarize(AbstractParseTree *root){
  if(!root->is_leaf()){
    ParseNode *rroot = static_cast<ParseNode*>(root);
    vector<AbstractParseTree*> nchildren;
    for(int i = 0; i < rroot->arity();++i){
      AbstractParseTree* child = rroot->get_child_at(i);
      unbinarize(child);
      if(is_temporary(child->get_label())){
          for(int j = 0; j < child->arity();++j){
              nchildren.push_back(child->get_child_at(j));
          }
          delete child;
      }else{
          nchildren.push_back(child);
      }
    }
    rroot->clear_children();
    for(int i = 0; i < nchildren.size();++i){
      rroot->add_child(nchildren[i]);
    }
  }
}

void unary_closure(AbstractParseTree *root,bool merge_tags){
    
    if(!root->is_leaf()){
    
        ParseNode *rroot = static_cast<ParseNode*>(root);
        for(int i = 0; i < rroot->arity();++i){
            
            AbstractParseTree *c = rroot->get_child_at(i);
            unary_closure(c,merge_tags);
            
            //handles the case where we need to merge with tags
            if(merge_tags && c->arity() == 1 && c->get_child_at(0)->is_leaf()){
                ParseToken *token = static_cast<ParseToken*>(c->get_child_at(0));
                token->set_label(c->get_label()+UNARY_CODE+token->get_label());
                delete c;
                rroot->children[i]=token;
            }
        }
        if(rroot->arity() == 1){
            AbstractParseTree *c =  rroot->get_child_at(0);
            if(!c->is_leaf()){//enforces strict CNF and avoids deleting leaves !
                rroot->label += UNARY_CODE + c->get_label();
                rroot->clear_children();
                for(int i = 0; i < c->arity();++i){
                    rroot->add_child(c->get_child_at(i));
                }
                delete c;
            }
        }
    }
}

/*
void unary_closure(AbstractParseTree *root,bool merge_tags){
   
    if(!root->is_leaf()){
     ParseNode *rroot = static_cast<ParseNode*>(root);
     for(int i = 0; i < rroot->arity();++i){
       unary_closure(rroot->get_child_at(i),merge_tags);
     }
     if(rroot->arity() == 1){
         AbstractParseTree *c =  rroot->get_child_at(0);
         if(!c->is_leaf()){//enforces strict CNF and avoids deleting leaves !
            rroot->label += UNARY_CODE + c->get_label();
            rroot->clear_children();
            for(int i = 0; i < c->arity();++i){
                rroot->add_child(c->get_child_at(i));
            }
            delete c;
        }
     }
   }
}
*/

bool is_unary_closed(PSTRING const &symbol){
  return symbol.find_first_of(UNARY_CODE) != wstring::npos;
}

void unpack_unaries(AbstractParseTree *root){
  if(!root->is_leaf()){
     ParseNode *rroot = static_cast<ParseNode*>(root);
     if(is_unary_closed(rroot->get_label())){
       vector<PSTRING> symbols;
       str::SimpleTokenizer sp(UNARY_CODE,L' ');
       size_t nsyms = sp.llex(rroot->get_label(),symbols);
       rroot->set_label(symbols[0]);
       vector<AbstractParseTree*> tmp_children = rroot->children;
       ParseNode *p = rroot;
       for(int i = 1 ; i < nsyms;++i){
           ParseNode *c = new ParseNode(symbols[i]);
           c->set_head();
           p->clear_children();
           p->add_child(c);
           p = c;
       }
       p->children = tmp_children;
     }
      
     for(int i = 0; i < rroot->arity();++i){
         AbstractParseTree *child = rroot->get_child_at(i);
         if(child->is_leaf() && is_unary_closed(child->get_label())){//deals with leaf nodes in the morph case
             vector<PSTRING> symbols;
             str::SimpleTokenizer sp(UNARY_CODE,L' ');
             size_t nsyms = sp.llex(child->get_label(),symbols);
             ParseNode *nchild = new ParseNode(symbols[0],root->is_head());
             rroot->children[i] = nchild;
             ParseNode *p = nchild;
             for(int i = 1; i < nsyms-1;++i){
                 ParseNode *c = new ParseNode(symbols[i],true);
                 p->add_child(c);
                 p=c;
             }
             p->add_child(child);
             child->set_label(symbols[nsyms-1]);
         }else{
             unpack_unaries(child);
         }
     }
  }
}

bool head_check(AbstractParseTree *root,bool isroot){
  
  if (isroot && !root->is_head()){return false;}
  if (!root->is_leaf()){
    unsigned int nheads = 0;
    for(int i = 0; i < root->arity();++i){
      if(root->get_child_at(i)->is_head()){nheads++;}
      if (!head_check(root->get_child_at(i),false)){return false;}
    }
    if (nheads != 1){return false;}
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////

Treebank::Treebank(){}

Treebank::Treebank(size_t N){//allocates an empty treebank of size N
    tree_list.resize(N);
}

Treebank::Treebank(Treebank const &treebank){
  for(int i = 0; i < this->size();++i){destroy(tree_list[i]);}
  tree_list.clear();
  for(int i = 0; i < treebank.size();++i){
    tree_list.push_back(clone(treebank[i]));
  }
  colnames = treebank.colnames;
}

Treebank& Treebank::operator=(Treebank const &treebank){
  for(int i = 0; i < this->size();++i){destroy(tree_list[i]);}
  tree_list.clear();
  for(int i = 0; i < treebank.size();++i){
    tree_list.push_back(clone(treebank[i]));
  }
  colnames = treebank.colnames;
  return *this;
}

Treebank::~Treebank(){
  this->clear();
}

void Treebank::clear(){
 for(int i = 0; i < this->size();++i){
     if(tree_list[i]!=NULL){destroy(tree_list[i]);}
 }
  tree_list.clear();
}

vector<PSTRING> Treebank::update_trees(string const &filename){
  ifstream is(filename);
  if (!is.is_open()){
    cerr << "Error:: wrong treebank filename, aborting."<<endl;
    exit(1);
  }

  //Process header
  unsigned int K = 0;
  str::SimpleTokenizer sp;
  string bfr;
  vector<PSTRING> header;  
  while(getline(is,bfr)){
    header.clear();
    K = sp.llex(bfr,header);
    if(K > 0){break;}
  }

  AbstractParseTree *tree_bfr;
  while(get_tree(is,tree_bfr,K)){
     tree_list.push_back(tree_bfr);
  }
  is.close();
  this->colnames = header;
  return header;
}

Treebank::Treebank(string const &filename){
  update_trees(filename);
}

void Treebank::get_allsymbols(vector<PSTRING> &nonterminals,vector<PSTRING> &axioms,vector<PSTRING> &tags)const{
    
    unordered_set<PSTRING> ntsymbol_set;
    for(int i = 0; i < nonterminals.size();++i){
        ntsymbol_set.insert(nonterminals[i]);
    }
    
    unordered_set<PSTRING> axsymbol_set;
    for(int i = 0; i < axioms.size();++i){
        axsymbol_set.insert(axioms[i]);
    }
    unordered_set<PSTRING> tag_set;
    for(int i = 0; i < tags.size();++i){
        tag_set.insert(tags[i]);
    }
    
    for(int i = 0; i < this->size();++i){
        get_symbols(tree_list[i],ntsymbol_set);
        get_tags(tree_list[i],tag_set);
        axsymbol_set.insert(tree_list[i]->get_label());
    }
    nonterminals.clear();
    tags.clear();
    axioms.clear();
    for(unordered_set<PSTRING>::const_iterator it = ntsymbol_set.begin();it != ntsymbol_set.end();++it){
        nonterminals.push_back(*it);
    }
    for(unordered_set<PSTRING>::const_iterator it = tag_set.begin();it != tag_set.end();++it){
        tags.push_back(*it);
    }
    for(unordered_set<PSTRING>::const_iterator it = axsymbol_set.begin();it != axsymbol_set.end();++it){
        axioms.push_back(*it);
    }
}


void Treebank::get_nonterminals(vector<PSTRING> &nonterminals,vector<PSTRING> &axioms)const{
    
  unordered_set<PSTRING> ntsymbol_set;
  for(int i = 0; i < nonterminals.size();++i){
    ntsymbol_set.insert(nonterminals[i]);
  }
    
  unordered_set<PSTRING> axsymbol_set;
  for(int i = 0; i < axioms.size();++i){
    axsymbol_set.insert(axioms[i]);
  }
    
  for(int i = 0; i < this->size();++i){
    get_symbols(tree_list[i],ntsymbol_set);
    axsymbol_set.insert(tree_list[i]->get_label());
  }
  nonterminals.clear();
  axioms.clear();
  for(unordered_set<PSTRING>::const_iterator it = ntsymbol_set.begin();it != ntsymbol_set.end();++it){
      nonterminals.push_back(*it);
  }
  for(unordered_set<PSTRING>::const_iterator it = axsymbol_set.begin();it != axsymbol_set.end();++it){
      axioms.push_back(*it);
  }
}

void Treebank::get_axioms(vector<PSTRING> &axlist)const{
   unordered_set<PSTRING> axioms;
   for(int i = 0; i < this->size();++i){
     axioms.insert(get_axiom(tree_list[i]));
   }
   axlist.clear();
   for(unordered_set<PSTRING>::const_iterator it = axioms.begin();it != axioms.end();++it){
    axlist.push_back(*it);
  }
}

void Treebank::get_tagset(vector<PSTRING> &tags)const{
  unordered_set<PSTRING> tag_set;
  for(int i = 0; i < this->size();++i){
    get_tags(tree_list[i],tag_set);
  }
  tags.clear();
  for(unordered_set<PSTRING>::const_iterator it = tag_set.begin();it != tag_set.end();++it){
    tags.push_back(*it);
  }
}

void Treebank::get_word_tokens(vector<PSTRING> &tokens)const{
  for(int i = 0; i < this->size();++i){
    get_wordlist(tree_list[i],tokens);
  }
}

const AbstractParseTree* Treebank::sample_tree()const{
  return tree_list[rand() % tree_list.size()]; 
}  

void Treebank::shuffle(){
    random_shuffle(tree_list.begin(),tree_list.end());
}

void Treebank::sample_trees(Treebank &sampled, unsigned int N)const{
  vector<unsigned int> idxes;
  for(int i = 0; i < tree_list.size();++i){
    idxes.push_back(i);
  }
  random_shuffle(idxes.begin(),idxes.end());
  sampled.clear();
  for(int i = 0; i < N && i < tree_list.size();++i){
    sampled.add_tree(clone(tree_list[idxes[i]]));
  }
}

void Treebank::add_tree(AbstractParseTree *tree){
  tree_list.push_back(tree);
}

void Treebank::split(Treebank &train,Treebank &dev)const{
  train.clear();
  dev.clear();
  size_t N  = tree_list.size();
  size_t Nt = (size_t) (N * 0.9);
  if (N>1 && Nt == 0){Nt = N-1;--N;}
  if(N==0 || Nt==0){cerr << "Not enough trees for training, bailing out. Sorry."<<endl;exit(0);}
  for (int i = 0; i < Nt;++i){train.add_tree(clone(tree_list[i]));}
  for (int i = Nt; i < N;++i){dev.add_tree(clone(tree_list[i]));}
}

void Treebank::fold(Treebank &train,Treebank &dev,unsigned k,unsigned Kfolds)const{
    
    train.clear();
    dev.clear();
    size_t N  = tree_list.size();
    float fold_size = (float)N / (float)Kfolds;
    
    size_t min_idx = (unsigned)k*fold_size;
    size_t max_idx = (unsigned)((k+1)*fold_size);
    if (max_idx > N){max_idx = N;}
    for(int i = 0 ; i < N ; ++i){
        if (i >= min_idx && i < max_idx){
            dev.add_tree(clone(tree_list[i]));
        }else{
            train.add_tree(clone(tree_list[i]));
        }
    }
    cout << "#train : " << train.size() << endl;
    cout << "#dev   : " << dev.size() << endl;
}


void Treebank::transform(Treebank &transformed,bool merge_tags,unsigned int markov_order,Decoration decoration){
  transformed.clear();
  transformed.colnames = colnames;
  for(int i = 0; i < tree_list.size();++i){
    AbstractParseTree *p = clone(tree_list[i]);
    if(decoration == Treebank::PARENT_DECORATION){parent_decorate(p);}
    if(decoration == Treebank::FLAJOLET_DECORATION){flajolet_decorate(p);}
    head_markovize(p,L"",markov_order);
    unary_closure(p,merge_tags);
    transformed.add_tree(p);
  }
}

void Treebank::detransform(Treebank &detransformed,Decoration decoration){
  detransformed.clear();
  detransformed.colnames = colnames;
  for(int i = 0; i < tree_list.size();++i){
    AbstractParseTree *p = clone(tree_list[i]);
    unpack_unaries(p);
    unbinarize(p);
    if(decoration != Treebank::NO_DECORATIONS){undecorate(p);}
    detransformed.add_tree(p);
  }
}
void Treebank::detransform(Decoration decoration){
    
    for(int i = 0; i < tree_list.size();++i){
        AbstractParseTree *p = clone(tree_list[i]);
        unpack_unaries(p);
        unbinarize(p);
        if(decoration != Treebank::NO_DECORATIONS){undecorate(p);}
        destroy(tree_list[i]);
        tree_list[i] = p;
    }
}


tuple<float,float,float> Treebank::evalb(Treebank const &other){

  tuple<float,float,float> c = make_tuple(0,0,0);
  if(this->size() != other.size()){
    cerr << "Incomparable treebanks while comparing. aborting."<<endl;
    return c;
  }else{
    float p = 0.0;
    float r = 0.0;
    float f = 0.0;
    for(int i = 0; i < this->size();++i){
      tuple<float,float,float> t = compare(tree_list[i],other.tree_list[i]);
      p += get<0>(t);
      r += get<1>(t);
      f += get<2>(t);
    }
    float N = (float) this->size();
    return make_tuple(p/N,r/N,f/N);
  }
}

void Treebank::update_encoder(){
 for(int i = 0; i < this->size();++i){
    encode_tree(tree_list[i]);    
 }
}

ostream& operator<<(ostream &os,const Treebank &treebank){
  for(int i = 0; i < treebank.size();++i){
    os << *treebank[i] << endl;
  }
  return os;
}


ConllGraph::ConllGraph(InputDag &input_sequence){
    this->input_sequence = &input_sequence;
    this->govRelation.resize(input_sequence.size()+1,0);
}

ConllGraph::ConllGraph(AbstractParseTree const *root, InputDag &input_sequence){
    this->input_sequence = &input_sequence;
    this->govRelation.resize(input_sequence.size()+1,0);
    pair<int,int> res = tree_as_graph(root);
    add_edge(res.first,0);//adds edge to dummy root node in conllx format
}

pair<int,int> ConllGraph::tree_as_graph(AbstractParseTree const *root,int startIdx){
    
    if(!root->is_leaf()){
        vector<int> child_indexes;
        int head_index = 0;
        
        int spidx = startIdx;
        
        for(int i = 0; i < root->arity();++i){
            
            AbstractParseTree *child = root->get_child_at(i);
            pair<int,int> idxes = tree_as_graph(child,spidx);
            int cidx = idxes.first;
            spidx    = idxes.second;
            
            if (child->is_head()){
                head_index = cidx;
            }else{
                child_indexes.push_back(cidx);
            }
        }
        for(int i = 0; i < child_indexes.size();++i){add_edge(child_indexes[i],head_index);}
        
        return make_pair(head_index,spidx);
        
    }else{return make_pair(startIdx+1,startIdx+1);}
}

void ConllGraph::add_edge(int dep,int gov){
    govRelation[dep] = gov;
}

ostream& operator<<(ostream &os,ConllGraph &graph){
    IntegerEncoder *enc = IntegerEncoder::get();
    for(int i = 1; i < graph.govRelation.size();++i){
            os << i << "\t" << str::encode(graph.input_sequence->at(i-1)->get_wordform())
                    << "\t" << str::encode(graph.input_sequence->at(i-1)->get_wordform())
                    << "\t" << enc->decode8(graph.input_sequence->at(i-1)->get_catcode())
                    << "\t" << enc->decode8(graph.input_sequence->at(i-1)->get_catcode())
                    << "\t" << "___"
                    << "\t" << graph.govRelation[i]
                    << "\t" << "UNTYPED"
                    << "\t" << "_"
                    << "\t" << "_" << endl;
    }
    return os;
}

PennTreebankOutStream::PennTreebankOutStream(bool active){
    stdout = true;
    this->active = active ;
}
PennTreebankOutStream::PennTreebankOutStream(string const &filename){
    stdout = false;
    os.open(filename);
    active = true;
    this->filename = filename;
}

PennTreebankOutStream::PennTreebankOutStream(PennTreebankOutStream const &other){
    stdout = other.stdout;
    this->filename = other.filename;
    os.open(other.filename);
    active = other.active;
}


PennTreebankOutStream& PennTreebankOutStream::operator=(PennTreebankOutStream const &other){
    stdout = other.stdout;
    os.open(other.filename);
    active = other.active;
    this->filename = other.filename;
    return *this;
}

void PennTreebankOutStream::flush_parse(AbstractParseTree const *root,InputDag &input_sequence){
    if(active){
        int idx = 0;
        if(stdout){
            ptb_flush(cout,root,input_sequence,idx);
            cout <<endl;
        }
        else{
            ptb_flush(os,root,input_sequence,idx);
            os <<endl;
        }
    }
}

void PennTreebankOutStream::flush_failure(InputDag &input_sequence){
    if(active){
        if(stdout){cout << "(())" <<endl;}
        else{os << "(())"<<endl;}
    }
}

void PennTreebankOutStream::close(){if (!stdout){os.close();stdout=true;active=false;}}




NativeOutStream::NativeOutStream(bool active){
    stdout = true;
    this->active = active ;
}
NativeOutStream::NativeOutStream(string const &filename){
    stdout = false;
    os.open(filename);
    active = true;
    this->filename = filename;
}

NativeOutStream::NativeOutStream(NativeOutStream const &other){
    stdout = other.stdout;
    this->filename = other.filename;
    os.open(other.filename);
    active = other.active;
}


NativeOutStream& NativeOutStream::operator=(NativeOutStream const &other){
    stdout = other.stdout;
    os.open(other.filename);
    active = other.active;
    this->filename = other.filename;
    return *this;
}

void NativeOutStream::flush_parse(AbstractParseTree const *root,InputDag &input_sequence){
    if(active){
        int idx = 0;
        if(stdout){
            native_flush(cout,root,input_sequence,idx,0);
            cout <<endl;
        }
        else{
            native_flush(os,root,input_sequence,idx,0);
            os <<endl;
        }
    }
}

void NativeOutStream::flush_failure(InputDag &input_sequence){
    if(active){
        IntegerEncoder *enc = IntegerEncoder::get();
        if(stdout){
            cout << "<FAILURE>"<<endl;
            for(int i = 0; i < input_sequence.size();++i){
                for(int j = 0; j < input_sequence[i]->size();++j){
                    cout << enc->decode8((*input_sequence[i])[j])+"\t";
                }
                cout << endl;
            }
            cout << "</FAILURE>"<<endl;

        }else{
            os << "<FAILURE>"<<endl;
            for(int i = 0; i < input_sequence.size();++i){
                for(int j = 0; j < input_sequence[i]->size();++j){
                    os << enc->decode8((*input_sequence[i])[j])+"\t";
                }
                os << endl;
            }
            os << "</FAILURE>"<<endl;
        }
    }
}

void NativeOutStream::close(){if (!stdout){os.close();stdout=true;active=false;}}





ConllOutStream::ConllOutStream(bool active){//cout stream
    stdout = true;
    this->active = active;
}
ConllOutStream::ConllOutStream(string const &filename){
    this->filename = filename;
    stdout = false;
    os.open(filename);
    active = true;
}

void ConllOutStream::open(string const &filename){
    stdout = false;
    os.open(filename);
    active=true;
}

void ConllOutStream::flush_parse(AbstractParseTree const *root,InputDag &input_sequence){
    ConllGraph graph(root, input_sequence);
    if(active){
        if(stdout){
            cout << graph << endl;
        }else{
            os << graph << endl;
        }
    }
}

ConllOutStream::ConllOutStream(ConllOutStream const &other){
    stdout = other.stdout;
    this->filename = other.filename;
    os.open(filename);
    active = other.active;
}


ConllOutStream& ConllOutStream::operator=(ConllOutStream const &other){
    stdout = other.stdout;
    this->filename = other.filename;
    os.open(filename);
    os.flush();
    active = other.active;
    return *this;
}

void ConllOutStream::flush_failure(InputDag &input_sequence){
    ConllGraph graph(input_sequence);
    if(active){
        if(stdout){
            cout << graph << endl;
        }else{
            os << graph << endl;
        }
    }
}

void ConllOutStream::close(){if (!stdout){os.close();stdout=true;active=false;}}


void MultiOutStream::flush_parse(AbstractParseTree const *root, InputDag &input_sequence){
    ptb_out.flush_parse(root,input_sequence);
    conll_out.flush_parse(root,input_sequence);
    native_out.flush_parse(root,input_sequence);
}

void MultiOutStream::flush_failure(InputDag &input_sequence){
    ptb_out.flush_failure(input_sequence);
    conll_out.flush_failure(input_sequence);
    native_out.flush_failure(input_sequence);
}
void MultiOutStream::close(){
    ptb_out.close();
    conll_out.close();
    native_out.close();
}

void MultiOutStream::setPennTreebankOutfile(string const &filename){
    ptb_out = PennTreebankOutStream(filename);
    
}

void MultiOutStream::setConllOutfile(string const &filename){
    conll_out.open(filename);
}

void MultiOutStream::setNativeOutfile(string const &filename){
    native_out = NativeOutStream(filename);
}






