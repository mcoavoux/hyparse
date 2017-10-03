#include "forest.h"
#include <limits>

// 32 bit hashing (Jenkins)
#define __rot__(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

#define __mix__(a,b,c) \
{ \
a -= c;  a ^= __rot__(c, 4);  c += b; \
b -= a;  b ^= __rot__(a, 6);  a += c; \
c -= b;  c ^= __rot__(b, 8);  b += a; \
a -= c;  a ^= __rot__(c,16);  c += b; \
b -= a;  b ^= __rot__(a,19);  a += c; \
c -= b;  c ^= __rot__(b, 4);  b += a; \
}

#define __final__(a,b,c) \
{ \
c ^= b; c -= __rot__(b,14); \
a ^= c; a -= __rot__(c,11); \
b ^= a; b -= __rot__(a,25); \
c ^= b; c -= __rot__(b,16); \
a ^= c; a -= __rot__(c,4);  \
b ^= a; b -= __rot__(a,14); \
c ^= b; c -= __rot__(b,24); \
}

uint32_t hashword(uint32_t const *k,size_t length, uint32_t initval){
  
  uint32_t a,b,c;

  /* Set up the internal state */
  a = b = c = 0xdeadbeef + (((uint32_t)length)<<2) + initval;

  /*------------------------------------------------- handle most of the key */
  while (length > 3)
  {
    a += k[0];
    b += k[1];
    c += k[2];
    __mix__(a,b,c);
    length -= 3;
    k += 3;
  }

  /*------------------------------------------- handle the last 3 uint32_t's */
  switch(length)                     /* all the case statements fall through */
  { 
  case 3 : c+=k[2];
  case 2 : b+=k[1];
  case 1 : a+=k[0];
    __final__(a,b,c);
  case 0:     /* case 0: nothing left to add */
    break;
  }
  /*------------------------------------------------------ report the result */
  return c;
}

size_t VertexHasher::operator()(Vertex const *v)const{
 
  unsigned int a = v->i;
  unsigned int b = v->j;
  unsigned int c = v->h;
         
  __mix__(a,b,c);
          
  a += v->symbol; 
 
  __final__(a,b,c);
  
  return c;

}

size_t EdgeHasher::operator()(Edge const *e)const{

  if (e->arity() == 1){
    uint32_t vals[8];
    vals[0] = e->head->i;
    vals[1] = e->head->j;
    vals[2] = e->head->h;
    vals[3] = e->head->symbol;
    vals[4] = e->tail[0]->i;
    vals[5] = e->tail[0]->j;
    vals[6] = e->tail[0]->h;
    vals[7] = e->tail[0]->symbol;
    return hashword(vals,8, 17);

  }else{//arity == 2
    uint32_t vals[12];
    vals[0] = e->head->i;
    vals[1] = e->head->j;
    vals[2] = e->head->h;
    vals[3] = e->head->symbol;
    vals[4] = e->tail[0]->i;
    vals[5] = e->tail[0]->j;
    vals[6] = e->tail[0]->h;
    vals[7] = e->tail[0]->symbol;
    vals[8] = e->tail[1]->i;
    vals[9] = e->tail[1]->j;
    vals[10] = e->tail[1]->h;
    vals[11] = e->tail[1]->symbol;
    return hashword(vals,12, 17);
  }
}

bool VertexEquality::operator()(Vertex const *v1,Vertex const *v2)const{return *v1 == *v2;}

bool EdgeEquality::operator()(Edge const *e1,Edge const *e2)const{
  if(e1->arity() != e2->arity()){return false;}
  if( *(e1->head) != *(e2->head)){return false;}
  for(int i = 0; i < e1->arity();++i){
    if (*(e1->tail[i]) != *(e2->tail[i])){return false;}
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////

Vertex::Vertex(TOK_CODE symbol,int i,int j,int h){
  this->symbol = symbol;
  this->i = i;
  this->j = j;
  this->h = h;
}
  
ostream& operator<<(ostream &os, Vertex const &v){
  IntegerEncoder *enc = IntegerEncoder::get();
  return os << "<" <<  enc->decode8(v.symbol)<< "[" << v.h << "]" << "," << v.i << "," << v.j <<  ">";
}

bool Vertex::operator==(Vertex const &other)const{return this->i == other.i && this->j == other.j && this->h == other.h && this->symbol == other.symbol;}
bool Vertex::operator!=(Vertex const &other)const{return !(*this == other);}

///////////////////////////////////////////////////////////////////////////


Edge::Edge(Vertex *head,Vertex *tail,float weight){
  this->head = head;
  this->tail.push_back(tail);  
  this->weight = weight;
}

Edge::Edge(Vertex *head,Vertex *tail_lhs,Vertex *tail_rhs,float weight){

  this->head = head;
  this->tail.push_back(tail_lhs);  
  this->tail.push_back(tail_rhs); 
  this->weight = weight;
}


ostream& operator<<(ostream &os, Edge const &v){
  os << *(v.head) << " -->";
  for(int i = 0; i < v.arity();++i){
    os << " "<<*(v.tail[i]);
  }
  return os << " :: " << v.get_weight();
}

///////////////////////////////////////////////////////////////////////////

BackDerivation::BackDerivation(){//leaf derivation
    inside_weight = 0;
}

BackDerivation::BackDerivation(Vertex *v){//leaf derivation
    inside_weight = 0;
    head = v;
}

BackDerivation::BackDerivation(Vertex *head,BackDerivation *child,double local_weight){
    inside_weight = child->get_inside_weight() + local_weight;
    this->head = head;
    tail.push_back(child);
}

BackDerivation::BackDerivation(Vertex *head,BackDerivation *left_child,BackDerivation *right_child,double local_weight){
    inside_weight = left_child->get_inside_weight() + right_child->get_inside_weight() + local_weight;
    this->head = head;
    tail.push_back(left_child);
    tail.push_back(right_child);

}

double BackDerivation::get_inside_weight()const{return inside_weight;}
size_t BackDerivation::arity()const{return tail.size();};


AbstractParseTree* BackDerivation::as_tree(vector<InputToken> const &input_sequence,bool root)const{
    IntegerEncoder *enc = IntegerEncoder::get();
    size_t a = arity();
    switch(a){
        case 0:
            return new ParseToken(input_sequence[this->head->h]);//true issue : bring back input sequence
        case 1:
        {
            AbstractParseTree *child = tail[0]->as_tree(input_sequence);
            child->set_head();
            ParseNode *n = new ParseNode(enc->decode(this->head->symbol));
            n->add_child(child);
            if (root){n->set_head();}
            return n;
        }
        case 2:
        {
            AbstractParseTree *childL = tail[0]->as_tree(input_sequence);
            AbstractParseTree *childR = tail[1]->as_tree(input_sequence);
            if(this->head->h == tail[0]->head->h){childL->set_head();}else{childR->set_head();}
            ParseNode *n = new ParseNode(enc->decode(this->head->symbol));
            n->add_child(childL);
            n->add_child(childR);
            if (root){n->set_head();}
            return n;
        }
        default:
            cerr << "back derivation crash "<<endl;
            exit(1);
    }
}
///////////////////////////////////////////////////////////////////////////



Forest::Forest(size_t n){
  this->N = n+1;
}

Forest::Forest(AbstractParseTree const *root){
    N = 0;
    add_tree(root);
}


Forest::~Forest(){
  for(unordered_set<Vertex*,VertexHasher,VertexEquality>::iterator it = nodes.begin();it != nodes.end();++it){delete *it;}
  for(unordered_set<Edge*,EdgeHasher,EdgeEquality>::iterator it = edges.begin(); it != edges.end() ; ++it){delete *it;}
}

pair<Vertex*,bool> Forest::add_node(Vertex *v){
  pair<unordered_set<Vertex*,VertexHasher,VertexEquality>::iterator,bool> res = nodes.insert(v);
  if(!res.second){delete v;}               //if not inserted, delete immediately.
  return make_pair(*(res.first),res.second);
}

bool Forest::add_edge(TOK_CODE X,unsigned int i,unsigned int j,TOK_CODE Y ,TOK_CODE Z, unsigned int k, float w,int hT,int hL,int hR){
  
  //add root node 
  Vertex *v = new Vertex(X,i,j,hT);
  pair<Vertex*,bool> resH = add_node(v);
  if(resH.second && i == 0 && j == N){//we have a root node (maybe check on h and symbol to avoid inconsistencies (...)
    root = v;                 // or make root a vector of roots.
  }

  //add tail nodes 
  pair<Vertex*,bool> resTl = add_node(new Vertex(Y,i,k,hL)); 
  pair<Vertex*,bool> resTr = add_node(new Vertex(Z,k,j,hR));
  
  return add_edge(new Edge(resH.first,resTl.first,resTr.first,w));
}

bool Forest::add_edge(TOK_CODE X,unsigned int i,unsigned int j,TOK_CODE Y, float w,int h){
  
  //add root node 
  Vertex *v = new Vertex(X,i,j,h);
  pair<Vertex*,bool> resH = add_node(v);
  if(resH.second && i == 0 && j == N){//we have a root node (maybe check on h and symbol to avoid inconsistencies (...)
    root = v;                 // or make root a vector of roots.
  }
  //add tail node 
  pair<Vertex*,bool> resT = add_node(new Vertex(Y,i,j,h));
  return add_edge(new Edge(resH.first,resT.first,w));
}

bool Forest::add_edge(Edge *e){
  pair< unordered_set<Edge*,EdgeHasher,EdgeEquality>::iterator,bool> res = edges.insert(e);
  //updates star-indexes
  if(res.second){
    bs_index[e->head].insert(e);
    for(int i = 0; i < e->arity();++i){fs_index[e->tail[i]].insert(e);}
  }
  return res.second;
}

tuple<int,int,int> Forest::add_treeR(AbstractParseTree const *root,int startidx){
  
  IntegerEncoder *enc = IntegerEncoder::get();

  if (root->arity() == 2){
    
    tuple<int,int,int> leftC = add_treeR(root->get_child_at(0),startidx);
    tuple<int,int,int> rightC = add_treeR(root->get_child_at(1),get<1>(leftC));
    int i = startidx;
    int k = get<1>(leftC);
    int j = get<1>(rightC);
    int hL = get<2>(leftC);
    int hR = get<2>(rightC);
    int hT = root->get_child_at(0)->is_head() ? hL : hR;
    TOK_CODE tcode = enc->get_code(root->get_label(),IntegerEncoder::PS_COLCODE);
    TOK_CODE lcode = enc->get_code(root->get_child_at(0)->get_label(),IntegerEncoder::PS_COLCODE);
    TOK_CODE rcode = enc->get_code(root->get_child_at(1)->get_label(),IntegerEncoder::PS_COLCODE);
  
    this->add_edge(tcode,i,j,lcode,rcode,k,0,hT,hL,hR);
    return make_tuple(i,j,hT);
  }

  if (root->arity() == 1){
    tuple<int,int,int> child = add_treeR(root->get_child_at(0),startidx);
    int i = startidx;
    int j = get<1>(child);
    int hC = get<2>(child);
    TOK_CODE tcode = enc->get_code(root->get_label(),IntegerEncoder::PS_COLCODE);
    TOK_CODE ccode = enc->get_code(root->get_child_at(0)->get_label(),IntegerEncoder::PS_COLCODE);
    add_edge(tcode,i,j,ccode,0,hC);  
    return make_tuple(i,j,hC);
  }  
  if (root->arity() == 0){return make_tuple(startidx,startidx+1,startidx);}
  cerr << "sth went wrong when converting tree into forest"<<endl;
  exit(1);
}


bool Forest::add_tree(AbstractParseTree const *root){ //this adds a tree to this forest if its yield is compatible (same length)

  InputDag input_sequence;
  tree_yield(root,input_sequence);
    if( N==0 ) {
        this->N = input_sequence.size()+1 ;
        add_treeR(root,0);
        return true;
    }
    if(input_sequence.size()+1 != N){return false;}
    
    add_treeR(root,0);
    return true;
    
}

ostream& Forest::as_grammar(ostream& os)const{
    for(unordered_set<Edge*,EdgeHasher,EdgeEquality>::const_iterator it = edges.begin();it!= edges.end();++it){
        os << **it  << endl;
    }
    return os;
}
                                                                                  
ostream& operator<<(ostream &os,Forest const &forest){return forest.as_grammar(cout);}

bool Forest::is_leaf(Vertex *v)const{
    
    star_const_iterator got = bs_index.find(v);
    return got == bs_index.end();
    
}

void Forest::leaves(vector<Vertex*> &leaf_vec)const{

    for(vertex_const_iterator it = nodes.begin() ; it != nodes.end() ; ++it){
        if( is_leaf(*it)){
            leaf_vec.push_back(*it);
        }
    }
}
 
void Forest::idom_vertices(Vertex *v,VX_SET &idom){
    star_const_iterator got = fs_index.find(v);
    for(edge_const_iterator it = got->second.begin(); it != got->second.end();++it){
        idom.insert((*it)->head);
    }
}

/*
bool make_best_derivation(ParseDerivation const &deriv){
    
    if(deriv[deriv.size()-1]->J == N-1){return false;}
    
    stack<BackDerivation> S;
    for(int i = 1; i < deriv.size();++i){
        
    }
    return true;
}
*/
/*
AbstractParseTree* Forest::viterbi(vector<InputToken> const &input_sequence){
    
    unordered_map<Vertex*,vector<BackDerivation> > viterbi_index; //stores the best derivation for node v
    
    //Finds forest leaves
    vector<Vertex*> input_vertices;
    leaves(input_vertices);
    for(int i = 0; i < input_sequence.size();++i){
        viterbi_index[input_vertices[i]] = BackDerivation(input_vertices[i]);
    }
    
    //init
    VX_SET agenda;
    for(int i = 0; i < input_vertices.size();++i){idom_vertices(input_vertices[i],agenda);}
    
    //fill viterbi index :: proceed in topological order
    while(!agenda.empty()){
        
        for(vertex_const_iterator at = agenda.begin(); at != agenda.end();++at){
            cout << "processing:" << **at << endl;
            double max_inside_weight = numeric_limits<double>::min();
            BackDerivation argmaxderiv;
            star_const_iterator got = bs_index.find(*at);
            for(edge_const_iterator et = got->second.begin(); et != got->second.end();++et){
                if((*et)->arity() == 2){
                    BackDerivation bd(*at, &viterbi_index[(*et)->tail[0]], &viterbi_index[(*et)->tail[1]], (*et)->get_weight());
                    if(bd.get_inside_weight() > max_inside_weight){
                        max_inside_weight = bd.get_inside_weight();
                        argmaxderiv = bd;
                    }
                }
                if((*et)->arity() == 1){
                    BackDerivation bd(*at, &viterbi_index[(*et)->tail[0]], (*et)->get_weight());
                    if(bd.get_inside_weight() > max_inside_weight){
                        max_inside_weight = bd.get_inside_weight();
                        argmaxderiv = bd;
                    }
                }
            }
            viterbi_index[*at] = argmaxderiv;
        }
        //Refill agenda
        VX_SET tmp_agenda;
        for(vertex_const_iterator at = agenda.begin(); at != agenda.end();++at){
                idom_vertices(*at,tmp_agenda);
        }
        agenda = tmp_agenda;
    }
    //Builds and returns the 1-best derivation
    return viterbi_index[this->root].as_tree(input_sequence,true);
}

*/












