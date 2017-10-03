#ifndef FOREST_H
#define FOREST_H

#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "treebank.h"


/**
 * The forest is encoded as a restricted form of hypergraph.
 * We derive and use the representation/terminology of 
 * G. Gallo et al. Directed hypergraphs and applications, Discrete Applied Mathematics, 42 (2-3),1993.
 */
class Vertex{

public:
  friend class EdgeHasher;
  friend class VertexHasher;
  friend class Forest;
  friend class BackDerivation;

  //An item of the form <X,i,j,h> where h is the head index.
  Vertex(TOK_CODE symbol,int i,int j,int h);
  
  friend ostream& operator<<(ostream &os, Vertex const &v);

  bool operator==(Vertex const &other)const;
  bool operator!=(Vertex const &other)const;

private:

  int i,j,h; //indexes
  TOK_CODE symbol;
  
};

class Edge{

 public:
  friend class EdgeHasher;
  friend class EdgeEquality;
  friend class Forest;

  Edge(Vertex *head,Vertex *tail,float weight);
  Edge(Vertex *head,Vertex *tail_lhs,Vertex *tail_rhs,float weight);

  size_t arity()const{return tail.size();}
  double get_weight()const{return weight;} //returns the local weight of this edge

  friend ostream& operator<<(ostream &os, Edge const &v);

 private:
  Vertex* head;
  vector<Vertex*> tail;  
  double weight;

};


//Forest related auxiliary functions
//Ptr Hashing
class VertexHasher{
 public:
  size_t operator()(Vertex const *v)const; 
};

class EdgeHasher{
 public:
  size_t operator()(Edge const *e)const; 
};

//Ptr equalities
class VertexEquality{
 public:
  bool operator()(Vertex const *v1,Vertex const *v2)const; 
};

class EdgeEquality{ //edge weight is ignored for equality 
 public:
  bool operator()(Edge const *e1,Edge const *e2)const;
};



/**
 * This is a helper class for the forest, for building weighted derivations of backpointers.
 * This basically wraps vertices as weighted trees of pointers.
 * BackDerivations can easily be wrapped into lists.
 */
class BackDerivation{
    
public:
    
    BackDerivation();//dummy constructor
    BackDerivation(Vertex *head);//leaf derivation
    BackDerivation(Vertex *head,BackDerivation *child,double local_weight);
    BackDerivation(Vertex *head,BackDerivation *left_child,BackDerivation *right_child,double local_weight);
    
    double get_inside_weight()const;
    size_t arity()const;
    
    AbstractParseTree* as_tree(vector<InputToken> const &input_sequence, bool root = false)const;//allocates recursively a new tree from this parse derivation
    
private:
    double inside_weight;
    Vertex *head;
    vector<BackDerivation*> tail;
};




//The forest also plays the role of container for vertices and edges (in charge of destroying them)
class Forest{

public:
    
  Forest(){N=0;};
  Forest(size_t n);//Builds a forest for a sentence of N words.
  Forest(AbstractParseTree const *root);//Builds a forest from a single tree.

  ~Forest();

 //adds an edge of the form <X,i,j> -> <Y,i,k> <Z,k,j> :: w  with i <= h <= j the index of the head on input automaton  
  bool add_edge(TOK_CODE X,unsigned int i,unsigned int j,TOK_CODE Y ,TOK_CODE Z, unsigned int k, float w,int hT = -1,int hL = -1,int hR = -1);
  //adds an edge of the form <X,i,j> -> <Y,i,j> with i <= h <= j the index of the head on input automaton  
  bool add_edge(TOK_CODE X,unsigned int i,unsigned int j,TOK_CODE Y,float w,int h = -1);
  
  bool add_tree(AbstractParseTree const *root); //this adds a tree to this forest if its yield is compatible (same length)
  //bool add_derivation(ParseDerivation const &derivation); //this adds a derivation to this forest if its yield is compatible (same length)
  
  //bool make_best_derivation(ParseDerivation const &derivation); //this adds a derivation to this forest if its yield is compatible (same length)
    
    
  //Tree extraction
  //TODO : not implemented yet!
  AbstractParseTree* kbest(int k, vector<InputToken> const &input_sequence);//yields the kth-best parse out of a weighted forest.
    
    
  ostream& as_automaton(ostream& os)const;     //outputs a tiburon tree automaton
  ostream& as_grammar(ostream& os)const;
  friend ostream& operator<< (ostream &os,Forest const &forest);
    
    
private:

  typedef unordered_set<Vertex*,VertexHasher,VertexEquality> VX_SET ;
  typedef unordered_set<Edge*,EdgeHasher,EdgeEquality>  EDGE_SET ;
  typedef unordered_set<Vertex*,VertexHasher,VertexEquality>::const_iterator vertex_const_iterator;
  typedef unordered_set<Vertex*,VertexHasher,VertexEquality>::iterator vertex_iterator;
  typedef unordered_set<Edge*,EdgeHasher,EdgeEquality>::iterator edge_iterator;
  typedef unordered_set<Edge*,EdgeHasher,EdgeEquality>::const_iterator edge_const_iterator;
  typedef unordered_map<Vertex*, unordered_set<Edge*,EdgeHasher,EdgeEquality> >::iterator star_iterator;
  typedef unordered_map<Vertex*, unordered_set<Edge*,EdgeHasher,EdgeEquality> >::const_iterator star_const_iterator;

    
  pair<Vertex*,bool> add_node(Vertex *v); //if the node is not added, it is immediately destroyed (!)
  bool add_edge(Edge *e);                 //if the edge is not added, it is immediately destroyed (!)
  tuple<int,int,int> add_treeR(AbstractParseTree const *root,int startidx=0);
  
    
  Vertex *root;
  size_t N; //the number of states in the source automaton (n+1 words)

    
  //Nodes properties
  bool is_leaf(Vertex *v)const;
  bool is_root(Vertex *v)const{return *v == *root;}
  
  //Navigation
  void leaves(vector<Vertex*> &input_sequence)const;//fills input sequence w/ leaves
  void idom_vertices(Vertex *v,VX_SET &idom);

    
  //inner members
  VX_SET    nodes;
  EDGE_SET  edges;
  unordered_map<Vertex*, unordered_set<Edge*,EdgeHasher,EdgeEquality> > bs_index; //lists the edges in the Backward Star of a node
  unordered_map<Vertex*, unordered_set<Edge*,EdgeHasher,EdgeEquality> > fs_index; //lists the edges in the Forward Star of a node
  unordered_map<Vertex*, vector<BackDerivation>,VertexHasher,VertexEquality> viterbi_index; //stores the best derivation for node v

};

#endif
