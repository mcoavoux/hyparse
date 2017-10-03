#ifndef DYNAMIC_ORACLE_H
#define DYNAMIC_ORACLE_H

#include <unordered_map>
#include <assert.h>
#include <math.h>
#include "treebank.h"
#include "state_graph.h"

using std::unordered_map;
using std::tuple;

#define DBG(x) std::cerr << x << std::endl;
//#define DEBUG_ORACLE

/**
 * Class for a tree-consistent constituent set.
 * Every binary constituents are stored. The tags are not stored (but the other unaries are).
 * This class also stores head position for binary constituents.
 */
class ConstituentSet{
public:
    enum {UNDEFINED, LEFT, RIGHT};

    /** Builds a constituent set from a derivation. */
    ConstituentSet(const ParseDerivation & derivation, SrGrammar const &grammar);

    /** Builds a constituent set from a parse tree. */
    ConstituentSet(AbstractParseTree const *root, SrGrammar const &grammar,InputDag &input_sequence,bool tagger_deriv);

    /** Test if there exists a constituent with span (i,j). Returns non-terminal symbol if it is the case, and 0 otherwise */
    TOK_CODE find_constituent(int i, int j);

    /** Returns true iff there exists a constituent (X,k,n) in gold tree, assuming the stack has the form : (s1,i,k)|(s0,k,j) and n > j */
    bool find_constituent_kn(int k, int j);

    /**
     * Returns true iff there exists a constituent (X,m,j) in gold tree, assuming the stack has the form : (s1,i,k)|(s0,k,j) and m < i.
     * Updates c to store constituents satisfying that condition.
     */
    TOK_CODE find_constituent_mj(int j, int i);

    /** Assuming, (X,i,j) is in gold tree, returns position of the head in that constituent : LEFT or RIGHT. */
    int get_head_position(int i, int j) const;

    /** Constructs a vector of (X,i,j) constituents. */
    void get_constituent_list(vector<tuple<int,int,TOK_CODE>> &list);

    friend ostream& operator<<(ostream &os, ConstituentSet const &set);

private:

    /** Builds a constituent set from a derivation. */
    void update(const ParseDerivation & derivation, SrGrammar const &grammar);

    unordered_map<int, unordered_map<int,TOK_CODE> > i2j;           // i2j[i][j] : NP --> (NP, i, j) is in the tree
    unordered_map<int, unordered_map<int,TOK_CODE> > j2i;           // j2i[j][i] : NP --> (NP, i, j) is in the tree
    unordered_map<int, unordered_map<int, int> > head_position;     // gives the head position of a binary constituent
};

/**
 * Dynamic Oracle, essentially defines an oracle function.
 * Currently, tagging is not supported.
 * Assumption : there is only one temporary symbol in the grammar.
 */
class DynamicOracle{
public:
    DynamicOracle(SrGrammar grammar);

    /** Checks that the oracle's prediction are allowed actions.
      * Update n_prediction and n_non_determinism count.
      * Returns true iff the oracle predicted at least one action.
      **/
    bool grammar_check(vector<bool> &actions, vector<bool> &allowed_actions);

    /** Predict the set of cost-0 actions by updating actions vector.
     *  TODO : use a vector of bool instead of a vector of float (use select_actions(vector<bool>&) in the grammar module)
     */
    bool operator()(ParseState const *s0, ConstituentSet &gold, vector<bool> &actions, vector<bool> &allowed_actions);

    /** Computes loss for a ConstituentSet with respect to a gold set. */
    int compute_loss(ConstituentSet &gold, ConstituentSet &pred);

    /** Prints the whole stack. */
    void print_stack(ParseState const *s0, ostream &os);

    /** Prints the whole derivation. */
    void print_derivation(ParseState const *s0, ostream &os);

private:

    // Auxiliary functions
    void select_reduce_sequence(ParseState const *s0, ParseState const *s1, vector<bool> &actions, TOK_CODE ps, vector<bool> &allowed_actions);
    void select_reduce(ParseState const *s0, ConstituentSet &gold, vector<bool> &actions, int I, int J, TOK_CODE ps, vector<bool> &allowed_actions);
    void select_shift(ParseState const *s0, ConstituentSet &gold, vector<bool> &actions, int j);
    void select_RU(ParseState const *s0, ConstituentSet &gold, vector<bool> &actions, int j);
    //void select_shift_old(ParseState const *s0, ConstituentSet &gold, vector<bool> &actions, int j);

    SrGrammar grammar;
    vector<TOK_CODE> temporaries;
    vector<TOK_CODE> nontemporaries;
    vector<TOK_CODE> nt2tmp;
    vector<TOK_CODE> tmp2nt;

public:
    int n_predictions = 0;
    int n_non_determinism=0;

    int n_followed_predictions = 0;
};




#endif // DYNAMIC_ORACLE_H
