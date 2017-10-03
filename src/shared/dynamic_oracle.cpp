#include "dynamic_oracle.h"
#include "globals.h"


ConstituentSet::ConstituentSet(const ParseDerivation & derivation, SrGrammar const &grammar){
    update(derivation, grammar);
}

ConstituentSet::ConstituentSet(AbstractParseTree const *root, SrGrammar const &grammar,InputDag &input_sequence,bool tagger_deriv){
    ParseDerivation derivation(root, grammar, input_sequence, tagger_deriv);
    update(derivation, grammar);
}

TOK_CODE ConstituentSet::find_constituent(int i, int j){
    auto iti = i2j.find(i);
    if (iti == i2j.end()) return 0;
    auto itj = i2j[i].find(j);
    if (itj == i2j[i].end()) return 0;
    return itj->second;
    //return i2j[i][j];
}

bool ConstituentSet::find_constituent_kn(int k, int j){
    auto it1 = i2j.find(k);
    if (it1 != i2j.end()){
        for (auto it2 = it1->second.begin(); it2 != it1->second.end(); it2++){
            if (it2->first > j){
                return true;
            }
        }
    }
    return false;
}

TOK_CODE ConstituentSet::find_constituent_mj(int j, int i){
    TOK_CODE res = 0;
    int idx = -10;
    auto it1 = j2i.find(j);
    if (it1 != j2i.end()){
        for (auto it2 = it1->second.begin(); it2 != it1->second.end(); it2++){
            if (it2->first < i && it2->first > idx){
                res = it2->second;
                idx = it2->first;
            }
        }
    }
    return res;
}

int ConstituentSet::get_head_position(int i, int j) const{
    auto got1 = head_position.find(i);
    assert(got1 != head_position.end());
    auto got2 = got1->second.find(j);
    assert(got2 != got1->second.end());
    return got2->second;
}


void ConstituentSet::get_constituent_list(vector<tuple<int,int,TOK_CODE>> &list){
    list.clear();
    for (auto iti = i2j.begin(); iti != i2j.end(); iti ++){
        int i = iti->first;
        for (auto itj = iti->second.begin(); itj != iti->second.end(); itj++){
            int j = itj->first;
            list.push_back(std::make_tuple(i,j,itj->second));
        }
    }
}

void ConstituentSet::update(const ParseDerivation & derivation, SrGrammar const &grammar){
    for (int i = 1; i < derivation.size(); i++){
        pair<int,int> span = derivation[i]->get_span();
        TOK_CODE ps = derivation[i]->get_top_symbol();
        if (span.first + 1 == span.second){
            if (derivation[i]->get_incoming_action().action_type == ParseAction::RU){
                i2j[span.first][span.second] = ps;
                j2i[span.second][span.first] = ps;
            }else{
                assert(derivation[i]->get_incoming_action().action_type == ParseAction::SHIFT || derivation[i]->get_incoming_action().action_type == ParseAction::GR);
            }
        }else{
            i2j[span.first][span.second] = ps;
            j2i[span.second][span.first] = ps;
            if (derivation[i]->get_incoming_action() == grammar.get_action(ParseAction::RL, ps)) head_position[span.first][span.second] = LEFT;
            else if (derivation[i]->get_incoming_action() == grammar.get_action(ParseAction::RR, ps)) head_position[span.first][span.second] = RIGHT;
            else{ cerr << "Error in dynamic oracle, aborting" << endl; exit(1);}
        }
    }
}


DynamicOracle::DynamicOracle(SrGrammar grammar){

    this->grammar = grammar;
    grammar.get_temporaries_codes(temporaries);
    grammar.get_nontemporaries_codes(nontemporaries);

    IntegerEncoder *enc = IntegerEncoder::get();
    for (TOK_CODE i : temporaries){
        PSTRING tps = enc->decode(i);
        PSTRING ps;
        if (tps == L"SENT:"){
            ps = L"ROOT@SENT";
        }else if (tps == L"PSEUDO:"){
            ps = L"ROOT@PSEUDO";
        }else{
            ps = PSTRING(tps.begin(), tps.end() - 1);
        }
        TOK_CODE tps_code = enc->get_code(tps, IntegerEncoder::PS_COLCODE);
        TOK_CODE ps_code  = enc->get_code(ps, IntegerEncoder::PS_COLCODE);

        if (tps_code > tmp2nt.size()){  tmp2nt.resize(tps_code + 1, 0);}
        if (ps_code >  nt2tmp.size()){  nt2tmp.resize(ps_code + 1, 0); }

        nt2tmp[ps_code] = tps_code;
        tmp2nt[tps_code] = ps_code;
    }
    for (TOK_CODE i : nontemporaries){
        PSTRING ps = enc->decode(i);
        PSTRING::size_type found = ps.find_last_of(L'@');

        if (found != PSTRING::npos){
            PSTRING tps = ps.substr(found + 1) + TMP_CODE;
            TOK_CODE tps_code = enc->get_code(tps, IntegerEncoder::PS_COLCODE);
            TOK_CODE ps_code  = enc->get_code(ps, IntegerEncoder::PS_COLCODE);

            if (ps_code >  nt2tmp.size()){  nt2tmp.resize(ps_code + 1, 0); }
            assert(nt2tmp[ps_code] == 0 || enc->decode(ps_code) == L"ROOT@SENT");
            nt2tmp[ps_code] = tps_code;
        }
    }

    // no it won't work
//    IntegerEncoder *enc = IntegerEncoder::get();
//    for (TOK_CODE i : nontemporaries){
//        PSTRING ps = enc->decode(i);
//        PSTRING tmp = ps + TMP_CODE;
//        TOK_CODE tmpcode = enc->get_code(tmp, IntegerEncoder::PS_COLCODE);
//        bool isGrammar = false;
//        for (TOK_CODE j : temporaries){
//            if (tmp_code == j)
//                isGrammar = true;
//        }
//        if (isGrammar){
//            nontmp2tmp[i] = tmpcode;
//        }
//    }
}

bool DynamicOracle::grammar_check(vector<bool> &actions, vector<bool> &allowed_actions){
    //assert(actions.size() == y_scores.size());
    bool res = false;
    int n = 0;
    for (int i = 0; i < actions.size(); i++){
        if (! allowed_actions[i])
            actions[i] = false;
        if (actions[i]){
            res = true;
            n++;
        }
    }
    if (n > 1){ n_non_determinism ++; }
#ifdef DEBUG_ORACLE
    cerr << "{";
    for (int i = 0; i < actions.size(); i++){
        if (actions[i]){
            cerr << grammar[i] << " ";
        }
    }
    if (n > 1){ cerr << "} number of optimal actions : " << n << endl;}
    else       cerr << "}" << endl;
#endif
    return res;
}

bool DynamicOracle::operator()(ParseState const *s0, ConstituentSet &gold, vector<bool> &actions, vector<bool> &allowed_actions){
    n_predictions ++;
#ifdef DEBUG_ORACLE
    cerr << "Configuration : ";
    print_stack(s0, cerr);
    cerr << endl << "Derivation : ";
    print_derivation(s0,cerr);
    cerr << endl;
#endif
    std::fill(actions.begin(), actions.end(), false);
    pair<int, int> s0span = s0->get_span();
    const int K = s0span.first, J = s0span.second;
    ParseState const *s1 = s0->stack_predecessor();

    if (s0->get_incoming_action() == grammar.get_shift_action()){                                   select_RU(s0, gold, actions, J);    return grammar_check(actions, allowed_actions);}   // GR or RU
    if (s0->is_init() || s1->is_init()){                                                            select_shift(s0, gold, actions, J); return grammar_check(actions, allowed_actions);}   // shift cases : stack has 0 or 1 symbol
    if (gold.find_constituent_kn(K, J)){                                                            select_shift(s0, gold, actions, J); return grammar_check(actions, allowed_actions);}   //               find (X,k,n)constituent
    if (grammar.is_temporary(s0->get_top_symbol()) && grammar.is_temporary(s1->get_top_symbol())){  select_shift(s0, gold, actions, J); return grammar_check(actions, allowed_actions);}   //               2 temporaries on top of stack

    pair<int,int> s1span = s1->get_span();
    const int I = s1span.first;
    assert(K == s1span.second);
    TOK_CODE ps = gold.find_constituent(I, J);
    if (ps != 0){
        select_reduce(s0, gold, actions, I, J, ps, allowed_actions);
        if (grammar_check(actions, allowed_actions)){
            return true;
        }
#ifdef DEBUG_ORACLE
        else{
            cerr << "Constituent (X,i,j) cannot be constructed" << endl;
        }
#endif
    }
    ps = gold.find_constituent_mj(J, I);
    if (ps != 0){
        select_reduce_sequence(s0, s1, actions, ps, allowed_actions);
        if (grammar_check(actions, allowed_actions)){
            return true;
        }
#ifdef DEBUG_ORACLE
    else{
        cerr << "Constituent (X,m,j) : search for action has failed" << endl;
    }
#endif
    }
    if (allowed_actions[grammar.get_action_index(grammar.get_shift_action())] && ! grammar.is_temporary(s0->get_top_symbol())){
        select_shift(s0, gold, actions, J);
        return grammar_check(actions, allowed_actions);
    }
    for (TOK_CODE nt : nontemporaries){
        actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, nt))] = true;
        actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, nt))] = true;
    }
    //std::fill(actions.begin(), actions.end(), true);
    return grammar_check(actions, allowed_actions);
}



int DynamicOracle::compute_loss(ConstituentSet &gold, ConstituentSet &pred){
    vector<tuple<int,int,TOK_CODE>> constituents;
    gold.get_constituent_list(constituents);
    int cpt = 0;
    for (auto it = constituents.begin(); it != constituents.end(); it++){
        int i = std::get<0>(*it);
        int j = std::get<1>(*it);
        TOK_CODE tok = std::get<2>(*it);
        TOK_CODE S = pred.find_constituent(i,j);
        if (! (tok == S || (grammar.is_temporary(tok) && grammar.is_temporary(S)))){
            cpt++;
        }
    }
    return cpt;
}

void DynamicOracle::print_stack(ParseState const *s0, ostream &os){
    if (! s0->is_init()){
        print_stack(s0->stack_predecessor(), os);
        pair<int,int> span = s0->get_span();
        TOK_CODE symbol = s0->get_top_symbol();
        os << "(" << IntegerEncoder::get()->decode8(symbol) << "," << span.first << "," << span.second << ")";
    }
}
void DynamicOracle::print_derivation(ParseState const *s0, ostream &os){
    if (! s0->is_init()){
        print_derivation(s0->history_predecessor(), os);
        os << " " << *s0;
    }
}


//bool DynamicOracle::select_reduce_sequence(ParseState const *s0, ParseState const *s1, vector<bool> &actions, unordered_map<int, TOK_CODE> &mj_constituents){
//    ParseState const *s2 = s1->stack_predecessor();
//    while (! s2->is_init() && mj_constituents.find(s2->get_span().first) == mj_constituents.end()){  // is there a reachable constituent in mj_constituents (= is there a constituent (X,m,_) in stack ?)
//        s2 = s2->stack_predecessor();
//    }
//    if (! s2->is_init()){
//        assert(mj_constituents.find(s2->get_span().first) != mj_constituents.end());
//        s2 = s1->stack_predecessor();
//        bool s0tmp = grammar.is_temporary(s0->get_top_symbol());
//        bool s1tmp = grammar.is_temporary(s1->get_top_symbol());
//        bool s2tmp = grammar.is_temporary(s2->get_top_symbol());
//        assert(!(s0tmp && s1tmp));
//        assert(! s2->is_init());
//        if (s2tmp){ // reduce to non temporary symbol
//            for (auto &i : nontemporaries){
//                if (! s0tmp){
//                    actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, i))] = true;
//                }
//                if (! s1tmp){
//                    actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, i))] = true;
//                }
//            }
//        }else{
//            for (auto &i : temporaries){
//                if (! s0tmp){
//                    actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, i))] = true;
//                }
//                if (! s1tmp){
//                    actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, i))] = true;
//                }
//            }
//        }
//#ifdef DEBUG_ORACLE
//        cerr << "Oracling a reduction sequence : success" << endl;
//#endif
//        return true;
//    }
//#ifdef DEBUG_ORACLE
//        cerr << "Oracling a reduction sequence : failure" << endl;
//#endif
//    return false;
//}

void DynamicOracle::select_reduce_sequence(ParseState const *s0, ParseState const *s1, vector<bool> &actions, TOK_CODE ps, vector<bool> &allowed_actions){
#ifdef DEBUG_ORACLE
        cerr << "Searching to predict (X,m,j) = " << IntegerEncoder::get()->decode8(ps) << endl;
#endif
    TOK_CODE tmp;
    TOK_CODE nontmp;

    if (grammar.is_temporary(ps)){
        tmp = ps;
        assert(tmp2nt.size() > ps);
        nontmp = tmp2nt[ps];
        assert(nontmp != 0);
    }else{
        assert(nt2tmp.size() > ps);
        tmp = nt2tmp[ps];
        nontmp = ps;
        //cerr << "ps = " << IntegerEncoder::get()->decode8(ps) << endl;
        assert(tmp != 0);

    }
    assert(! s0->stack_predecessor()->is_init());
    assert(! s0->stack_predecessor()->stack_predecessor()->is_init());
    if (grammar.is_temporary(s0->stack_predecessor()->stack_predecessor()->get_top_symbol())){
        if (! grammar.is_temporary(s0->get_top_symbol())){
            int action_code = grammar.get_action_index(grammar.get_action(ParseAction::RL, nontmp));
            if (allowed_actions[action_code]){
                actions[action_code] = true;
            }else{
                for (TOK_CODE t : nontemporaries){
                    actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, t))] = true;
                }
            }
        }
        if (! grammar.is_temporary(s0->stack_predecessor()->get_top_symbol())){
            ParseAction paa = grammar.get_action(ParseAction::RR, nontmp);
            int action_code = grammar.get_action_index(grammar.get_action(ParseAction::RR, nontmp));
            if (allowed_actions[action_code]){
                actions[action_code] = true;
            }else{
                for (TOK_CODE t : nontemporaries){
                    actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, t))] = true;
                }
            }
        }

    }else{
        if (! grammar.is_temporary(s0->get_top_symbol())){
            actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, tmp))] = true;
        }
        if (! grammar.is_temporary(s0->stack_predecessor()->get_top_symbol())){
            actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, tmp))] = true;
        }
    }

//    vector<TOK_CODE> stack;
//    stack.push_back(s0->get_top_symbol());
//    stack.push_back(s1->get_top_symbol());
//    ParseState const *s2 = s1->stack_predecessor();
//    while (! s2->is_init() && mj_constituents.find(s2->get_span().first) == mj_constituents.end()){  // is there a reachable constituent in mj_constituents (= is there a constituent (X,m,_) in stack ?)
//        stack.push_back(s2->get_top_symbol());
//        s2 = s2->stack_predecessor();
//    }
//    if (! s2->is_init()){
//        assert(mj_constituents.find(s2->get_span().first) != mj_constituents.end());
//        TOK_CODE Xmj = mj_constituents.find(s2->get_span().first)->second;
//#ifdef DEBUG_ORACLE
//        cerr << "Searching to predict (X,m,j) = " << IntegerEncoder::get()->decode8(Xmj) << endl;
//#endif
//        s2 = s1->stack_predecessor();
//        bool s0tmp = grammar.is_temporary(s0->get_top_symbol());
//        bool s1tmp = grammar.is_temporary(s1->get_top_symbol());
//        bool s2tmp = grammar.is_temporary(s2->get_top_symbol());
//        assert(!(s0tmp && s1tmp));
//        assert(! s2->is_init());
//        if (s2tmp){ // reduce to non temporary symbol
//            for (auto &i : nontemporaries){
//                if (! s0tmp){
//                    actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, i))] = true;
//                }
//                if (! s1tmp){
//                    actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, i))] = true;
//                }
//            }
//        }else{
//            if (s0tmp){  actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, s0->get_top_symbol()))] = true; }
//            if (s1tmp){  actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, s1->get_top_symbol()))] = true; }
//            if (!s0tmp && !s1tmp){
//                if (grammar.is_temporary(Xmj)){
//                    actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, Xmj))] = true;
//                    actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, Xmj))] = true;
//                }else{
//                    //actions[grammar.get_action(ParseAction::RL, Xmj)] = true; // ideally : get nontemporary symbol corresponding to Xmj  (?? merge unaries problems, is this symbol unique ?)
//                    //actions[grammar.get_action(ParseAction::RR, Xmj)] = true;
//                    for (auto &i : temporaries){
//                        actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, i))] = true;
//                        actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, i))] = true;
//                    }
//                }
//            }
//        }
//#ifdef DEBUG_ORACLE
//        cerr << "Oracling a reduction sequence : success" << endl;
//#endif
//        return true;
//    }
//#ifdef DEBUG_ORACLE
//        cerr << "Oracling a reduction sequence : failure" << endl;
//#endif
//    return false;
}


void DynamicOracle::select_reduce(ParseState const *s0, ConstituentSet &gold, vector<bool> &actions, int I, int J, TOK_CODE ps, vector<bool> &allowed_actions){
#ifdef DEBUG_ORACLE
        cerr << "Oracling a reduction (i,j)" << endl;
#endif
    assert(! (grammar.is_temporary(s0->get_top_symbol()) && grammar.is_temporary(s0->stack_predecessor()->get_top_symbol())));
    if (! grammar.is_temporary(ps)  || s0->stack_predecessor()->stack_predecessor()->is_init()){
        if (grammar.is_temporary(s0->get_top_symbol())){
            actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, ps))] = true;
            return;
        }
        if (grammar.is_temporary(s0->stack_predecessor()->get_top_symbol())){
            actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, ps))] = true;
            return;
        }
        int head_position = gold.get_head_position(I, J);
        switch (head_position){
            case ConstituentSet::LEFT :actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, ps))] = true; break;
            case ConstituentSet::RIGHT:actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, ps))] = true; break;
            default : cerr << "Error with head position in dynamic oracle, aborting" << endl; exit(1);
        }
    }else{
        TOK_CODE nontmp = tmp2nt[ps];
        if (grammar.is_temporary(s0->stack_predecessor()->stack_predecessor()->get_top_symbol()) && gold.find_constituent_mj(J, s0->stack_predecessor()->stack_predecessor()->get_span().first +1)){
            if (grammar.is_temporary(s0->get_top_symbol())){
                //actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, nontmp))] = true;
                for (int i = 0; i < nontemporaries.size(); i++){
                    actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, nontemporaries[i]))] = true;
                }
                return;
            }
            if (grammar.is_temporary(s0->stack_predecessor()->get_top_symbol())){
                //actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, nontmp))] = true;
                for (int i = 0; i < nontemporaries.size(); i++){
                    actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, nontemporaries[i]))] = true;
                }
                return;
            }
            int head_position = gold.get_head_position(I, J);
            switch (head_position){
//                case ConstituentSet::LEFT :actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, nontmp))] = true; break;
//                case ConstituentSet::RIGHT:actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, nontmp))] = true; break;
                case ConstituentSet::LEFT :for (int i = 0; i < nontemporaries.size(); i++){actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, nontemporaries[i]))] = true; } break;
                case ConstituentSet::RIGHT:for (int i = 0; i < nontemporaries.size(); i++){actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, nontemporaries[i]))] = true; } break;
                default : cerr << "Error with head position in dynamic oracle, aborting" << endl; exit(1);
            }
            return;
        }

        if (allowed_actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, ps))] || allowed_actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, ps))]){
            actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, ps))] = true;
            actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, ps))] = true;
            return;
        }else{
            for (int i = 0; i < nontemporaries.size(); i++){actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, nontemporaries[i]))] = true; }
            for (int i = 0; i < nontemporaries.size(); i++){actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, nontemporaries[i]))] = true; }
//            actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, nontmp))] = true;
//            actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, nontmp))] = true;
            return;
        }
    }
}

//void DynamicOracle::select_reduce(ParseState const *s0, ConstituentSet &gold, vector<bool> &actions, int I, int J, TOK_CODE ps){
//    assert(! (grammar.is_temporary(s0->get_top_symbol()) && grammar.is_temporary(s0->stack_predecessor()->get_top_symbol())));
//    if (grammar.is_temporary(s0->get_top_symbol())){
//        actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, ps))] = true;
//        return;
//    }
//    if (grammar.is_temporary(s0->stack_predecessor()->get_top_symbol())){
//        actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, ps))] = true;
//        return;
//    }
//    int head_position = gold.get_head_position(I, J);
//    switch (head_position){
//        case ConstituentSet::LEFT :actions[grammar.get_action_index(grammar.get_action(ParseAction::RL, ps))] = true; break;
//        case ConstituentSet::RIGHT:actions[grammar.get_action_index(grammar.get_action(ParseAction::RR, ps))] = true; break;
//        default : cerr << "Error with head position in dynamic oracle, aborting" << endl; exit(1);
//    }
//}


//void DynamicOracle::select_shift(ParseState const *s0, ConstituentSet &gold, vector<bool> &actions, int j){
//    TOK_CODE ps = gold.find_constituent(j, j+1);
//    if (ps != 0){
//        actions[grammar.get_action_index(grammar.get_action(ParseAction::RU, ps))] = true;
//    }else{
//        actions[grammar.get_action_index(grammar.get_ghost_action())] = true;
//    }
//    actions[grammar.get_action_index(grammar.get_shift_action())] = true;
//}

void DynamicOracle::select_shift(ParseState const *s0, ConstituentSet &gold, vector<bool> &actions, int j){
#ifdef DEBUG_ORACLE
        cerr << "Oracling a shift" << endl;
#endif
    actions[grammar.get_action_index(grammar.get_shift_action())] = true;
}

void DynamicOracle::select_RU(ParseState const *s0, ConstituentSet &gold, vector<bool> &actions, int j){
#ifdef DEBUG_ORACLE
        cerr << "Oracling a unary reduction / ghost reduction" << endl;
#endif
    TOK_CODE ps = gold.find_constituent(j, j+1);
    if (ps != 0){   actions[grammar.get_action_index(grammar.get_action(ParseAction::RU, ps))] = true; }
    else{           actions[grammar.get_action_index(grammar.get_ghost_action())] = true; }
}

//void DynamicOracle::select_shift_old(ParseState const *s0, ConstituentSet &gold, vector<bool> &actions, int j){
//    if (s0->get_incoming_action() == grammar.get_shift_action()){               // if last action is shift : choose between RU and GR
//        TOK_CODE ps = gold.find_constituent(j, j+1);
//        if (ps == 0){                                                           // if there is no unary constituent : choose GR
//            actions[grammar.get_action_index(grammar.get_ghost_action())] = true;
//        }else{                                                                  // else : RU
//            actions[grammar.get_action_index(grammar.get_action(ParseAction::RU, ps))] = true;
//        }
//    }else{          // last action is not shift : choose shift
//        actions[grammar.get_action_index(grammar.get_shift_action())] = true;
//    }
//}








ostream& operator<<(ostream &os, ConstituentSet const &set){

    for (auto& it1 : set.i2j){
        for (auto& it2 : it1.second){
            os << "(" << IntegerEncoder::get()->decode8(it2.second);
            int i = it1.first;
            int j = it2.first;
            if (i +1 !=j && set.get_head_position(i,j) == ConstituentSet::LEFT)
                os << "-L,";
            else
                os << "-R,";
            os << i << "," << j << ")";
        }
    }
    return os;
}
