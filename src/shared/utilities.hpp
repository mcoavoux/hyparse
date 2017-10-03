template<class Item>
void Counter<Item>::aboveEq_threshold(unsigned int freq_threshold,vector<Item> &result)const{
  result.clear();
  for(typename unordered_map<Item,unsigned int>::const_iterator it = count_map.begin(); it != count_map.end();++it){
    if(it->second >= freq_threshold){
      result.push_back(it->first);
    }
  }
}

template<class Item>
void Counter<Item>::add_count(Item const &item){
  typename unordered_map<Item,unsigned int>::iterator got;
  got = count_map.find(item);
  if (got != count_map.end()){
    got->second += 1;
  }else{
    count_map[item]= 1;
  }
}

template<class Item>
void Counter<Item>::add_all_counts(vector<Item> const &observations){
  for(typename vector<Item>::const_iterator it = observations.begin();it != observations.end();++it){
    add_count(*it); 
  }
}

template<class Item>
void Counter<Item>::all_entries(vector<Item> &entries)const{
  entries.clear();
  for(typename unordered_map<Item,unsigned int>::const_iterator it = count_map.begin(); it != count_map.end();++it){
    entries.push_back(it->first);
  }
}


template<class YLABEL>
Bimap<YLABEL>::Bimap(vector<YLABEL> const &labels){ //from list of actions
    make_map(labels);
}

template<class YLABEL>
void Bimap<YLABEL>::clear(){
    labels.clear();
    indexes.clear();
    min_label_code=0;
}

template<class YLABEL>
Bimap<YLABEL>::Bimap(Bimap<YLABEL> const &other){
    labels  = other.labels;
    indexes = other.indexes;
    min_label_code = other.min_label_code;
}

template<class YLABEL>
Bimap<YLABEL>& Bimap<YLABEL>::operator=(Bimap<YLABEL> const &other){
    clear();
    labels  = other.labels;
    indexes = other.indexes;
    min_label_code = other.min_label_code;
    return *this;
}

template<class YLABEL>
void Bimap<YLABEL>::make_map(vector<YLABEL> const &labels){
    
    this->labels = labels;
    YCODE max_label_code = labels[0].get_code();
    min_label_code = labels[0].get_code();
    for(int i = 1; i < labels.size();++i){
        if (labels[i].get_code() > max_label_code){max_label_code = labels[i].get_code();}
        if (labels[i].get_code() < min_label_code){min_label_code = labels[i].get_code();}
    }
    indexes.clear();
    indexes.resize(max_label_code-min_label_code+1,0);
    for(int i = 0; i < labels.size();++i){
        indexes[labels[i].get_code()-min_label_code] = i;
    }
}

//Specialization for unsigned int
template<>
inline Bimap<TOK_CODE>::YINDEX Bimap<TOK_CODE>::get_index(TOK_CODE label)const{
    return indexes.at(label - min_label_code);
}

template<>
inline void Bimap<TOK_CODE>::make_map(vector<TOK_CODE> const &labels){
    this->labels = labels;
    YCODE max_label_code = labels[0];
    min_label_code = labels[0];
    for(int i = 1; i < labels.size();++i){
        if (labels[i] > max_label_code){max_label_code = labels[i];}
        if (labels[i] < min_label_code){min_label_code = labels[i];}
    }
    indexes.clear();
    indexes.resize(max_label_code-min_label_code+1,0);
    for(int i = 0; i < labels.size();++i){
        indexes[labels[i]-min_label_code] = i;
    }
}

//==========================================================//

template <class YLABEL>
ConfusionMatrix<YLABEL>::ConfusionMatrix(vector<YLABEL> const &label_set){
    dictionary.make_map(label_set);
    //init matrix
    for(int i = 0; i < dictionary.nlabels();++i){
        cmatrix.push_back(vector<float>(dictionary.nlabels(),0.0));
    }
}


template <class YLABEL>
float& ConfusionMatrix<YLABEL>::operator()(YLABEL const &yref,YLABEL const &ypred){
    return cmatrix[dictionary.get_index(yref)][dictionary.get_index(ypred)];
}

template <class YLABEL>
string ConfusionMatrix<YLABEL>::to_string()const{
    
    string res = "        ";
    //header
    for(int i = 0; i  < cmatrix.size();++i){
        res += dictionary.get_label(i)+"     ";
    }
    res += "\n";
    for(int i  = 0; i < cmatrix.size();++i){
        res+= dictionary.get_label(i)+"   ";
        for(int j = 0; j < cmatrix.size();++j){
            res+= to_string(cmatrix[i][j])+"      ";
        }
        res += "\n";
    }
    return res;
}




