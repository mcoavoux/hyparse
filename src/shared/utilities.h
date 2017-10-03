#ifndef UTILITIES_H
#define UTILITIES_H

#include "globals.h"
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

void copy_file(string const &filein,string const &fileout);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility class for the grammar and other classes

/**
 * Action Bimap
 * An integer bimap encodes a bijection between labels and their contiguous indexes in vector or matrices
 * and allows to retrieve index from label and action from index in constant time.
 * The mapping is <YLABEL,YINDEX>
 * The label type must implement the unsigned int get_code() function or be an unsigned int
 */

template<class YLABEL>
class Bimap{
    
public:
    
    typedef unsigned int YINDEX;
    
    Bimap(){};
    Bimap(Bimap<YLABEL> const &other);
    Bimap& operator=(Bimap<YLABEL> const &other);
    
    Bimap(vector<YLABEL> const &labels); //from set of labels
    //(codes are not supposed to be contiguous, although it is much better)
    
    inline YINDEX get_index(YLABEL label)const{return indexes.at(label.get_code() - min_label_code);}
    inline const YLABEL& get_label(YINDEX index)const{return labels.at(index);}
    inline size_t nlabels()const{return labels.size();};
    
    void make_map(vector<YLABEL> const &labels);//does the actual encoding of the map
    void clear();
private:
    
    typedef unsigned int YCODE; //a numeric code for the YLABEL type
    vector<YLABEL> labels;
    vector<YINDEX> indexes;
    YCODE min_label_code=0;
    
};

template <class YLABEL>
class ConfusionMatrix{
    
public:
    
    ConfusionMatrix(vector<YLABEL> const &label_set);
    float& operator()(YLABEL const &yref,YLABEL const &ypred);
    string to_string()const;
    
private:
    Bimap<YLABEL> dictionary;
    vector<vector<float>> cmatrix;
    
};




/**
 *  Counter : a class for counting item frequencies from a list
 */
template<class Item>
class Counter{
public: 
	
  /**
   * adds a count to the counter
   */
  void add_count(Item const &item);
  void add_all_counts(vector<Item>const &observations);

  /**
   * Fills in the result list with items of counts >= threshold
   */	
  void aboveEq_threshold(unsigned int freq_threshold,vector<Item> &result)const;

  /**
   * fills in the list w/ all entries
   */
  void all_entries(vector<Item> &entries)const;
  /**
   * Num items in this counter
   */
  size_t size(){return count_map.size();}

private:
  unordered_map<Item,unsigned int> count_map;

};

#include "utilities.hpp"

#endif
