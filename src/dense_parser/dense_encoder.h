#ifndef DENSE_ENCODER_H
#define DENSE_ENCODER_H

#include "srgrammar.h"
#include "sparse.h"
#include "sparse_encoder.h"
#include <assert.h>



//class DenseFeatureTemplate : public FeatureTemplate {
//    unsigned int get_index_in_lookup(StateSignature const &signature){
//        assert(sensors.size() == 1);
//        return sensors[0]->get_value(signature);
//    }
//};  // difficile à maintenir

/**
 * Encodes a configuration on an int vector corresponding to rows in a look up table.
 * Look-up table is a vector of (various length) Eigen::RowVectorXd
 */
class DenseEncoderGlobal{

public:
    DenseEncoderGlobal();
    DenseEncoderGlobal(string const &filename,TemplateTypeDefinition const &ttd);
    DenseEncoderGlobal(const char *filename,TemplateTypeDefinition const &ttd);
    DenseEncoderGlobal(const DenseEncoderGlobal &other);
    DenseEncoderGlobal& operator=(const DenseEncoderGlobal &other);

    void add_template(FeatureTemplate const &ntemplate);

    void encode(vector<int> &encoding, StateSignature const &signature, bool clear=true);

    size_t ntemplates()const;

    //saves templates to file
    //void save(string const &filename)const;


    /** Returns the column identifier acessed by the templates.
     * This method is used to map each template towards the dimensionality of the corresponding embedding
     */
    int get_colidx(int tpl_idx);

    bool has_morpho()const{
        return morpho;
    }//says if morph templates are defined

private:
    vector<FeatureTemplate> templates;
    unsigned int T;
    bool morpho = false;
};


typedef DenseEncoderGlobal DenseEncoder;

///////////////////////////////////////////////////////////////////////
#endif // DENSE_ENCODER_H
