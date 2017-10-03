//
//  utilities.cpp
//  hyparse2
//
//  Created by bcrabbe on 09/11/2014.
//  Copyright (c) 2014 INRIA. All rights reserved.
//

#include "utilities.h"
#include <fstream>
#include <string>






void copy_file(string const &filein,string const &fileout){
    wifstream inFile(filein);
    wofstream outFile(fileout);
    wstring bfr;
    while(getline(inFile,bfr)){
        outFile << bfr << endl;
    }
    inFile.close();
    outFile.close();
}
