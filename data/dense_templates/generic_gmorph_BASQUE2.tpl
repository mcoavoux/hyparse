#Somewhat standard template set (expressed in ttd notation)
#works with the _ttd treebanks

#Unigrams
P(1)  = s0(t,h,tag)           
P(2)  = s0(t,h,word)             
P(3)  = s0(t,c,word)             
P(4)  = s1(t,h,tag)              
P(5)  = s1(t,h,word)           
P(6)  = s1(t,c,word)          
P(7)  = s2(t,h,tag)           
P(8)  = s2(t,h,word)            
P(9)  = s2(t,c,word)

P(10) = q0(word)
P(11) = q1(word)
P(12) = q2(word)
P(13) = q3(word)
P(14) = q0(tag)
P(15) = q1(tag)
P(16) = q2(tag)
P(17) = q3(tag)

P(18) = s0(l,h,tag)          
P(19) = s0(r,h,tag)          
P(20) = s1(l,h,tag)        
P(21) = s1(r,h,tag)            
P(22) = s0(l,c,word)          
P(23) = s0(r,c,word)          
P(24) = s1(l,c,word)        
P(25) = s1(r,c,word)            

P(26) = s0(l,h,word)          
P(27) = s0(r,h,word)          
P(28) = s1(l,h,word)        
P(29) = s1(r,h,word)            



# Morphological features #

## erl ##
P(30) = s0(t,h,erl)
P(31) = s1(t,h,erl)
P(32) = q0(erl)
P(33) = q1(erl)

## nork ##
P(34) = s0(t,h,nork)
P(35) = s1(t,h,nork)
P(36) = q0(nork)
P(37) = q1(nork)

## num ##
P(38) = s0(t,h,num)
P(39) = s1(t,h,num)
P(40) = q0(num)
P(41) = q1(num)

## aspect ##
P(42) = s0(t,h,aspect)
P(43) = s1(t,h,aspect)
P(44) = q0(aspect)
P(45) = q1(aspect)

## nori ##
P(46) = s0(t,h,nori)
P(47) = s1(t,h,nori)
P(48) = q0(nori)
P(49) = q1(nori)

## dadudio ##
P(50) = s0(t,h,dadudio)
P(51) = s1(t,h,dadudio)
P(52) = q0(dadudio)
P(53) = q1(dadudio)

## nor ##
P(54) = s0(t,h,nor)
P(55) = s1(t,h,nor)
P(56) = q0(nor)
P(57) = q1(nor)

## case ##
P(58) = s0(t,h,case)
P(59) = s1(t,h,case)
P(60) = q0(case)
P(61) = q1(case)

## mood ##
P(62) = s0(t,h,mood)
P(63) = s1(t,h,mood)
P(64) = q0(mood)
P(65) = q1(mood)

