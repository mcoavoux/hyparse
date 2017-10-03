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


P(30) = s0(t,h,mwe)
P(31) = s0(r,h,mwe)
P(32) = s0(l,h,mwe)
P(33) = s1(t,h,mwe)

P(34) = q0(mwe)
P(35) = q1(mwe)
P(36) = q2(mwe)
P(37) = q3(mwe)

P(38) = s0(t,h,num)
P(39) = s1(t,h,num)
P(40) = q0(num)
P(41) = q1(num)

P(42) = s0(t,h,gen)
P(43) = s1(t,h,gen)
P(44) = q0(gen)
P(45) = q1(gen)

P(46) = s0(t,h,mood)
P(47) = s1(t,h,mood)
P(48) = q0(mood)
P(49) = q1(mood)





### slightly modified from
#P(38) = s0(t,h,num)
#P(39) = s1(t,h,num)
#P(40) = s0(l,h,num)

#P(41) = s0(t,h,gen)
#P(42) = s1(t,h,gen)
#P(43) = s0(l,h,gen)

#P(44) = s0(t,h,mood)
#P(45) = s1(t,h,mood)
#P(46) = s0(l,h,mood)

