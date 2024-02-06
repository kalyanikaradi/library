#!/usr/bin/env python
# coding: utf-8

# # NUMPY

# In[1]:


import numpy as np
#np is used as numpy


# In[2]:


arr1 = np.array([])
arr1


# In[3]:


my_list=[1,2,3,4,5]
print(my_list)
print(type(my_list))


# In[4]:


a= np.array(my_list)


# In[5]:


a


# In[6]:


type(a)#nd array is number dimension array


# In[7]:


a.ndim #used to check number of dimensions in array 


# In[8]:


a.size # used to check number of items


# In[9]:


a.shape #shape of array


# In[10]:


# array can be of n dimension

my_matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
my_matrix


# In[11]:


b=np.array(my_matrix)#generates a 2-d array
b


# In[12]:


#array summary
print('the Dimension of array',b.ndim)# dimensions of given array


# In[13]:


print('the size of array:',b.size)#number of elements in array


# In[14]:


print('the datatype of element:',b.dtype)#datatype of elements in array


# In[15]:


print('the type of structure:',type(b))


# In[16]:


print('the shape:',b.shape)


# In[17]:


b.shape


# In[18]:


arr1=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
arr1


# In[19]:


arr1.shape


# In[20]:


arr1.ndim


# In[21]:


arr1.size


# In[22]:


#reshape
arr1


# In[23]:


arr2=arr1.reshape((1,3,4))#reshape function takes new shape


# In[24]:


arr2.shape


# In[25]:


arr2


# In[26]:


#arange
np.arange(15)


# In[27]:


np.arange(100,2,-4)


# In[28]:


np.arange(0,11,2)


# In[29]:


#zeros
import numpy as np


# In[30]:


np.zeros(3)


# In[31]:


np.zeros((5,5))


# In[32]:


#ones
import numpy as np


# In[33]:


np.ones(3)


# In[34]:


np.ones((6,3))


# In[35]:


np.ones((4,5,6))


# In[36]:


#linspace

import numpy as np


# In[37]:


np.linspace(1,15)#default 50 obsetvations


# In[38]:


np.linspace(1,50)


# In[39]:


np.linspace(5,25,10)#equally spaced 10 values


# In[40]:


# retstep return step computed by linspace
np.linspace(0,25,retstep=True)#start #end (here end is included)


# In[41]:


np.linspace(0,200,10)#default retstep=False


# In[42]:


np.linspace(0,200,10,retstep=True)


# In[43]:


#eye
np.eye(5)# genertes 2d array of(5,5)


# In[44]:


np.eye(9)


# In[45]:


#create an eye from a zeros 


# In[46]:


#boardcasting in array
big_one=np.ones((3,4))
print(big_one)


# In[47]:


big_one.dtype


# In[48]:


big_one * 3


# In[49]:


bigger_one=big_one *6 -2
bigger_one


# In[50]:


bigger=np.array(big_one*3 -0.4, dtype='int')
print(bigger)


# In[51]:


bigger.dtype


# In[52]:


type(bigger)


# In[53]:


bigger.shape


# In[54]:


bigger/bigger


# In[55]:


arr1 = np.arange(20)
1/arr1


# In[56]:


arr1 + arr1


# In[57]:


arr1 ** arr1#** is power


# In[58]:


#use of copy function
arr2 = arr1


# In[59]:


arr2


# In[60]:


arr2[:10] = 30# using indexing to modify arr2
arr2


# In[61]:


arr1


# In[62]:


arr3 = arr1.copy() # generate a copy / creates a backup
arr3[:10]=100


# In[63]:


arr3


# In[64]:


arr3[10:]=100


# In[65]:


arr3


# In[66]:


arr1


# In[67]:


#random number generation
import numpy as np


# In[68]:


np.random.rand()#gives random values from 0 to1 


# In[69]:


np.random.rand(10)#values frpm 0 to 10


# In[70]:


np.random.rand(10).reshape((2,5))


# In[71]:


#creating array from uniform distribution
new_arr=np.random.rand(5,3)
#2 dimensional array of shape(5,3)


# In[72]:


new_arr


# In[73]:


np.random.rand(3,3)


# In[74]:


arr1=np.random.randn(50)#50 observations from std normal distribution


# In[75]:


arr1


# In[76]:


max(arr1)


# In[77]:


min(arr1)


# In[78]:


np.random.randn(3,3)


# In[79]:


#randint
np.random.randint(1,100)


# In[80]:


np.random.randint(1,100,10)


# In[81]:


np.random.randint(40,60,50)


# In[82]:


arr=np.arange(20)
ranarr = np.random.randint(0,100,10)


# In[83]:


arr


# In[84]:


ranarr


# In[85]:


#max min argmAX ARGMIN
ranarr


# In[86]:


ranarr.max()


# In[87]:


arr.max()


# In[88]:


ranarr.argmax()#index location of highest ele


# In[89]:


ranarr.min()


# In[90]:


ranarr.argmin()


# In[91]:


ran2=ranarr.reshape(2,5)


# In[92]:


ran2


# In[93]:


ran2.argmax()


# In[94]:


ran2.max()


# In[95]:


#NUMPY INDEXING AND SELECTION


# In[96]:


import numpy as np


# In[97]:


#creating sample array
arr=np.arange(0,21)


# In[98]:


arr


# In[99]:


#creating sample array
arr=np.arange(10,100,5)
arr


# In[100]:


len(arr)


# In[101]:


arr[-1]


# In[102]:


#get a value at an index
arr[9]


# In[103]:


arr[1:11:2]


# In[104]:


#filtering
arr=np.array([1,2,1010,4,108,18,71,610])
arr


# In[105]:


arr<100


# In[106]:


arr[5]


# In[107]:


#where condition is used op index loc values
np.where(arr>100)


# In[108]:


np.where(arr<100)


# In[109]:


np.where(arr==100)


# In[110]:


import numpy as np


# In[111]:


arr_2d = np.array(([1,2,3],[12,15,18],[64,96,128]))
arr_2d


# In[112]:


arr_2d[1:,:2]


# In[113]:


arr_2d[2,2]


# In[114]:


arr_2d[:,2]


# In[115]:


arr_2d.shape


# In[116]:


arr_2d[1]


# In[117]:


arr_2d[:,2]


# In[118]:


arr_2d[1:,1:]


# In[119]:


arr_2d[1,1]


# In[120]:


arr_2d[0:2]


# In[121]:


arr_2d[1,1]


# In[122]:


arr_2d[0:2]


# In[123]:


arr_2d[:,2]


# In[124]:


#slicing
arr_2d[:2,1:]


# In[125]:


#FANCY INDEXING
arr1=np.ones((10,10))
arr1.shape


# In[126]:


arr2d = np.zeros(arr1.shape)


# In[127]:


arr2d


# In[128]:


arr2d.shape


# In[129]:


no_of_rows=arr2d.shape[0]


# In[130]:


#set up array
j=0
for i in range (no_of_rows):
    arr2d[i]=np.arange(j+1,j+11)
    j+=10
arr2d


# In[131]:


arr2d[[3,7]]


# In[132]:


arr2d[[6,3,8]]


# In[133]:


arr2d[:,[2,4,6,8]]


# In[134]:


arr2d[[6,4,2,7]]


# In[135]:


arr2d


# In[136]:


arr2d[:,0]<40


# In[137]:


arr2d[:,4]<20


# In[138]:


arr2d[2:,]<30


# In[139]:


arr2d[np.where(arr2d[:,0]<40)]


# In[140]:


arr2d[np.where(arr2d[0,:]%2==0)]


# # PANDA

# In[141]:


import pandas as pd


# In[142]:


#SERIES


# In[143]:


#creating empty series
d=pd.Series()


# In[144]:


d


# In[145]:


print(type(d))


# In[146]:


#creating series with one element
d=pd.Series(17)


# In[147]:


d


# In[148]:


#creating series using tuple
t=(10,11,12)
d=pd.Series(t)


# In[149]:


d


# In[150]:


#creating series using list
d=pd.Series([45,78,56,445,78])


# In[151]:


d


# In[ ]:





# In[152]:


#creating series with array
import numpy as np
arr=np.array([1,2,3,4])
d1=pd.Series(arr)
d1


# In[154]:


arr2=np.array([[1,2],[3,4],[5,6]])
arr2.shape


# In[155]:


s2=pd.Series(arr2)


# In[156]:


d


# In[157]:


#using dictionary
d={'a':1,'b':2,'c':3,'d':4}


# In[158]:


d.values()


# In[159]:


d.keys()


# In[160]:


d.items()


# In[161]:


d['a']


# In[162]:


b=pd.Series(d)


# In[163]:


b


# In[164]:


b[0]


# In[165]:


b['a']


# In[166]:


#slicing
d=pd.Series([1,2,3,4,5,6,77,88,765,543,33,54,67])
d


# In[167]:


d[0:9]


# In[168]:


d[[4,5,6,3]]


# In[169]:


d[4]=34#changing value


# In[170]:


d[[4,5,3,6]]


# In[171]:


d[6:10]=[100,200,300,400]


# In[172]:


d


# In[173]:


arr3=np.array([1,2,4,6])
s=pd.Series(arr3,index=['one','two','three','four'])
s


# In[174]:


arr4=np.array([1,2,4,6])
s1=pd.Series(arr4)
s1


# In[175]:


print(s)


# In[176]:


#DATAFRAME
import pandas as pd
d=pd.DataFrame()


# In[177]:


print(d)


# In[178]:


d


# In[179]:


print(type(d))


# In[180]:


#import pandas as pd


# In[181]:


data=[['abc',1],['def',2],['ghi',3],['','hij'],['klm','a']]


# In[182]:


data


# In[183]:


d1=pd.DataFrame(data)


# In[184]:


d1


# In[185]:


#dataframe using dict
data={'name':['a','b','c','d','e','f'],'age':[45,12,48,25,12,56]}


# In[186]:


d=pd.DataFrame(data)


# In[187]:


d


# In[188]:


d.rename({'name':'candidate','age':'age_years'},axis=1,inplace=True)


# In[189]:


d


# In[190]:


data=[['Alex',10,'maths'],['bob',12,'Science'],['Kelly',15,'eco'],
     ['Boris',14,'geo'],['Ken',18,'English']]


# In[191]:


data


# In[192]:


data[2][0]


# In[193]:


data[3][2]


# In[194]:


data[2][0]='asdf'


# In[195]:


data


# In[196]:


import pandas as pd


# In[200]:


d=pd.DataFrame(data,columns=['name','age','subject'])


# In[201]:


d


# In[205]:


d(1,1)          


# In[206]:


d.name


# In[207]:


#accessing one single column
d['name']


# In[208]:


d[['name','subject']]


# In[209]:


d.iloc[1,1]#iloc is index location


# In[210]:


d


# In[211]:


d.iloc[0,0]="ALEX"


# In[212]:


d


# In[213]:


d['Gender']='Male'


# In[214]:


d


# In[215]:


d['age']=d['age']-1#each age -1


# In[216]:


d


# In[217]:


d.iloc[:,[0,1]]


# In[218]:


d.iloc[[3,1],[0,1]]


# In[219]:


d.iloc[1:3,[0,1]]


# In[220]:


d.iloc[1]


# In[221]:


pwd


# In[222]:


d


# In[ ]:





# In[ ]:




