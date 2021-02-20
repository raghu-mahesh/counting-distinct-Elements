# counting-distinct-Elements using FM Algorithm and hyperloglog
import numpy as np 
import pandas as pd 
import hashlib
from collections import defaultdict
import statistics
df=pd.read_csv('/content/supermarket_sales - Sheet1.csv')
df.dtypes
df.head(15)
df.shape 
product=df['Product line']
type(prod_list)
def hash_string_1(s): 
     return hashlib.sha1(str.encode(s)).hexdigest() 
def hash_string_2(s): 
      return hashlib.sha384(str.encode(s)).hexdigest()
def hash_string_3(s):
   return hashlib.sha512(str.encode(s)).hexdigest()
def hash_string_4(s):
    return hashlib.md5(str.encode(s)).hexdigest()

def hexstring_to_int(s):
    return int(s, 16)
def int_to_bin(i):
  return bin(i)
def trailing_zeroes(bs): 
    s = str(bs)
    s = s[2:]

    l= len(s)-len(s.rstrip('0')) 
    if l==len(s):
        return 0 
    else:
        return l
mtz=[-1,-1,-1,-1]
count=[]
for name in prod_list: 
    hstring=hash_string_1(name) 
    hint=hexstring_to_int(hstring) 
    hbin=int_to_bin(hint) 
    tz=trailing_zeroes(hbin)
    if mtz[0]==-1 or mtz[0]<tz: 
        mtz[0]=tz
count.append(2**mtz[0])

for name in prod_list:
    hstring=hash_string_2(name)
    hint=hexstring_to_int(hstring)
    hbin=int_to_bin(hint)
    tz=trailing_zeroes(hbin)
    if mtz[1]==-1 or mtz[1]<tz:
        mtz[1]=tz
count.append(2**mtz[1])

for name in prod_list:
    hstring=hash_string_3(name) 
    hint=hexstring_to_int(hstring)
    hbin=int_to_bin(hint)
    tz=trailing_zeroes(hbin)
    if mtz[2]==-1 or mtz[2]<tz: 
        mtz[2]=tz
count.append(2**mtz[2])

for name in prod_list:
    hstring=hash_string_4(name)
    hint=hexstring_to_int(hstring) 
    hbin=int_to_bin(hint) 
    tz=trailing_zeroes(hbin)
    if mtz[3]==-1 or mtz[3]<tz:
        mtz[3]=tz
count.append(2**mtz[3])
print('Using FM algorithm to find distinct elements') 
print('Count of distinct product_name are :')
print('Using sha1 hash : ',count[0]) 
print('Using sha384 hash : ',count[1]) 
print('Using sha512 hash : ',count[2]) 
print('Using md5 hash : ',count[3]) 
import statistics
set1 = count[:2] 
set2 = count[2:]
s1=statistics.mean(set1) 
s2=statistics.mean(set2) 
median = (s1+s2)/2
print('Distinct number of product name using fm algorithm mean-median approximation :') 
print(median)
corr=map(lambda x: x/0.71351,count) 
cr=list(corr)
print('Count of distinct product_name after applying correction factor :') 
print('Using sha1 hash : ',cr[0])
print('Using sha384 hash : ',cr[1]) 
print('Using sha512 hash : ',cr[2]) 
print('Using md5 hash : ',cr[3])
unq_prod=len(df['Product line'].unique()) 
fm1=median
print('The count of actual number of distinct product names :') 
print(unq_prod)
error=median-unq_prod

error_rate=error/unq_prod*100 
fm1e=error_rate
print('The % error of fm algorithm is ') 
print(error_rate)
outcome = df['gross income'].values
print(outcome)
print(type(outcome))
dis=outcome.tolist()
print(dis)
con=list(map(int, dis))
print(con)
print("Hash functions are defined as (a*x+b)\%c, where x is an element of the set.")
inputCount = int(input("Enter the number of hash functions: "))
abcList = []

for i in range(inputCount):
  inputList = input("Enter the space-separated values of a, b and c: ").split(" ")
  abcList.append([int(i) for i in inputList])
finalCountsRecorded = []

for i in abcList:
  binElems = []
  for j in set(con):                          
    binElems.append(str(bin((i[0]*j+i[1])%i[2])).split("b")[1])   
  greatestTrailing = 0                         
  for k in binElems:                          
    reversedCount = k[::-1]       
    count = 0
    for i in reversedCount:                     
      if(i=='1'):
        if(count>greatestTrailing):
          greatestTrailing = count              
        break
      else:
        count+=1
  finalCountsRecorded.append(2**greatestTrailing)         
                                    
print("Counts recorded for each hash: ",finalCountsRecorded)

divider = inputCount//2
set1 = finalCountsRecorded[:divider]
set2 = finalCountsRecorded[divider:]
s1=statistics.median(set1)
s2=statistics.median(set2)
median_t = [(s1+s2/2)]          
                        

print("Approximate number of elements from mean-median approximation: ",(median_t))
med=list(map(int, median_t))
print(med)
strings = [str(integer) for integer in med]
a_string = "".join(strings)
an_integer = int(a_string)

print(an_integer)
rt_actual=len(df1['outcome'].unique()) 
fm2=an_integer
print('the count of actual number of distinct retail price is :') 
rt_actual
error=abs(an_integer-rt_actual) 
error_rate=error/rt_actual*100 
fm2e=error_rate
print('% error using fm algorithm :')

print(error_rate) 

def first_nonzero_bit(bs): 
  bs = bs[2:]
  #return len(bs) - bs.index('1') 
  return bs[::-1].index('1')
def element_to_register_nonzero(elem): 
    elem_hash = hash_string_1(elem)
    register = hexstring_to_int(get_hll_register_num(elem_hash)) 
    hll_hash = get_hll_hash(elem_hash)
    fnz_bit = first_nonzero_bit(int_to_bin(hexstring_to_int(hll_hash))) 
    return (register, fnz_bit)
def cardinality_estimate(maxbits):

    tot_regs = len(maxbits)

    two_inv_sum = sum(map(lambda m: pow(2, -1*m), maxbits))

    return 0.7213/(1 + 1.079/tot_regs) * pow(tot_regs,2) * 1/two_inv_sum 
def get_hll_register_num(hs):

    return hs[:3]
def get_hll_hash(hs):

    return hs[-14:]
def HLL(items):

    dim_reg_maxbit = defaultdict(lambda: defaultdict(int)) 
    for item in items:
        i_dim, i_elem = item,item
        i_reg, i_fnz_bit = element_to_register_nonzero(i_elem) 
        dim_reg_maxbit[i_dim][i_reg] = max(dim_reg_maxbit[i_dim][i_reg], i_fnz_bit)
    estimates = []

    for dim in dim_reg_maxbit:
        maxbits = [v for _, v in dim_reg_maxbit[dim].items()] 
        estimates.append((dim, cardinality_estimate(maxbits)))
    return estimates
def count(items):

    res = defaultdict(lambda: set([])) 
    for dim in items:
        res[dim].add(dim)
    return [(k, len(v)) for k, v in res.items()]
def print_cmp(estimates, actuals): 
  width = 55
  print("{: <{width}}\t{: <{width}}\t{: <{width}}".format("product name", "estimate", "actual", width=width))
  for i, j in zip(estimates, actuals):
    print("{: <{width}}\t{: <{width}}\t{: <{width}}".format(i[0], i[1], j[1], width=width))
hll_counts = HLL(prod_list) 
actual_counts = count(prod_list)

print('Distinct Product name - estimated vs actual number of occurences using hyperloglog algorithm : ')
print_cmp(hll_counts, actual_counts)

hlcount=0

for i in hll_counts: hlcount+=1
print('Total number of distinct product names using hyperloglog algorithm :') 
print(hlcount)
from HLL import HyperLogLog

hll = HyperLogLog(10) 
for i in prod_list:
    hll.add(i)

estimate = hll.cardinality() 
hll
estimate
hll1=estimate
unq_prod=len(df['Product line'].unique())
print('Actual count of Distinct product names is :')
unq_prod

error=abs(estimate-unq_prod) 
error_rate=error/unq_prod*100 
hll1e=error_rate
print('%error in hll algorithm is :') 
error_rate
hll = HyperLogLog(12) 
for i in outcome :
  hll.add(i)

estimate = hll.cardinality() 
hll2=estimate
print('Distinct product names using hyperloglog python librabry :',estimate) 
rt_actual=len(df['gross income'].unique())
print('Actual count of Distinct retail prices is :') 
rt_actual
error=abs(estimate-rt_actual) 
error_rate=error/rt_actual*100 
hll2e=error_rate
print('%error in hll algorithm is :') 
error_rate
x = PrettyTable()

x.field_names = ["Compare","Actual count","FM Count","FM_ERROR_%","HLL Count","HLL_ERROR_%"]
x.add_row(["Product line",unq_prod,fm1,fm1e,hll1,hll1e]) 
x.add_row(["gross income",rt_actual,fm2,fm2e,hll2,hll2e])

print(x)
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set()
blue_bar = (fm1e,fm2e)
orange_bar = (hll1e,hll2e) 
ind = np.arange(2) 
plt.figure(figsize=(10,5)) 
width = 0.3
plt.bar(ind, blue_bar , width, label='flajolet martin') 
plt.bar(ind + width, orange_bar, width, label='hyperloglog')

plt.xlabel('element') 
plt.ylabel('error rate %') 
plt.title('FM vs HLL')
plt.xticks(ind + width / 2, ('product name', 'gross income'))

plt.legend(loc='best') 
plt.show()
