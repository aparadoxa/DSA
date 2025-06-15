# Assignment #7: 20250402 Mock Exam

Updated 1624 GMT+8 Apr 2, 2025

2025 spring, Complied by <mark>郑涵予 物理学院</mark>



> **说明：**
>
> 1. **⽉考**：AC<mark>6</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>
> 2. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 3. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 4. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### E05344:最后的最后

http://cs101.openjudge.cn/practice/05344/



思路：

直接用队列进行模拟即可（用时约2min)

代码：

```python
n,k=map(int,input().split())
from collections import deque
q=deque()
for i in range(1,n+1):
    q.append(i)
num=0
res=[]
while len(q)>1:
    num+=1
    if num==k:
        res.append(q.popleft())
        num=0
    else:
        q.append(q.popleft())
print(' '.join(map(str,res)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\z3359\Pictures\Screenshots\屏幕截图 2025-04-02 205637.png)



### M02774: 木材加工

binary search, http://cs101.openjudge.cn/practice/02774/



思路：

二分查找经典题目，直接写就行（用时约3min)

代码：

```python
n,k=map(int,input().split())
a=[]
for _ in range(n):
    a.append(int(input()))
def check(l):
    num=0
    for x in a:
        num+=x//l
        if num>=k:
            return True
    return False
left,right=0,sum(a)//k
while left<right:
    mid=(left+right+1)//2
    if check(mid):
        left=mid
    else:
        right=mid-1
print(left)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\z3359\Pictures\Screenshots\屏幕截图 2025-04-02 205746.png)



### M07161:森林的带度数层次序列存储

tree, http://cs101.openjudge.cn/practice/07161/



思路：

这题甚至感觉是这次考试里最难的,一开始先跳过了，做完第五题才回来做掉这题。思路大概是用一个队列来储存访问到的节点，每次从队列左边出队取节点。（用时约20min)

代码：

```python
from collections import deque
class Node():
    def __init__(self,val="",num=0):
        self.val=val
        self.num=num
        self.children=[]

res=[]
def postorder(root):
    if not root:
        return
    for x in root.children:
        postorder(x)
    res.append(root.val)

def create(s):
    root=Node(s[0])
    index=0
    root.num=int(s[1])
    q=deque()
    q.append(root)
    while q:
        cur=q.popleft()
        for i in range(cur.num):
            index+=2
            node=Node(s[index],int(s[index+1]))
            q.append(node)
            cur.children.append(node)
    return root

n=int(input())
for _ in range(n):
    s=input().split()
    root=create(s)
    postorder(root)
print(' '.join(res))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\z3359\Pictures\Screenshots\屏幕截图 2025-04-02 210118.png)



### M18156:寻找离目标数最近的两数之和

two pointers, http://cs101.openjudge.cn/practice/18156/



思路：

题目给了双指针所以就把数组排序后用双指针历遍即可（用时约5min)

代码：

```python
target=int(input())
a=list(map(int,input().split()))
a.sort()
n=len(a)
i,j=0,n-1
res=float('inf')
delta=float('inf')
while j>i:
    if abs(a[i]+a[j]-target)<=delta:
        if a[i]+a[j]<res or abs(a[i]+a[j]-target)<delta:
            res=a[i]+a[j]
        delta=abs(a[i]+a[j]-target)
    if a[i]+a[j]>target:
        j-=1
    elif a[i]+a[j]<target:
        i+=1
    else:
        break
print(res)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\z3359\Pictures\Screenshots\屏幕截图 2025-04-02 205908.png)



### M18159:个位为 1 的质数个数

sieve, http://cs101.openjudge.cn/practice/18159/



思路：

先把符合条件的素数都找出来，然后查找即可（用时约5min)

代码：

```python
t=int(input())
a=[]
from math import sqrt
def check(x):
    if str(x)[-1]!='1':
        return False
    for i in range(3,int(sqrt(x))+1):
        if x%i==0:
            return False
    return True
for i in range(9,10002,2):
    if check(i):
        a.append(i)
for _ in range(t):
    n=int(input())
    ans=[]
    for x in a:
        if x>=n:
            break
        ans.append(x)
    print("Case{}:".format(_+1))
    if not ans:
        print('NULL')
    else:
        print(' '.join(map(str,ans)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\z3359\Pictures\Screenshots\屏幕截图 2025-04-02 205942.png)



### M28127:北大夺冠

hash table, http://cs101.openjudge.cn/practice/28127/



思路：

刚好现在在学数算，写一个类定义一下比较函数可以很方便地进行排序（用时约10min)

代码：

```python
class Group():
    def __init__(self,name):
        self.name=name
        self.problem=set()#记录做对题目数量
        self.time=0#提交次数
    def __lt__(self,other):
        if len(self.problem)!=len(other.problem):
            return len(self.problem)>len(other.problem)
        if self.time!=other.time:
            return self.time<other.time
        return self.name<other.name

m=int(input())
mp={}#名字到队伍的映射
res=[]
for _ in range(m):
    s=input().split(',')
    if s[0] not in mp:
        g=Group(s[0])
        mp[s[0]]=g
        res.append(g)
    temp=mp[s[0]]
    temp.time+=1
    if s[-1]=='yes':
        temp.problem.add(s[1])
res.sort()
for i in range(min(12,len(res))):
    print(i+1,end=" ")
    print(res[i].name,len(res[i].problem),res[i].time,end=" ")
    print()
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\z3359\Pictures\Screenshots\屏幕截图 2025-04-02 210317.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>

这次机考比较基础，重温了一下python里面字典、集合的用法，感觉并没有涉及很多数算内容，希望期末不要比这难太多。临近期中没太多时间写代码，还在吃寒假的老本。期中考完得好好复习一下各种算法了。









