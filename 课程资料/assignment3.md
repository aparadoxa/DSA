# Assignment #3: 惊蛰 Mock Exam

Updated 1641 GMT+8 Mar 5, 2025

2025 spring, Complied by <mark>郑涵予 物理学院</mark>



> **说明：**
>
> 1. **惊蛰⽉考**：<mark>AC4</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
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

### E04015: 邮箱验证

strings, http://cs101.openjudge.cn/practice/04015



思路：

上学期计概就做过，，唯一的坑点是容易忽略'.'在'@'前面的情况。虽然知道要可以用正则表达式但是考场上赶时间还是直接写了暴力检验的方法。（用时约5min)



代码：

```python
def check(s):
    if s.count('@')!=1:return False
    if s[0]=='@' or s[0]=='.' or s[-1]=='@' or s[-1]=='.':return False
    if '.@' in s or '@.' in s:return False
    n=len(s)
    delta=False
    for i in range(n):
        if s[i]=='@':delta=True
        if delta and s[i]=='.':return True
    return False

while 1:
    try:
        s=input()
        print("YES" if check(s) else "NO")
    except:break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>



![](C:\Users\z3359\Pictures\Screenshots\屏幕截图 2025-03-05 233606.png)

### M02039: 反反复复

implementation, http://cs101.openjudge.cn/practice/02039/



思路：

直接按照题目要求进行模拟即可（用时约5min)



代码：

```python
n=int(input())
s=input()
m=len(s)//n
ans=[]
index=0
for i in range(m):
    temp=[]
    for j in range(n):
        temp.append(s[index])
        index+=1
    if i&1:ans.append(temp[::-1].copy())
    else:ans.append(temp.copy())
res=""
for j in range(n):
    for i in range(m):
        res+=ans[i][j]
print(res)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\z3359\Pictures\Screenshots\屏幕截图 2025-03-05 233708.png)



### M02092: Grandpa is Famous

implementation, http://cs101.openjudge.cn/practice/02092/



思路：

这题事实上并不难，但是纯英文题干有些难以理解，最后还是结合样例才猜出了题目是什么意思。（用时约10min)



代码：

```python
while 1:
    n,m=map(int,input().split())
    if not n and not m:break
    mp={}
    M=0
    for _ in range(n):
        a=list(map(int,input().split()))
        for x in a:
            if x not in mp:mp[x]=1
            else:
                mp[x]+=1
            M=max(M,mp[x])
    ans=[]
    s=-1
    for key in mp.keys():
        if mp[key]!=M:s=max(s,mp[key])
    for key in mp.keys():
        if mp[key]==s:ans.append(key)
    print(' '.join(map(str,sorted(ans))))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\z3359\Pictures\Screenshots\屏幕截图 2025-03-05 233741.png)



### M04133: 垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/



思路：

这也是计概中出现过的题，直接利用max和min来防止越界可以减小讨论的代码量。（用时约10min)

代码：

```python
d=int(input())
n=int(input())
a=[[0]*1025 for _ in range(1025)]
for _ in range(n):
    x,y,w=map(int,input().split())
    for i in range(max(0,x-d),min(1025,x+d+1)):
        for j in range(max(0,y-d),min(1025,y+d+1)):
            a[i][j]+=w
M=0
for i in range(1025):
    for j in range(1025):
        M=max(M,a[i][j])
num=0
for i in range(1025):
    for j in range(1025):
        if a[i][j]==M:num+=1
print(num,M)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\z3359\Pictures\Screenshots\屏幕截图 2025-03-05 233916.png)



### T02488: A Knight's Journey

backtracking, http://cs101.openjudge.cn/practice/02488/



思路：

题目中要求的是字典序最小，所以在bfs的时候各个方向的位移要有顺序地枚举，另外可以用pre数组来记录节点的上一个节点是什么。然而考试的时候看错题目了没做出来（用时约1h)

代码：

```python
t=int(input())
directions=[(-1,-2),(1,-2),(-2,-1),(2,-1),(-2,1),(2,1),(-1,2),(1,2)]
for case in range(1,t+1):
    m,n=map(int,input().split())
    a=[[False]*n for _ in range(m)]
    pre=[[(-1,-1)]*n for _ in range(m)]
    delta=False
    path=[]
    def dfs(i,j,step):
        global delta
        if delta:return
        if step==m*n:
            delta=True
            nx,ny=i,j
            while nx!=-1 and ny!=-1:
                path.append((nx,ny))
                temp=pre[nx][ny]
                nx,ny=temp[0],temp[1]
            return
        for dx,dy in directions:
            nx,ny=i+dx,j+dy
            if 0<=nx<m and 0<=ny<n and not a[nx][ny] and not delta:
                a[nx][ny]=True
                pre[nx][ny]=(i,j)
                dfs(nx,ny,step+1)
                a[nx][ny]=False
                pre[nx][ny]=(-1,-1)
    ans=""
    for j in range(n):
        if delta:break
        for i in range(m):
            if delta:break
            a[i][j]=True
            dfs(i,j,1)
            a[i][j]=False
            if delta:
                for res in path[::-1]:
                    ans+=chr(ord('A')+res[1])
                    ans+=str(res[0]+1)
    print("Scenario #{}:".format(case))
    print(ans if delta else "impossible")
    print()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>



![](C:\Users\z3359\Pictures\Screenshots\屏幕截图 2025-03-05 234129.png)

### T06648: Sequence

heap, http://cs101.openjudge.cn/practice/06648/



思路：

思路非常巧妙的一道题，自己做题的时候想到了两两合并保留前n个和的做法，但是没有使用heapq进行合并导致python代码会MLE,换用C++试图强行通过后发现又TLE了，最后看了题解成功AC.(用时约1h)

代码：

```python
import heapq
n=0
def merge(res,a):
    q=[]
    temp=[]
    visited=set()
    visited.add((0,0))
    heapq.heappush(q,(a[0]+res[0],0,0))
    while len(temp)<n:
        s,i,j=heapq.heappop(q)
        temp.append(s)
        if i+1<len(res) and (i+1,j) not in visited:
            visited.add((i+1,j))
            heapq.heappush(q,(res[i+1]+a[j],i+1,j))
        if j+1<len(a) and (i,j+1) not in visited:
            visited.add((i,j+1))
            heapq.heappush(q,(res[i]+a[j+1],i,j+1))
    return temp
t=int(input())
for _ in range(t):
    m,n=map(int,input().split())
    res=list(map(int,input().split()))
    res.sort()
    for __ in range(m-1):
        a=list(map(int,input().split()))
        a.sort()
        res=merge(res,a)
    print(' '.join(map(str,res)))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\z3359\Pictures\Screenshots\屏幕截图 2025-03-05 234525.png)



## 2. 学习总结和收获

<mark>如果发现作业题目相对简单，有否寻找额外的练习题目，如“数算2025spring每日选做”、LeetCode、Codeforces、洛谷等网站上的题目。</mark>



这次月考的一个惨痛教训就是要好好读题。第五题一上来看到骑士周游就先入为主地想到之前做过的启发式搜索，又因为题目是英文的没完全看懂，导致直接忽略了字典序最小这个要求，一直WA了十来次也始终找不到bug.剩余不到10min的时候才猛然发现题目读错了，可惜也没能抓住最后的机会改完代码。想起做上学期机考题蛇入迷宫的时候一开始也是因为没想到蛇可以在身体横着的情况下竖着移动，身体竖着的情况下横着移动（虽然这确实很反直觉）而在一开始写出了错误的代码（不过或许是因为上学期并未选课，在考场下做比较放松，很快发现了这个问题，这也说明考场心态还是不稳，还是要多练），愈发觉得把题读对才是最重要的，否则写再多代码都白搭。







