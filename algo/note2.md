---
title: 算法题记二
date: 2024-07-15
categories:
  - 算法
tags:
  - 题目
plugins:
  - mathjax
description: 积累一些有意思的题目
---

# 算法题笔记二

### 数组每个数往左往右最多可以吸收多少个数

题源 [ARC189D](https://atcoder.jp/contests/arc189/tasks/arc189_d)

假设你是第 $k$ 只史莱姆，每次操作，你可以吃掉左右相邻的体积严格小于你的一只史莱姆，并获得它的体积。把你可以达到的最大体积记为 $f(k)$。求出所有的 $f(i)$。

题解做法：对 $a[k]$，当前吸收区间为 $[l,r]$，希望向左向右扩展区间，直到 $a[l-1]\geq sum(l,r)$ 为止，这个值可以通过 ST 表上二分求出来。假设得到新区间 $[l',r']$，继续对新区间求解。每次扩展，体积至少变为原来的两倍，因此最多扩展 $logV$ 次

灵神做法：
从大到小思考。

a 中最大值的答案是多少？次大值的答案是多少？

设：

a[i] 左边最近 > a[i] 的数的下标为 left[i]，若不存在则为 -1。

a[i] 左边最近 >= a[i] 的数的下标为 leftGE[i]，若不存在则为 -1。

a[i] 右边最近 > a[i] 的数的下标为 right[i]，若不存在则为 n。

a[i] 右边最近 >= a[i] 的数的下标为 rightGE[i]，若不存在则为 n。


分类讨论：

如果 rightGE[i]-leftGE[i]=2，那么啥也做不了，ans[i] = a[i]。

否则，由于 a 中元素都是正数，在吃掉一个史莱姆后，我们的体积一定比初始体积大。

计算从 left[i]+1 到 right[i]-1 的子数组和 s。

如果 left[i] >= 0，且子数组和 s 比 a[i] 左边更大数还要大，那么左边更大数能吃掉的数，a[i] 也可以吃掉，所以 ans[i] 就是左边更大数的答案，即 ans[i] = ans[left[i]]。

否则如果 right[i] < n，且子数组和 s 比 a[i] 右边更大数还要大，那么右边更大数能吃掉的数，a[i] 也可以吃掉，所以 ans[i] 就是右边更大数的答案，即 ans[i] = ans[right[i]]。

其他情况，ans[i] = s。

代码见[这里](https://atcoder.jp/contests/arc189/submissions/61900999)

### 数组中多少 a[i]+a[j] 的和满足二进制前 k-1 位为 0，第 k 位不为 0

题源 [ABC384F](https://atcoder.jp/contests/abc384/tasks/abc384_f) 

求 $\displaystyle \sum_{i=1}^N \sum_{j=i}^N f(A_i+A_j)$，其中运算指的是：当二进制最低位为 $0$ 时就一直左移

思路明确：需要知道哪些 pair 的前 $k-1$ 位为 $0$，第 $k$ 位不为 $0$。做法是**记 $g[i]$ 表示前 $i$ 位都是 $0$ 的 pair 的和，用 $g[i]-g[i+1]$ 即为所求**

具体来说，如果 $(A_i+A_j)$ 被 $2^k$ 整除，意味着 $A_j\equiv -A_i \mod 2^k$。在遍历 $A_j$ 时维护 $A_i$ 的信息，这里用到了负数取模的技巧

注：atcoder 不卡 `unordered_map`，用 `map` 会超时，也可以用数组作为桶

```cpp
        LL g[26] = {};
        for (int k = 0; k <= 25; k++) {
            int msk = 1 << k;
            unordered_map<int, LL> sum;
            unordered_map<int, int> cnt;
            for (int x: a) {
                int r = (msk - x % msk) % msk; // -a[j] % 2^k
                cnt[r] ++;
                sum[r] += x;
                g[k] += 1LL * cnt[x % msk] * x + sum[x % msk];
            }
        }
        LL res = 0;
        for (int i = 0; i < 25; i++) {
            res += (g[i] - g[i + 1]) >> i;
        }
```

### 构造 n 的排列，使得数组 a 是它的 LIS，要求字典序最小

题源 [ARC125C](https://atcoder.jp/contests/arc125/tasks/arc125_c) 第一个数填什么：填小于 $A_1$ 的不行，会使 LIS 变长；填大于 $A_1$ 的也不行，无法使字典序最小，因此只能填 $A_1$

对于每个 $A_i$，它后面可以跟当前最小的不在 $a$ 中的数

填完这个数后，填小于 $A_{i+1}$ 的不行，因为会使 LIS 变长；填大于 $A_{i+1}$ 的也不行，因此只能填 $A_{i+1}$

按照这个规律填数，但是 $A_k$ 不能这样处理，否则比 $A_k$ 大的数没法填，因此要把 $n$ 到 $A_k$ 从大到小填

注意最后可能还有数没填，要记得补上去

```cpp
        for (int i = 0; i < k; i++) {
            cin >> a[i];
            st[a[i]] = 1;
        }
        int j = 1;
        for (int i = 0; i < k - 1; i++) {
            cout << a[i] << ' ';
            while (st[j]) {
                j ++;
            }
            if (j < a[i]) {
                cout << j << ' ';
                j ++;
            }
        }
        for (int i = n; i >= a[k - 1]; i--) {
            cout << i << ' ';
        }
        for (int i = a[k - 1] - 1; i >= j; i--) { // 这里别忘记
            if (!st[i]) {
                cout << i << ' ';
            }
        }
```

### 构造 n 的排列，LIS 长度为 a，LDS 长度为 b

题源 [ARC091C](https://atcoder.jp/contests/arc091/tasks/arc091_c) 首先 $a>n$ 或 $b>n$ 时是无解的

其次 $a+b>n+1$ 也是无解的，因为这意味着至少两个数同时在 LIS 和 LDS 中，意味着 $x<y$ 又 $x>y$，矛盾

构造的技巧是先构造 LIS，比如 $n=8,a=3,b=3$，先构造出 $6,7,8$，考虑在中间插入数字。这样的好处是 LDS 只会出现在 LIS 的中间，那就可以在中间插入 $b-1$ 个数即可，即 $6,2,1,\ 7,4,3,\ 8,5$

这里蕴含了一个无解条件：最多 $a$ 段，每段最长是 $b$，那么如果 $a\times b<n$，无解

```cpp
        int tot = 0, mx = n - a;
        for (int x = 0, y = n - a + 1; y <= n; y++) {
            cout << y << ' ';
            tot ++;
            for (int i = min(x + b - 1, mx); i > x; i--, tot++) {
                cout << i << ' ';
            }
            x = min(x + b - 1, mx);
        }
```

### 求最短路，但是路径的边必须是数组的子序列

题源 [ABC271E](https://atcoder.jp/contests/abc271/tasks/abc271_e)

给定 $m$ 条带权有向边，可能有重边，然后给定一个数组，找到一条从 $1$ 到 $n$ 的路径，满足路径上边的编号是数组的子序列，输出最小距离（边权和），或报告不存在

乍一看是最短路，实际上是一道 DP。按顺序读入数组，每条边选或不选，选的话就可以更新距离，相当于最短路的松弛操作

```cpp
        vector<LL> f(n + 1, 1e18);
        f[1] = 0;
        for (int i = 0, x; i < k; i++) {
            cin >> x;
            int u = edges[x].u, v = edges[x].v, w = edges[x].w;
            f[v] = min(f[v], f[u] + w);
        }
```

### 求出树中与节点 u 距离为 k 的节点

题源 [ABC267F](https://atcoder.jp/contests/abc267/tasks/abc267_f)

给定 $q$ 个询问，每次求出任意一个到 $u$ 的距离为 $k$ 的点，或报告不存在

关键思路是：对于 $u$ 找到距离它最远的点 $p$，求出以 $p$ 为根时，$u$ 的 $k$ 级祖先

最远的点一定在**直径**的端点上。需要对直径的两个端点都做一次 dfs，如何将三次 dfs 优雅地写在一个循环中呢？请看代码

```cpp
        for (int i = 0, u, k; i < q; i++) {
            cin >> u >> k;
            qs[u].push_back({i, k});
        }
        vector<int> res(q, -1);
        vector<int> nodes(n);
        for (int i = 0, rt = 1; i < 3; i++) {
            int mx = -1;
            auto dfs = [&](auto &&dfs, int u, int fa, int d) -> void {
                if (d > mx) {
                    mx = d;
                    rt = u; // 用 rt 记录下一次 dfs 的根
                }
                nodes[d] = u; // 记录路径上的点
                for (auto &[i, k]: qs[u]) {
                    if (d >= k) {
                        res[i] = nodes[d - k];
                    }
                }
                for (int v: g[u]) {
                    if (v != fa) {
                        dfs(dfs, v, u, d + 1);
                    }
                }
            };
            dfs(dfs, rt, 0, 0);
        }
```

### [LC2289 使数组按非降序排列的操作数](https://leetcode.cn/problems/steps-to-make-array-non-decreasing/description/) 思维题 转换为单调栈

题意是：在一步操作中，移除所有满足 `nums[i - 1] > nums[i]` 的 `nums[i]`

如果一个元素左边有比它更大的，那么这个元素一定是在它之后被删除的

对于一个非降序列，删除时刻一定是递增的，只用考虑它左边的元素中删除时刻最大的那个，然后 +1 就是这个数的删除时刻

```cpp
class Solution {
public:
    int totalSteps(vector<int>& nums) {
        stack<pair<int, int>> s;
        int res = 0;
        for (int x: nums)
        {
            int maxt = 0;
            while (s.size() && s.top().first <= x)
            {
                maxt = max(maxt, s.top().second);
                s.pop();
            }
            if (s.size()) maxt ++;
            res = max(res, maxt);
            s.push({x, maxt});
        }
        return res;
    }
};
```

### [LC2699 修改图中边权](https://leetcode.cn/problems/modify-graph-edge-weights/)

把所有负权都改为 `1`，最短路长度还是大于 `target` ，那么无解，因为边权只能变大。

何种顺序来增大边权？增大到多少合适呢？

第一次 dij 求 `-1` 改为 `1` 后的 `dist[]`，第二次 dij 求解增加边权的最短路

怎么改？对于当前遍历到的边，如果希望最短路走这条边，就有 $d_{1,x}+w+d_{0,e}-d_{0,y}=t$，化简得到 $w=t-d_{0,e}+d_{0,y}-d_{1,x}$

核心：那不妨再跑一遍 Dijkstra，由于 Dijkstra 算法保证每次拿到的点的最短路就是最终的最短路，所以按照 Dijkstra 算法遍历点/边的顺序去修改，就不会对**已确定的**最短路产生影响。

如果第二遍 Dijkstra 跑完后，从起点到终点的最短路仍然小于 `target`，那么就说明无法修改，返回空数组。

```cpp
struct edge {
    int to, rid, w;
    bool sp;
};
class Solution {
public:
    vector<vector<int>> modifiedGraphEdges(int n, vector<vector<int>>& edges, int source, int destination, int target) {
        vector<edge> g[n];
        for (auto& t: edges)
        {
            int x = t[0], y = t[1], w = t[2];
            bool sp = w < 0;
            if (sp) w = 1;
            g[x].push_back({y, (int)g[y].size(), w, sp});
            g[y].push_back({x, (int)g[x].size() - 1, w, sp});
        }        

        int dist[2][n];
        int delta;
        memset(dist, 0x3f, sizeof dist);
        dist[0][source] = dist[1][source] = 0;

        auto dijkstra = [&](int k)
        {
            bool st[n];
            memset(st, 0, sizeof st);
            for (int i = 0; i < n; i++)
            {
                int t = -1;
                for (int j = 0; j < n; j++)
                    if (!st[j] && (t == -1 || dist[k][t] > dist[k][j])) t = j;
                
                st[t] = true;
                for (int j = 0; j < g[t].size(); j++)
                {
                    int y = g[t][j].to, w = g[t][j].w;
                    if (k == 1 && g[t][j].sp)
                        if (dist[0][y] + delta - dist[1][t] > w)
                        {
                            w = dist[0][y] + delta - dist[1][t];
                            g[t][j].w = w;
                            g[y][g[t][j].rid].w = w;
                        }

                    if (dist[k][y] > dist[k][t] + w)
                        dist[k][y] = dist[k][t] + w;
                }
            }
        };

        dijkstra(0);
        delta = target - dist[0][destination];
        if (delta < 0) return {};

        dijkstra(1);
        cout << dist[1][destination];
        if (dist[1][destination] < target) return {};

        vector<vector<int>> res;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < g[i].size(); j++)
                if (g[i][j].to > i)
                    res.push_back({i, g[i][j].to, g[i][j].w});
        
        return res;
    }
};
```

### 顺丰专场

[链接](https://leetcode.cn/contest/sf-tech) 在这里

**顺丰 04：射线法判断一个点是否在多边形内**

```cpp
class Solution {
public:
    const double eps = 1e-6;
    struct Point {
        double x, y;
        Point(double x = 0, double y = 0): x(x), y(y) {}
        //向量+
        Point operator +(const Point &b)const
        {
            return Point(x+b.x,y+b.y);
        }
        //向量-
        Point operator -(const Point &b)const
        {
            return Point(x-b.x,y-b.y);
        }
        //点积
        double operator *(const Point &b)const
        {
            return x*b.x + y*b.y;
        }
        //叉积
        //P^Q>0,P在Q的顺时针方向；<0，P在Q的逆时针方向；=0，P，Q共线，可能同向或反向
        double operator ^(const Point &b)const
        {
            return x*b.y - b.x*y;
        }
    };
    int dcmp(double x) // 判断两个 double 在 eps 精度下的大小关系
    {
        if (fabs(x) < eps) return 0;
        else return x < 0 ? -1 : 1;
    }
    bool onSeg(Point p1, Point p2, Point q) // 判断 q 是否在 p1 p2 的线段上
    {
        return dcmp((p1 - q) ^ (p2 - q)) == 0 && dcmp((p1 - q) * (p2 - q)) <= 0;
    }

    bool isPointInPolygon(double x, double y, vector<double>& coords) {
        vector<Point> polygon;
        int n = coords.size();
        for (int i = 0; i < n; i += 2) polygon.push_back(Point(coords[i], coords[i + 1]));
        n = polygon.size();
        auto inPolygon = [&](Point p) -> bool 
        {
            bool flag = false;
            Point p1, p2;
            for (int i = 0, j = n - 1; i < n; j = i ++)
            {
                p1 = polygon[i];
                p2 = polygon[j];
                if (onSeg(p1, p2, p)) return true; // 在多边形的一条线上
                if ((dcmp(p1.y - p.y) > 0 != dcmp(p2.y - p.y) > 0) && dcmp(p.x - (p.y - p1.y) * (p1.x - p2.x) / (p1.y - p2.y) - p1.x) < 0)
                    flag = !flag;
            }
            return flag;
        };
        return inPolygon(Point(x, y));
    }
};
```

### LCP 75. 传送卷轴

> 给矩阵，有空地、起点、终点、墙壁。有一次传送的机会，可以在空地上把人传送到镜像位置（落地点也必须是空地）。问被传送后到达终点的最小距离。

题意翻译：`A[i][j]` 表示在 `(i, j)` 被传送后，最少需要移动多少次才能到达魔法水晶，那么问题就变为“从起点出发走到终点，最小化中间经过的 `A[i][j]` 的最大值” 。

先计算出每个点到终点的最短距离（BFS），那么有 `A[i][j] = max(dis[n - 1 - i][j], dis[i][m - 1 - j])`，这一步是好求的。接下来可以**二分答案**，也可以用 `Dijkstra` 求解瓶颈路。

难点在于某些地方传送之后到不了终点，此时要返回 `-1` ，那么如何判断这种情况？答：在这种情况下，`dist[tx][ty] = INF`

```cpp
class Solution {
public:
    int challengeOfTheKeeper(vector<string>& maze) {
        int n = maze.size(), m = maze[0].size();
        int sx, sy, tx, ty;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                if (maze[i][j] == 'S') sx = i, sy = j;
                else if (maze[i][j] == 'T') tx = i, ty = j;
        
        int d[n][m];
        bool st[n][m];
        memset(d, 0x3f, sizeof d);
        memset(st, 0, sizeof st);
        queue<PII> q;
        q.push({tx, ty});
        d[tx][ty] = 0;
        st[tx][ty] = 1;
        while (q.size())
        {
            auto t = q.front();
            q.pop();
            for (int i = 0; i < 4; i++)
            {
                int a = t.x + dx[i], b = t.y + dy[i];
                if (a < 0 || a >= n || b < 0 || b >= m || st[a][b] || maze[a][b] == '#') continue;
                d[a][b] = d[t.x][t.y] + 1;
                q.push({a, b});
                st[a][b] = 1;
            }
        }
        if (d[sx][sy] == 0x3f3f3f3f) return -1; // 提前判断
        
        int A[n][m];
        memset(A, 0, sizeof A);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
            if (maze[i][j] == '.')
                {
                    int a = n - 1 - i, b = j;
                    if (maze[a][b] != '#') A[i][j] = max(A[i][j], d[a][b]);
                    a = i, b = m - 1 - j;
                    if (maze[a][b] != '#') A[i][j] = max(A[i][j], d[a][b]);
                }
        
        int dist[n][m];
        memset(dist, -1, sizeof dist);
        priority_queue<PIII> pq;
        pq.push({-A[sx][sy], {sx, sy}});
        while (pq.size())
        {
            auto t = pq.top();
            pq.pop();
            int d = -t.x, x = t.y.x, y = t.y.y;
            if (dist[x][y] >= 0) continue;
            dist[x][y] = d;
            for (int i = 0; i < 4; i++)
            {
                int a = x + dx[i], b = y + dy[i];
                if (a < 0 || a >= n || b < 0 || b >= m || maze[a][b] == '#') continue;
                pq.push({-max(d, A[a][b]), {a, b}});
            }
        }
        
        if (dist[tx][ty] < 0x3f3f3f3f) return dist[tx][ty];
        return -1;
    }
};
```

### LCP 79. 提取咒文

> 给字符矩阵，初始在左上角，每次移动一个格，问取得给定字符串的所有字符（按顺序）的最少步数，不行则返回 -1 。同一位置可多次提取。

定义状态 `(i, j, k)` 表示当前位置 `(i, j)` 成功提取了 `k` 个字母

起点：`(0, 0, 0)` 终点 `(.. , .. , len)`

这是一个网格图 BFS ，时间复杂度是 $O(mnl)$

技巧是用 `step` 直接记录当前步数，若到终点就返回；每一步都处理提取/不提取

```cpp
class Solution {
public:
    struct point {
        int x, y, k;
    };
    int extractMantra(vector<string>& matrix, string mantra) {
        bool st[110][110][110] = {0};
        int m = matrix.size(), n = matrix[0].size(), len = mantra.size();
        queue<point> q;
        q.push({0, 0, 0});
        st[0][0][0] = 1;
        int step = 1;
        while (q.size())
        {
            int sz = q.size();
            while (sz -- )
            {
                auto t = q.front();
                q.pop();
                int x = t.x, y = t.y, k = t.k;
                if (matrix[x][y] == mantra[k]) 
                {
                    if (k == len - 1) return step;
                    if (!st[x][y][k + 1])
                    {
                        st[x][y][k + 1] = 1;
                        q.push({x, y, k + 1});
                    }
                }
                for (int i = 0; i < 4; i++)
                {
                    int a = x + dx[i], b = y + dy[i];
                    if (a < 0 || a >= m || b < 0 || b >= n || st[a][b][k]) continue;
                    st[a][b][k] = 1;
                    q.push({a, b, k});
                }
            }
            step ++;
        }
        return -1;
    }
};
```

### LCP 80. 生物进化录

> 给一棵树，用 `01` 字符串来表示这棵树，从根节点出发，若前往子节点则加 `0` ，退回父节点则加 `1` ，返回字典序最小的字符串。最终指针可以停在任意位置。

对于树的问题，要想到子问题：对应子树的字典序也是最小，因此可以递归处理

把所有子树的结果排序，然后拼接起来，开头加 0 ，末尾加 1

最后结果去掉开头的 0 和结尾的所有 1 （因为可以停在任意位置）就是答案

问题：为什么排序拼接的结果就是最小？

答：因为 0 和 1 的总数不变，原来字典序最小的，去掉末尾的 1 字典序还是最小的

复杂度分析较为复杂，感性上是 $O(n^2)$

**值得一提的是**，c++ 的 sort 函数对于字符串默认是按字典序来排的，而不是长度

```cpp
class Solution {
public:
    string evolutionaryRecord(vector<int>& parents) {
        int n = parents.size();
        vector<int> g[n];
        for (int i = 1; i < n; i++) g[parents[i]].push_back(i);

        function<string(int)> dfs = [&](int u) ->string
        {
            if (g[u].empty()) return "01";
            vector<string> s;
            for (int x: g[u]) s.push_back(dfs(x));
            sort(s.begin(), s.end());
            string res = "0";
            for (auto& x: s) res += x;
            res += "1";
            return res;
        };

        string res = dfs(0);
        while (res.back() == '1') res.pop_back();
        return res.substr(1);
    }
};
```

### LCP 81. 与非的谜题

> 给数组，其中元素代表一个 k 位二进制数，给若干操作
>
> - 若 `type = 0`，表示修改操作，将谜题数组中下标 `x` 的数字变化为 `y`；
> - 若 `type = 1`，表示运算操作，将数字 `y` 进行 `x*n` 次「与非」操作，第 `i` 次与非操作为 `y = y NAND arr[i%n]`；形象地解释为穿过整个数组 x 次
>     返回所有操作的结果的异或和

二进制的题马上想到拆位，考虑每个比特位穿过数组之后是多少：0 穿过后要么是 0 要么是 1 ，1 穿过后要么是 0 要么是 1

与非运算不满足结合律，不能单纯维护某段区间的与非值，怎么办？某个比特位穿过 `(i,j)` 可以看成是先穿过 `(i,k)` ，得到的结果再穿过 `(k+1,j)` ，这就把一个大问题变成小问题，可以用**线段树**维护！

> 维护 0 从 `[i..k]` 穿过后的值 `l[0]`，1 从 `[i..k]` 穿过后的值 `l[1]`，对于区间 `[k+1..j]` 同理，得 `r[0]` 和 `r[1]`，于是，0 从区间 `[i..j]` 穿过后的值就为 `r[l[0]]`，0 从区间 `[i..j]` 穿过后的值就为 `r[l[1]]`。

当修改一个点时，所有包含这个点的区间值要重新计算

下一个难点：分类讨论比特位 `y` 穿过次数为 `x` ，设穿过一次结果为 `y1`

- `x=1` 时，答案是 `y1`
- `y=y1` 表明穿多少次都是 `y1`
- `y2=y1` 表明从 `y1` 开始穿多少次都不变
- 否则 `y1!=y` `y2!=y1` 又因为取值要么 0 要么 1 ，只能是交替出现，答案具有周期性，与 `x` 的奇偶性有关

学习：当线段树维护的值变成这样时，模板怎么改

```cpp
struct Node {
    int l, r;
    int f[2];
} tr[N << 2];

class Solution {
public:
    int k, msk;
    vector<int> arr;
    void pushup(int u)
    {
        auto &t = tr[u].f, &l = tr[u << 1].f, &r = tr[u << 1 | 1].f;
        t[0] = 0, t[1] = 0;
        for (int i = 0; i < k; i++)
        {
            t[0] |= r[l[0] >> i & 1] & (1 << i);
            t[1] |= r[l[1] >> i & 1] & (1 << i);
        }
    }
    void build(int u, int l, int r)
    {
        tr[u] = {l, r};
        if (l == r) 
        {
            tr[u].f[0] = msk, tr[u].f[1] = ~arr[l];
            return;
        }
        int mid = l + r >> 1;
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
    void update(int u, int x, int v)
    {
        if (tr[u].l == x && tr[u].r == x) tr[u].f[1] = ~v;
        else
        {
            int mid = tr[u].l + tr[u].r >> 1;
            if (x <= mid) update(u << 1, x, v);
            else update(u << 1 | 1, x, v);
            pushup(u);
        }
    }
    int getNandResult(int k, vector<int>& arr, vector<vector<int>>& operations) {
        this->k = k;
        this->arr = arr;
        msk = (1 << k) - 1;
        int n = arr.size();
        build(1, 0, n - 1);
        int r, y, t;
        int res = 0;
        for (auto &op: operations)
            if (op[0])
            {
                for (int i = 0; i < k; i++)
                {
                    y = op[2] >> i & 1;
                    t = tr[1].f[y] >> i & 1;
                    if (op[1] != 1 && t != y && (tr[1].f[t] >> i & 1) != t)
                        t = y ^ (op[1] & 1);
                    res ^= (t << i);
                }
            }
            else update(1, op[1], op[2]);
        return res;
    }
};
```

### 序列 +1-1 问题

题源 [CF13C](https://codeforces.com/problemset/problem/13/C)

- 给定一个序列，每次操作可以把某个数加上 1 或减去 1。要求把序列变成非降数列。最小化操作次数。
    - 最小代价意味着答案序列的每一个数都是原序列中出现过的（证明看题解），感性理解是每个数要么不动，要么就变成 `a[i-1]`
    - 状态定义：$f[i][j]$ 表示前 $i$ 个数满足条件，最大数不大于原序列第 $j$ 个元素的最小代价
    - $f[i][j] = min(f[i][j - 1], f[i - 1][j] + abs(a[i] - b[j]);$

```cpp
int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) 
    {
        scanf("%d", &a[i]);
        b[i] = a[i];
    }
    sort(b + 1, b + 1 + n);
    memset(f, 0x3f, sizeof f);
    for (int i = 1; i <= n; i++) f[0][i] = 0;
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
            f[1][j] = min(f[1][j - 1], f[0][j] + abs(a[i] - b[j]));
        swap(f[0], f[1]); // 本题卡空间，所以用滚动数组
    }
    LL res = 1e18;
    for (int i = 1; i <= n; i++) res = min(res, f[0][i]);
    printf("%lld\n", res);
    return 0;
}
```

- 给定一个有 `n` 个正整数的数组，一次操作中，可以把任意一个元素加一或减一。（元素可被减至负数或 0），求使得原序列严格递增的求最小操作次数。
    - 经典套路：每一个 `a[i]` 让它减去 `i` ，就把问题转化为非严格递增了

```cpp
for (int i = 1; i <= n; i++) 
    {
        scanf("%d", &a[i]);
        a[i] -= i - 1;
        b[i] = a[i];
    }
```

### 树上启发式合并：蓝桥杯周赛二 T7

[dsu-on-tree-oiwiki](https://oi-wiki.org/graph/dsu-on-tree/)

树上每个点有权值，每次询问返回节点 `x` 的所有 `k` 层子节点中，最大权值是多少。`k` 层子节点 `v` 满足

- `v` 在以 `x` 为根的子树中
- 在整棵树中，`dep[v]-dep[x]=k`

`1 <= u,v,k,x,n,q <= 1e5`

分析：用 dsu on tree，对于每个节点维护一个 `map` 存所有 `dep` 的点的最大权值，继承 `sz` 最大的子节点的信息即可

```cpp
vector<int> g[N];
vector<PII> qs[N];
int w[N], sz[N], son[N], dep[N], res[N], n, q, u, v, x, k;
map<int, int> mp[N];

void dfs(int u, int fa)
{
    for (int v: g[u])
    {
        if (v == fa) continue;
        dep[v] = dep[u] + 1;
        dfs(v, u);
        if (son[u] == -1 || sz[v] > sz[son[u]]) son[u] = v;
        sz[u] += sz[v];
    }
    // 启发式合并 继承 sz 最大的子节点（重子节点）
    if (son[u] != -1) mp[u].swap(mp[son[u]]);
    mp[u][dep[u]] = w[u];
    for (int v: g[u])
    {
        if (v == fa || v == son[u]) continue;
        for (auto &p: mp[v])
        {
            int d = p.x, mx = p.y;
            mp[u][d] = max(mp[u][d], mx);
        }
    }
    for (auto &p: qs[u])
        res[p.y] = mp[u][dep[u] + p.x];
}

int main()
{
    scanf("%d%d", &n, &q);
    for (int i = 1; i <= n; i++) 
    {
        scanf("%d", &w[i]);
        sz[i] = 1, son[i] = -1;
    }
    for (int i = 1; i < n; i++)
    {
        scanf("%d%d", &u, &v);
        g[u].push_back(v);
        g[v].push_back(u);
    }
    for (int i = 0; i < q; i++)
    {
        scanf("%d%d", &x, &k);
        qs[x].push_back({k, i});
    }
    dfs(1, 0);
    for (int i = 0; i < q; i++) printf("%d\n", res[i]);
    return 0;
}
```

### Kruskal  重构树：网络稳定性

题意是每次询问求出图中两节点的所有路径中最大化最小边权那条，即瓶颈路

重构树：边权从大到小排，跑 kruskal，在加边 `(u,v,w)` 时，新建一个节点 `p` ，连边 `p->find(u)` `p->find(v)`，再把并查集中 `find(u)` `find(v)` 与 `p` 合并，把 `p` 的点权设为 `w`

这样一来，我们构建出一个二叉堆，其中叶子节点为原图上的点，其余节点都代表了原图的一条边，点权为原图的边权

结论：原图中两个点之间的所有简单路径上最小边权的最大值 = 最大生成树上两个点之间的简单路径上的最小值 = Kruskal 重构树上两点之间的 lca 的权值

为什么就是 lca 的权值：首先最大生成树中，路径是唯一的，最小权值就是这条路径形成时最后一条加的边，此时 `p` 作为两个连通分支的父节点把它们连起来，自然也就是 lca 了

主要考点：最大生成树，重构树，lca（倍增）

```cpp
int n, m, q;
struct edge {
    int u, v, w;
    bool operator<(const edge &t) const {
        return w > t.w;
    }
} e[N];
int p[N];
int find(int x)
{
    if (x != p[x]) p[x] = find(p[x]);
    return p[x];
}

vector<int> g[N];
int val[N];
int f[N][20], dep[N]; // 倍增 深度数组

bool st[N];
void dfs(int u, int fa) // 预处理深度
{
    st[u] = 1;
    // 记录父节点
    f[u][0] = fa;
    dep[u] = dep[fa] + 1;
    for (int x: g[u])
        if (x != fa)
            dfs(x, u);
}

int lca(int u, int v)
{
    if (dep[u] < dep[v]) swap(u, v);
    int d = dep[u] - dep[v];
    for (int i = 19; i >= 0; i--)
        if (d >> i & 1) u = f[u][i];
    if (u == v) return u;
    for (int i = 19; i >= 0; i--) 
        if (f[u][i] != f[v][i]) u = f[u][i], v = f[v][i];
    return f[u][0];
}

int main()
{
    scanf("%d%d%d", &n, &m, &q);
    for (int i = 1; i <= n + m; i++) p[i] = i; // 范围为 n + m
    for (int i = 0, u, v, w; i < m; i++)
        scanf("%d%d%d", &e[i].u, &e[i].v, &e[i].w);
    sort(e, e + m);
    int cur = n;
    for (int i = 0; i < m; i++)
    {
        int u = e[i].u, v = e[i].v, w = e[i].w;
        u = find(u), v = find(v);
        if (u == v) continue;
        ++ cur;
        p[u] = p[v] = cur; // 新增节点 把当前边的两端点作为它的两个儿子 
        g[cur].push_back(u);
        g[cur].push_back(v);
        val[cur] = w; // 点权设为边权
    }
    for (int i = cur; i; i--) 
        if (!st[i]) dfs(i, 0);
    // 倍增
    for (int j = 1; j < 20; j++)
        for (int i = 1; i <= cur; i++)
            f[i][j] = f[f[i][j - 1]][j - 1];
    // 处理询问
    while (q -- )
    {
        int u, v;
        scanf("%d%d", &u, &v);
        if (find(u) != find(v)) puts("-1"); // 不连通
        else printf("%d\n", val[lca(u, v)]);
    }
    return 0;
}
```

### ACW5284 构造矩阵

构造 `n * m` 矩阵，每行和每列的乘积都是 `k`，其中 `k=1` 或 `k=-1`，问方案数

思考：可以先填 `(n-1)*(m-1)` 矩阵，然后再补全；什么时候无解？当 `k=-1` 且 `n+m` 为奇数时：把所有数乘两遍，效果等于每行和每列的结构乘起来，但是前者等于 `1`，后者等于 `-1`，因此无解；类似离散作业题的思路

细节：数据范围 `1e18` 想到快速幂，指数传进 `LL`，写 `qmi(qmi(2,(n-1)%MOD),(m-1)%MOD)` 会过不了，指数这块不能取模


### [2193. 得到回文串的最少操作次数](https://leetcode.cn/problems/minimum-number-of-moves-to-make-palindrome/)

每次交换相邻字符，把字符串变为回文的最小交换次数。

经典贪心思路

关键观察：任何字母在交换前在字符串中的**相对顺序**和交换后在字符串中的相对顺序一致。因为任何时刻都没必要交换两个相同的字母

推论：字母 `c` 相对位置在前一半的，交换后在结果的绝对位置的前一半中，也就是说前一半 `c` 不会跑到后一半中

下一个问题：结果串中前一半的字母到底在什么位置？由于相对顺序不变，相当于把这一半字母按相对顺序挪到结果的前一半中。

然后再处理后半串，剩下的一半字母，假设前一半字母下标是 `1` 到 `n`，可以映射到后一半的字母中，形成一个打乱的 `1-n` 序列，接下来要解决的就是把这个序列排好序的交换次数，这是经典的逆序对统计问题，用树状数组维护。（想象把后半段反转，需要使其排序为 `1-n`）

```cpp
class Solution {
public:
    int minMovesToMakePalindrome(string s) {
        // 每个字母出现的次数
        unordered_map<char, int> freq;
        for (char c: s) {
            ++freq[c];
        }

        int ans = 0;
        // 前一半和后一半
        unordered_map<char, vector<int>> left, right;
        int lcnt = 0, rcnt = 0;

        // 统计「组间交换」的操作次数
        for (int i = 0; i < s.size(); ++i) {
            char c = s[i];
            if (left[c].size() + 1 <= freq[c] / 2) {
                // 属于前一半
                ++lcnt;
                left[c].push_back(lcnt);
                ans += (i - lcnt + 1);
            }
            else {
                // 属于后一半
                ++rcnt;
                right[c].push_back(rcnt);
            }
        }
        
        // 如果长度为奇数，需要在前一半末尾添加一个中心字母
        if (s.size() % 2 == 1) {
            for (auto [c, occ]: freq) {
                if (occ % 2 == 1) {
                    ++lcnt;
                    left[c].push_back(lcnt);
                    break;
                }
            }
        }

        // 得到排列
        vector<int> perm((s.size() + 1) / 2);
        for (auto&& [c, rlist]: right) {
            auto& llist = left[c];
            for (int i = 0; i < rlist.size(); ++i) {
                perm[rlist[rlist.size() - i - 1] - 1] = llist[i];
            }
        }
        reverse(perm.begin(), perm.end());
        
        // 计算逆序对，统计「组内交换」的操作次数
        // 暴力法
        auto get_brute_force = [&]() -> int {
            int n = perm.size();
            int cnt = 0;
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    if (perm[i] > perm[j]) {
                        ++cnt;
                    }
                }
            }
            return cnt;
        };

        // 树状数组法
        auto get_bit = [&]() -> int {
            int n = perm.size();
            vector<int> tree(n + 1);

            auto lowbit = [](int x) {
                return x & (-x);
            };

            auto query = [&](int x) -> int {
                int ret = 0;
                while (x) {
                    ret += tree[x];
                    x -= lowbit(x);
                }
                return ret;
            };

            auto update = [&](int x) {
                while (x <= n) {
                    ++tree[x];
                    x += lowbit(x);
                }
            };

            int cnt = 0;
            for (int i = n - 1; i >= 0; --i) {
                int num = perm[i];
                cnt += query(num - 1);
                update(num);
            }
            return cnt;
        };

        // return ans + get_brute_force();
        return ans + get_bit();
    }
};
```

### 给 3 个人分 n 个物品，每个人不超过 limit 个，问方案数

解法一：考虑枚举第一个人分多少，然后剩下两个人中，第一个人可以分的物品数有上下限，第一个人定下来后第二个人也定下来了，因此可以 $O(1)$ 求出，**难点在于上下限怎么求**

```cpp
long long distributeCandies(int n, int limit) {
    LL res = 0;
    for (int i = 0; i <= n && i <= limit; i++)
    {
        int l = max(0, n - i - limit), r = min(limit, n - i);
        if (l <= r) res += r - l + 1;
    }
    return res;
}
```

解法二：容斥原理

不考虑限制随便分，有 `C(n+2,2)` 种（隔板法可为空）；然后考虑有一个人至少分到 `limit+1` 个，剩下 `(n-limit-1)` 个物品任意分给 3 个人，有 `3*C(n-limit-1+2,2)` 种；然后考虑两个人至少分到 `limit+1`，同理得到 `3*C(n-2*(limit+1)+2,2)`；最后考虑 3 个人都分到至少 `limit+1`，就是 `C(n-3*(limit+1)+2,2)`。由容斥原理，答案是 **所有-1 个不满足 +2 个不满足-3 个不满足**

```cpp
class Solution {
    long long c2(long long n) {
        return n > 1 ? n * (n - 1) / 2 : 0;
    }

public:
    long long distributeCandies(int n, int limit) {
        return c2(n + 2) - 3 * c2(n - limit + 1) + 3 * c2(n - 2 * limit) - c2(n - 3 * limit - 1);
    }
};
```

---

### 斐波那契跳跃

> 博弈论：给定 n 的一个排列，两个人轮流移动棋子
>
> - 只能从小的数跳到大的数
> - 移动的步长必须是斐波那契数
> - 每次移动的步长是递增的
>     问从每个 i 开始游戏，先手必胜还是必败

分析：考虑这样定义状态：`f[i][j][k]` 表示当前在位置 `i`，目前可以移动的最小步长是斐波那契数列的第 `j` 项，现在是第 `k` 个人移动。那么只要枚举那些更大的斐波那契数，设为 `fib[j+x]`，只要可以移动到下一个位置（`a[i+fib[j+x]]>a[i]` 或 `a[i-fib[j+x]]>a[i]`），且那个状态的 `f` 是必败态，就说明当前必胜。这个思路可以通过记忆化搜索来实现

知识点：状态定义，斐波那契数不多-> 暴力枚举

```cpp
bool dp(int i, int j, int who) {
    if (f[i][j][who] != 0) return f[i][j][who] == 1 ? true : false;
    int &ret = f[i][j][who];
    ret = -1;

    for (int jj = j; jj <= MAXP; jj++) {
        int ii = i - fib[jj];
        if (ii > 0 && A[i] < A[ii]) {
            bool t = dp(ii, jj + 1, who ^ 1);
            if (t == false) return ret = 1, true;
        }
        ii = i + fib[jj];
        if (ii <= n && A[i] < A[ii]) {
            bool t = dp(ii, jj + 1, who ^ 1);
            if (t == false) return ret = 1, true;
        }
    }
    return false;
}

int main() {
    fib[0] = fib[1] = 1;
    for (int i = 2; i <= MAXP; i++) fib[i] = fib[i - 1] + fib[i - 2];

    scanf("%d", &n);
    for (int i = 1; i <= n; i++) scanf("%d", &A[i]);
    for (int i = 1; i <= n; i++) for (int j = 1; j <= MAXP; j++) f[i][j][0] = f[i][j][1] = 0;
    for (int i = 1; i <= n; i++) {
        if (dp(i, 1, 0)) printf("Little Lan\n");
        else printf("Little Qiao\n");
    }
    return 0;
}
```

### 星石传送阵

经典图论中虚拟节点问题

> 给定 `n` 个点，每个点有一个值 `x`，对应的 `f(x)=(sum%n)+1`，其中 `sum` 是把 `x` 分成一堆因子，乘积为 `x`，但是要和最小，对应的最小和。
> 边的构成：`f(x)` 相同的点互相可达，值为 `x` 的点可以连到值为 `f(x)` 的点
> 问：从 `a` 到 `b` 的最短路

知识点一：怎么求 `sum`？定理：如果 `a*b=s`，则 `a+b≤s (a,b>1)`

定理的证明可以用二次函数。定理说明要把 `x` 尽可能分解才能得到最小的 `f(x)`，因此考的就是质因数分解后的因数之和

知识点二：怎么表示 `f(x)` 相同的点？建立一个虚拟点，编号为 `t=f(x)+n`，`i->t` 的边权为 `1`，`t->i` 的边权为 `0`。后续所有 `f(x)` 相同的点都可以到 `t` 这个点。（其实 `f(x)` 要模 `n` 也为了方便给这些虚拟节点编号，这样总编号不超过 `2n`）

之后就是常规的 dijkstra 最短路环节

```cpp
long long dijkstra() {
    for (int i = 1; i <= n * 2; i++) dis[i] = -1;
    priority_queue<pli, vector<pli>, greater<pli>> pq;
    pq.push(pli(0, S));
    while (!pq.empty()) {
        pli p = pq.top(); pq.pop();
        int sn = p.second;
        if (dis[sn] >= 0) continue;
        dis[sn] = p.first;
        for (int i = 0; i < e[sn].size(); i++) {
            int fn = e[sn][i];
            if (dis[fn] >= 0) continue;
            pq.push(pli(dis[sn] + v[sn][i], fn));
        }
    }
    return dis[T];
}

int main() {
    for (int i = 2; i <= MAXP; i++) if (!flag[i]) {
        prime.push_back(i);
        for (int j = i * 2; j <= MAXP; j += i) flag[j] = true;
    }

    scanf("%d%d%d", &n, &S, &T);
    for (int i = 1; i <= n; i++) {
        int x; scanf("%d", &x);
        long long sm = 0;
        for (int p : prime) {
            if (p > x) break;
            while (x % p == 0) {
                sm += p;
                x /= p;
            }
        }
        if (x > 1) sm += x;
        sm = sm % n + 1;

        e[i].push_back(sm); v[i].push_back(1);
        e[sm].push_back(i); v[sm].push_back(1);
        e[i].push_back(sm + n); v[i].push_back(1);
        e[sm + n].push_back(i); v[sm + n].push_back(0);
    }

    printf("%lld\n", dijkstra());
    return 0;
}
```

### 字符串，相邻元素限制

从字符串中删除一些元素，使得

- 长度为偶数
- 对于所有奇数 `i`（从 `1` 开始），`s[i]=s[i+1]`

问最少删除次数

1. 贪心：用数组记录当前字符有无出现过，如果有，则答案长度 +2，并清除数组；否则记录当前字符出现过

```cpp
        int n = s.size(), m = 0;
        bool st[26] = {};
        for (auto &i: s)
            if (st[i - 'a'])
            {
                m += 2;
                for (int j = 0; j < 26; j++) st[j] = 0;
            }
            else st[i - 'a'] = 1;
        cout << n - m << endl;
```

1. dp：令 `f[i]` 为考虑前 `i` 个字符，能构成的最大长度（这个定义统一了奇偶数下标的情况），转移就是选或不选

```cpp
        int n = s.size();
        vector<int> pos(26, -1);
        vector<int> f(n);
        for (int i = 0; i < n; i++)
        {
            if (i) f[i] = f[i - 1];
            if (pos[s[i] - 'a'] != -1)
                f[i] = max(f[i], pos[s[i] - 'a'] ? f[pos[s[i] - 'a'] - 1] + 2 : 2);
            pos[s[i] - 'a'] = i;
        }
        cout << n - f[n - 1] << endl;
```

### 合并石子加强版——诈骗题

> 有 `n` 堆石子围成环，相邻合并代价为两者乘积，合并完后个数为两者之和，问合并成一堆的最小代价。

关键在于手玩样例，假设 `a` `b` `c`，`ab+(a+b)c` 等于 `a(b+c)+bc`，因此其实跟顺序没有关系，答案是固定的

解法一：用最小堆模拟过程

解法二：动态维护前缀和，每次 `res+x*sum`，这样算跟最终结果是一样的

**本题其实是蓝桥杯 2020A 组原题：超级胶水**

注意：**此题卡 LL，要用 ULL**


### 求中位数为 m 的子段个数

题源 [CF1005E2](https://codeforces.com/problemset/problem/1005/E2)

知识点一：转化为大于等于 `m` 的子段数 - 中位数大于等于 `m+1` 的子段数

知识点二：怎么知道某个子段中位数大于等于 `m`：`notLess > less`

知识点三：对于 `a[i]`，令 `x[i]=1 if a[i]>=m else x[i] = -1`

做法：遍历数组，令 `sum=x[0]+...+x[i]`，那么 `[j...i]` 这一段的中位数大于等于 `m` 就等价于区间和大于 0。对于右端点 `i`，只要找到所有 `j`，满足 `j` 对应的前缀和小于 `sum`。

技巧一：遍历的时候，`sum` 每次变化量为 1，因此可以直接令 `s[sum]` 表示所有小于等于 `sum` 的前缀和个数。每次变化的实现看代码

技巧二：这里 `sum` 的值域为 `[-n,n]`，用数组维护区间，加个偏移量：初始化 `sum = n`

```cpp
LL get(int k)
{
    vector<int> s(n * 2 + 1);
    int sum = n; // 其实是 0 只是做了偏移
    LL res = 0;
    s[sum] = 1;
    LL cnt = 0;
    for (int i = 1; i <= n; i++)
    {
        if (a[i] < k) cnt -= s[-- sum];
        else cnt += s[sum ++];
        res += cnt;
        s[sum] ++;
    }
    return res;
}

cout << get(m) - get(m + 1) << endl;
```

### 统计 a mod b <= T 的数对数，其中 a ∈ [1,A] b ∈ [1,B]

对于答案的理解

```cpp
int A, B, L, R;
LL get(int lim)
{
    if (lim < 0) return 0;
    LL res = 0;
    // 枚举 f(a, b) 中的 b
    for (int i = 1; i <= B; i++)
    {
        if (i - 1 <= lim) res += A; // 如果所有余数都不超过 lim，那么 a 任意填 
        else 
        {
            // 分成 0 -- i -- 2i -- ... ti -- A
            int t = A / i;
            res += t * (lim + 1); // a = t * x 其中 0<=x<=lim
            t = A % i;
            res += min(t, lim);
        }
    }
    return res;
}
```

### 长度有上限/下限的最大子数组和
> https://codeforces.com/problemset/problem/1796/D
> 
> 你需要把 a 中恰好 k 个数增加 x，其余数减少 x。<br/>
> 该操作必须恰好执行一次。<br/>
> 在最优操作下，a 的最大连续子数组和的最大值是多少？<br/>
> 注意子数组可以是空的，元素和为 0。<br/>

$O(n)$ 做法。

如果 $x<0$，那么可以把 $x$ 变成 $-x$，同时 $k$ 变成 $n-k$。
下面的讨论满足 $x>=0$。

为方便计算，先把所有数都减去 $x$，于是操作变成把 $k$ 个数增加 $2x$。

分类讨论：
1. 如果子数组长度超过 $k$，那么子数组内有 $k$ 个数可以增加 $2x$，总和增加 $2kx$。我们计算的是长度有下限的最大子数组和<br/>
用前缀和思考，`s[right]-s[left]` 最大，那么 `s[left]` 尽量小，且 `right-left > k`，所以枚举 `right` 的同时，要维护 `s[0]` 到 `s[right-k-1]` 的最小值

2. 如果子数组长度不超过 $k$，那么子数组内所有数都可以增加 $2x$。我们计算的是长度有上限的最大子数组和，这可以用**前缀和+单调队列**解决

总的来说，这题同时考察了最大子数组和的长度下限变体和长度上限变体，是一道不错的综合题目

```cpp
    cin >> n >> k >> x;
    if (x < 0) {
        x = -x;
        k = n - k;
    }
    LL res = 0;
    LL pre = 0, mn = 0, pre2 = 0;
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        pre += a[i] - x;
        if (i >= k) {
            res = max(res, pre - mn + 2LL * k * x);
            pre2 += a[i - k] - x;
            mn = min(mn, pre2);
        }
    }
    for (int i = 0; i < n; i++) {
        sum[i + 1] = sum[i] + a[i] + x;
    }
    deque<int> q{0};
    for (int i = 1; i <= n; i++) {
        if (q.front() < i - k) {
            q.pop_front();
        }
        while (q.size() && sum[i] <= sum[q.back()]) {
            q.pop_back();
        }
        q.push_back(i);
        res = max(res, sum[i] - sum[q.front()]);
    }
    cout << res << '\n';
```