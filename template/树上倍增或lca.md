[LC1483. 树节点的第 K 个祖先]: https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/description/

预处理 $f(i,k)$ 表示 $i$ 的第 $2^k$ 个父节点是谁，从 $v$ 走若干个 2 的次幂步到祖先 $m$ 

怎么算：首先 $f(i,0)=p[i]$ ，$f(i,1)=f(f(i,0),0)$ ，$f(i,2)=f(f(i,1),1)$ ，递推得 $f(i,k)=f(f(i,k-1),k-1)$ ，注意算一个点时要先算完它的祖先，即按拓扑序来，可以用宽搜实现

```c++
class TreeAncestor {
public:
    // 每个节点的父节点要先求好 即按照拓扑序 可用宽搜实现
    vector<vector<int>> f; // 状态转移
    vector<vector<int>> g; // 宽搜 建图
    
    TreeAncestor(int n, vector<int>& p) {
        f = vector<vector<int>>(n, vector<int>(16, -1));
        g = vector<vector<int>>(n);
        
        int root = 0;
        for (int i = 0; i < n; i ++ ) {
            if (p[i] == -1) root = i;
            else {
                g[p[i]].push_back(i);
            }
        }
            
        
        queue<int> q;
        q.push(root);
        
        while (q.size()) {
            int x = q.front();
            q.pop();
            
            for (auto y : g[x]) {
                f[y][0] = x; // 2^0 
                for (int i = 1; i < 16; i ++ ) {
                    if (f[y][i - 1] != -1) // 特判
                        f[y][i] = f[f[y][i - 1]][i - 1]; // 先走 2^(k-1) 步 再走 2^(k-1) 步
                }
                q.push(y);
            }
        }
    }
    
    int getKthAncestor(int x, int k) {
        for (int i = 0; i < 16; i ++ )
            if (k >> i & 1) {
                x = f[x][i];
                if (x == -1) return x;
            }
                
        return x;
    }
};
```

---

lca 模板，对上题：

```c++
class TreeAncestor {
    vector<int> depth;
    vector<vector<int>> pa;
public:
    TreeAncestor(vector<pair<int, int>> &edges) {
        int n = edges.size() + 1;
        int m = 32 - __builtin_clz(n); // n 的二进制长度
        vector<vector<int>> g(n);
        for (auto [x, y]: edges) { // 节点编号从 0 开始
            g[x].push_back(y);
            g[y].push_back(x);
        }

        depth.resize(n);
        pa.resize(n, vector<int>(m, -1));
        function<void(int, int)> dfs = [&](int x, int fa) {
            pa[x][0] = fa;
            for (int y: g[x]) {
                if (y != fa) {
                    depth[y] = depth[x] + 1;
                    dfs(y, x);
                }
            }
        };
        dfs(0, -1);

        for (int i = 0; i < m - 1; i++)
            for (int x = 0; x < n; x++)
                if (int p = pa[x][i]; p != -1)
                    pa[x][i + 1] = pa[p][i];
    }

    int get_kth_ancestor(int node, int k) {
        for (; k; k &= k - 1)
            node = pa[node][__builtin_ctz(k)];
        return node;
    }

    // 返回 x 和 y 的最近公共祖先（节点编号从 0 开始）
    int get_lca(int x, int y) {
        if (depth[x] > depth[y])
            swap(x, y);
        // 使 y 和 x 在同一深度
        y = get_kth_ancestor(y, depth[y] - depth[x]);
        if (y == x)
            return x;
        for (int i = pa[x].size() - 1; i >= 0; i--) {
            int px = pa[x][i], py = pa[y][i];
            if (px != py) {
                x = px;
                y = py;
            }
        }
        return pa[x][0];
    }
};
```

---

洛谷模板题：**最近公共祖先**的离线 Tarjan 算法

1.  任选一个点为根节点，从根节点开始
2.  遍历该点 $u$ 的所有子节点 $v$ ，并标记这些子节点 $v$ 被访问过
3.  若 $v$ 有子节点，返回 2 ，否则下一步
4.  合并 $v$ 到 $u$ 上
5.  寻找与当前点 $u$ 有询问关系的点 $v$ 
6.  若 $v$ 被访问过，则可以确认 $u$ 和 $v$ 的最近公共祖先为 $v$ 被合并到的父节点 $a$ 

基于并查集和 DFS 的实现：

```c++
#include <iostream>
#include <cstring>
using namespace std;

const int N = 500010, M = N * 2;

int n, m, s;
int h[N], e[M], ne[M], idx1;
int qh[N], qe[M], qne[M], idx2;
int p[N], lca[M * 2];
bool st[N];

void add(int a, int b)
{
    e[idx1] = b, ne[idx1] = h[a], h[a] = idx1 ++;
}

void qadd(int a, int b)
{
    qe[idx2] = b, qne[idx2] = qh[a], qh[a] = idx2 ++;
}

int find(int x)
{
    if (x != p[x]) p[x] = find(p[x]);
    return p[x];
}

void tarjan(int u)
{
    st[u] = true;
    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (st[j]) continue;
        tarjan(j);
        p[j] = u;
    }
    
    for (int i = qh[u]; i != -1; i = qne[i])
    {
        int j = qe[i];
        if (st[j])
        {
            lca[i] = find(j);
            //printf("%d %d %d\n", u, j, lca[i]);
            // 由于将每一组查询变为两组，所以2n-1和2n的结果是一样的
            if (i % 2 == 0) lca[i + 1] = lca[i];
            else lca[i - 1] = lca[i];
        }
    }
}

int main()
{
    scanf("%d%d%d", &n, &m, &s);
    for (int i = 1; i <= n; i++) p[i] = i;
    memset(h, -1, sizeof h);
    memset(qh, -1, sizeof qh);
    for (int i = 1; i < n; i++)
    {
        int x, y;
        scanf("%d%d", &x, &y);
        add(x, y), add(y, x);
    }
    for (int i = 1; i <= m; i++)
    {
        int x, y;
        scanf("%d%d", &x, &y);
        qadd(x, y), qadd(y, x);
    }
    tarjan(s);
    for (int i = 0; i < m; i++)
        printf("%d\n", lca[i * 2]);
    
    return 0;
}
```

---

应用：[2846. 边权重均等查询](https://leetcode.cn/problems/minimum-edge-weight-equilibrium-queries-in-a-tree/)

```c++
class Solution {
public:
    vector<int> minOperationsQueries(int n, vector<vector<int>>& edges, vector<vector<int>>& queries) {
        vector<vector<PII>> g(n);
        for (auto &e: edges)
        {
            int x = e[0], y = e[1], w = e[2] - 1;
            g[x].push_back({y, w});
            g[y].push_back({x, w});
        }
        
        int m = 32 - __builtin_clz(n);
        vector<vector<int>> pa(n, vector<int>(m, -1));
        vector<vector<array<int, 26>>> cnt(n, vector<array<int, 26>>(m));
        vector<int> d(n);
        function<void(int, int)> dfs = [&](int x, int fa)
        {
            pa[x][0] = fa;
            for (auto [y, w]: g[x])
                if (y != fa)
                {
                    cnt[y][0][w] = 1;
                    d[y] = d[x] + 1;
                    dfs(y, x);
                }
        };
        dfs(0, -1);
        
        for (int i = 0; i < m - 1; i++)
            for (int x = 0; x < n; x++)
            {
                int p = pa[x][i];
                if (p != -1)
                {
                    int pp = pa[p][i];
                    pa[x][i + 1] = pp;
                    for (int j = 0; j < 26; j++)
                        cnt[x][i + 1][j] = cnt[x][i][j] + cnt[p][i][j];
                }
            }
        
        vector<int> res;
        for (auto &q: queries)
        {
            int x = q[0], y = q[1];
            int len = d[x] + d[y];
            int cw[26] = {0};
            if (d[x] > d[y]) swap(x, y);
            
            for (int k = d[y] - d[x]; k; k = k & (k - 1))
            {
                int i = __builtin_ctz(k);
                int p = pa[y][i];
                for (int j = 0; j < 26; j++)
                    cw[j] += cnt[y][i][j];
                y = p;
            }
            
            if (y != x)
            {
                for (int i = m - 1; i >= 0; i--)
                {
                    int px = pa[x][i], py = pa[y][i];
                    if (px != py)
                    {
                        for (int j = 0; j < 26; j++)
                            cw[j] += cnt[x][i][j] + cnt[y][i][j];
                        x = px, y = py;
                    }
                }
                for (int j = 0; j < 26; j++)
                    cw[j] += cnt[x][0][j] + cnt[y][0][j];
                x = pa[x][0];
            }
            
            int lca = x;
            len -= d[lca] * 2;
            res.push_back(len - *max_element(cw, cw + 26));
        }
        return res;
    }
};
```

[2836. 在传球游戏中最大化函数值](https://leetcode.cn/problems/maximize-value-of-function-in-a-ball-passing-game/)

```c++
class Solution {
public:
    long long getMaxFunctionValue(vector<int>& a, long long k) {
        int n = a.size();
        int f[n][36];
        long long w[n][36];
        for(int i = 0; i < n; ++i) {
            f[i][0] = a[i];
            w[i][0] = i;
        }
        for(int j = 1; j < 36; ++j) {
            for(int i = 0; i < n; ++i) {
                f[i][j] = f[f[i][j-1]][j-1];
                w[i][j] = w[i][j-1] + w[f[i][j-1]][j-1];
            }
        }
        long long res = 0;
        for(int i = 0; i < n; ++i) {
            long long cur = 0;
            int pos = i;
            for(int j = 0; j < 36; ++j) {
                if((k >> j) & 1) {
                    cur += w[pos][j];
                    pos = f[pos][j];
                }
            }
            res = max(res, cur + pos);
        }
        return res;
    }
};
```

---

