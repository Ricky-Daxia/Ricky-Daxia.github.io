### 强连通分量

时间戳 dfn[x]：节点 x 第一次被访问的顺序

回溯值 low[x]：从节点 x 出发所能访问的最早时间戳

时间复杂度 O(n+m)

```c++
vector<int> g[N];
int dfn[N], low[N], tot;
int stk[N], instk[N], top;
int scc[N], sz[N], cnt;

void tarjan(int x) {
    dfn[x] = low[x] = ++ tot;
    stk[++ top] = x, instk[x] = 1;
    for (int y: g[x]) {
        if (!dfn[y]) {
            tarjan(y);
            low[x] = min(low[x], low[y]);
        }
        else if (instk[y]) 
            low[x] = min(low[x], dfn[y]);
    }
    if (dfn[x] == low[x]) {
        ++ cnt;
        while (stk[top] != x) {
            scc[stk[top]] = cnt;
            sz[cnt] ++;
            instk[stk[top]] = 0;
            top --;
        }
        scc[x] = cnt;
        sz[cnt] ++;
        instk[x] = 0;
        top --;
    }
}
```



### 割点

判定规则：

如果 x 不是根节点，当搜索树中存在 x 的子节点 y，满足 low[y] >= dfn[x]，那么 x 是割点

如果 x 是根节点，当搜索树至少存在两个子节点 y1, y2，满足上述条件，那么 x 是割点

证明：从 y 出发，在不通过 x 的前提下，不管走到哪都无法到达比 x 更早访问的节点，故删去 x 后以 y 为根的子树断开

如果有重边和自环，不影响判定

```c++
vector<int> g[N];
int dfn[N], low[N], tot;
int cut[N], root;

void tarjan(int x) {
    dfn[x] = low[x] = ++ tot;
    int child = 0;
    for (int y: g[x]) {
        if (!dfn[y]) {
            tarjan(y);
            low[x] = min(low[x], low[y]);
            if (low[y] >= dfn[x]) {
                child ++;
                if (x != root || child > 1) cut[x] = 1;
            }
        }
        else low[x] = min(low[x], dfn[y]);
    }
}
```



### 割边

判定规则：满足 low[y] > dfn[x] 即可，(x, y) 就是割边

为什么不加等号：因为判定割边时，不允许走 (x, y) 的反边更新 low 值

下列代码中，当 isBridge[x] = 1 时，(fa[x], x) 为割边

```c++
vector<int> g[N];
int dfn[N], low[N], fa[N], tot, cnt;
bool isBridge[N];

void tarjan(int u, int f) {
    fa[u] = f;
    dfn[u] = low[u] = ++ tot;
    for (int v: g[u]) {
        if (!dfn[v]) {
            tarjan(v, u);
            low[u] = min(low[u], low[v]);
            if (low[v] > dfn[u]) {
                isBridge[v] = 1;
                ++ cnt;
            }
        }
        else if (dfn[v] < dfn[u] && v != fa)
            low[u] = min(low[u], dfn[v]);
    }
}
```





### SCC 缩点例题

https://codeforces.com/problemset/problem/427/C

https://www.luogu.com.cn/problem/P2812