时间戳：dfs 第一次访问到某个节点的“时间”，这个时间是从 1 递增的

-   一个节点的子树的节点的时间戳，一定大于这个节点的时间戳且连续
-   某些链的时间戳也是连续的

对于操作：

1.  将树从 x 到 y 的最短路径上所有节点值加上 z
2.  求 x 到 y 的最短路径上所有节点的值之和
3.  以 x 为根的子树内所有节点值 + z
4.  求以 x 为根的子树内所有节点值之和

3 和 4 就可以套用连续区间的做法实现了：线段树



树链剖分：把树拆成若干不相交的链

重儿子：所有子节点中，大小最大的那个

重链：从一个轻儿子（根节点也是轻儿子）开始，一路往重儿子走，走出来的链叫重链

dfs 标时间戳时优先往重儿子走，轻儿子无所谓

结论：

-   一条重链上的节点的时间戳都是连续的

对于操作 3

```c++
void mson(int x, int z) {
    modify(dfn[x], dfn[x] + sz[x] - 1, z);
}
```

对于操作 4

```c++
int qson(int x) {
    return query(dfn[x], dfn[x] + sz[x] - 1);
}
```

是不是很简单



引理：除根节点外任何一个节点的父亲都一定在一条重链上

证明：父节点有儿子，就一定有重儿子，说明父节点在重链上

结论：

-   任何一条路径都是由重链的一部分和叶子组成的

对于操作 1 和 2

如果要查询的两个点在同一条重链上，只需要在线段树上查询

否则：

-   维护两个指针指向两个节点，不停让所在链顶部节点深度较大的指针沿着所在重链往上跳
-   即 p 从当前跳到 top[p]
-   一边跳一边在线段树上操作，加完后 p 跳到当前的父亲处，还是在一条重链上
-   循环，直到两指针跳到同一节点或同一重链

```c++
void mchain(int x, int y, int z) {
    while (top[x] != top[y]) {
        if (dep[top[x]] < dep[top[y]]) 
            std::swap(x, y);
        modify(dfn[top[x]], dfn[x], z);
        x = fa[top[x]];
    }
    if (dep[x] > dep[y]) 
        std::swap(x, y);
    modify(dfn[x], dfn[y], z);
}
```



---

模板题的代码：https://www.luogu.com.cn/problem/P3384

```c++
int h[N], e[N * 2], ne[N * 2], idx;
int fa[N], dep[N], son[N], sz[N];
int top[N]; // 每一个节点所属重链的根节点
int dfn[N]; // 时间戳
int w[N]; // dfs 序后节点的权值，用线段树维护
int tim; // 时间戳计数器
int v[N]; // 存放所有节点的权值
int n, m, r, p;

void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++;
}
// 线段树带懒标记
struct Node {
    int l, r, dt, val;
} tr[N * 4];
void pushup(int u)
{
    tr[u].val = ((LL)tr[u << 1].val + tr[u << 1 | 1].val) % p;
}
void pushdown(int u)
{
    auto &root = tr[u], &left = tr[u << 1], &right = tr[u << 1 | 1];
    if (root.dt)
    {
        left.dt += root.dt, right.dt += root.dt;
        left.val = (left.val + (LL)(left.r - left.l + 1) * root.dt % p) % p;
        right.val = (right.val + (LL)(right.r - right.l + 1) * root.dt % p) % p;
        root.dt = 0;        
    }
}

void build(int u, int l, int r)
{
    if (l == r) tr[u] = {l, r, 0, w[l] % p};
    else
    {
        tr[u] = {l, r};
        int mid = l + r >> 1;
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int l, int r, int d)
{
    if (tr[u].l >= l && tr[u].r <= r)
    {
        tr[u].dt += d;
        tr[u].val = (tr[u].val + (LL)(tr[u].r - tr[u].l + 1) * d) % p;
    }
    else
    {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid) modify(u << 1, l, r, d);
        if (r > mid) modify(u << 1 | 1, l, r, d);
        pushup(u);
    }
}
int query(int u, int l, int r)
{
    if (tr[u].l >= l && tr[u].r <= r) return tr[u].val;
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    LL res = 0;
    if (l <= mid) res += query(u << 1, l, r);
    if (r > mid) res += query(u << 1 | 1, l, r);
    return res % p;
}

// 记录重儿子，深度，以及子树大小
void dfs1(int u, int f)
{
    fa[u] = f;
    dep[u] = dep[f] + 1;
    sz[u] = 1;
    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (j == f) continue;
        dfs1(j, u);
        sz[u] += sz[j];
        if (sz[j] > sz[son[u]]) son[u] = j;
    }
}

// 树链剖分
void dfs2(int u, int t)
{
    dfn[u] = ++ tim; 
    top[u] = t; // u 所属重链的父节点
    w[tim] = v[u]; // dfs 序后节点权值
    if (!son[u]) return; // 叶节点
    dfs2(son[u], t);
    for (int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (j == fa[u] || j == son[u]) continue;
        dfs2(j, j);
    }
}

// 操作 1 x-y 链上所有节点加 z
void modifyChain(int x, int y, int z)
{
    z %= p; // 因为答案要模 p，所以先对 z 取模
    while (top[x] != top[y])
    {
        if (dep[top[x]] < dep[top[y]]) 
            std::swap(x, y);
        modify(1, dfn[top[x]], dfn[x], z);
        x = fa[top[x]];
    }
    if (dep[x] > dep[y]) 
        std::swap(x, y);
    modify(1, dfn[x], dfn[y], z);
}

// 操作 2 树从 x 到 y 结点最短路径上所有节点的值之和
int queryChain(int x, int y)
{
    LL res = 0;
    while (top[x] != top[y])
    {
        if (dep[top[x]] < dep[top[y]]) 
            std::swap(x, y);
        res = (res + query(1, dfn[top[x]], dfn[x])) % p;
        x = fa[top[x]];
    }
    if (dep[x] > dep[y])
        std::swap(x, y);
    res = (res + query(1, dfn[x], dfn[y])) % p;
    return res;
}

// 操作 3 以 x 为根节点的子数内所有节点值都加上 z
void modifySon(int x, int z)
{
    modify(1, dfn[x], dfn[x] + sz[x] - 1, z);
}

// 操作 4 求以 x 为根节点的子数内所有节点值之和
int querySon(int x)
{
    return query(1, dfn[x], dfn[x] + sz[x] - 1);
}

int main()
{
    memset(h, -1, sizeof h);
    scanf("%d%d%d%d", &n, &m, &r, &p); // 节点个数 操作个数 根节点序号 取模数
    for (int i = 1; i <= n; i++) scanf("%d", &v[i]);
    for (int i = 1, u, v; i < n; i++)
    {
        scanf("%d%d", &u, &v);
        add(u, v);
        add(v, u);
    }
    dfs1(r, r);
    dfs2(r, r);
    build(1, 1, n);
    for (int i = 1, op, x, y, z; i <= m; i++)
    {
        scanf("%d", &op);
        if (op == 1)
        {
            scanf("%d%d%d", &x, &y, &z);
            modifyChain(x, y, z);
        }
        else if (op == 2)
        {
            scanf("%d%d", &x, &y);
            printf("%d\n", queryChain(x, y) % p);
        }
        else if (op == 3)
        {
            scanf("%d%d", &x, &z);
            modifySon(x, z);
        }
        else 
        {
            scanf("%d", &x);
            printf("%d\n", querySon(x) % p);
        }
    }
    return 0;
}
```

---

