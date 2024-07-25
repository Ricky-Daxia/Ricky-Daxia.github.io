#### 345. 牛站

这题是说给 $T$ 条边构成的图，求 $S$ 到 $E$ 恰经过 $N$ 条边（可重复经过）的最短路，保证一定有解。

解法是借鉴**矩阵快速幂**思想的 **floyd 算法**，定义 $f[k][i][j]$ 表示从 $i$ 到 $j$ 恰经过 $k$ 条边的最短路，有 `f[a+b][i][j]=min(f[a+b][i][j], f[a][i][k]+f[b][k][j];` ，把 $N$ 进行二进制分解，然后倍增来求

```C++
#include <iostream>
#include <cstring>
#include <algorithm>
#include <map>
using namespace std;

const int N = 210;
int n, k, T, S, E;
map<int, int> mp;
int g[N][N], res[N][N];

void mul(int c[][N], int a[][N], int b[][N])
{
    static int tmp[N][N];
    memset(tmp, 0x3f, sizeof tmp);
    for (int k = 1; k <= n; k++)
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++)
                tmp[i][j] = min(tmp[i][j], a[i][k] + b[k][j]);
    memcpy(c, tmp, sizeof tmp);
}

void qmi()
{
    memset(res, 0x3f, sizeof res);
    for (int i = 1; i <= n; i++) res[i][i] = 0; // 经过 0 条边
    while (k)
    {
        if (k & 1) mul(res, res, g);
        mul(g, g, g);
        k >>= 1;
    }
}

int main()
{
    cin >> k >> T >> S >> E; // 离散化
    memset(g, 0x3f, sizeof g);
    //这里我们来解释一下为什么不去初始化g[i][i]=0呢？
    //我们都知道在类Floyd算法中有严格的边数限制，如果出现了i->j->i的情况其实在i->i中我们是有2条边的
    //要是我们初始化g[i][i]=0,那样就没边了，影响了类Floyd算法的边数限制！
    if (!mp.count(S)) mp[S] = ++ n;
    if (!mp.count(E)) mp[E] = ++ n;
    S = mp[S], E = mp[E];
    while (T -- )
    {
        int a, b, c;
        cin >> c >> a >> b;
        if (!mp.count(a)) mp[a] = ++ n;
        if (!mp.count(b)) mp[b] = ++ n;
        a = mp[a], b = mp[b];
        g[a][b] = g[b][a] = min(g[a][b], c);
    }
    qmi();
    cout << res[S][E] << endl;
    return 0;
}
```

#### 1426. 魔法

这道题跟上面那道是一样的思想，都是借助快速幂思想做 floyd 。

题意说从 $1$ 到 $n$ ，最多使用 $k$ 次魔法，每次使用可以使当前经过的边权变为相反数，问最终的权和最小值。

核心思路是定义 $f(k,i,j)$ 表示从 $i$ 到 $j$ 最多使用 $k$ 次魔法的最小值，**枚举最后一个分界点 $t$ ，使得从 $i$ 到 $t$ 最多用 $k-1$ 次魔法，从 $t$ 到 $j$ 最多用 1 次魔法**。就有 `f[k][i][j]=min{f[k][i][j], f[k-1][i][t]+f[1][t][j]};` 核心是这里的取 min 运算同样满足结合律（把 min 换成加法就成了矩阵乘法，矩满足结合律才可以先算 `(F(k-2)F(1)) F(1) = F(k-2) (F(1)F(1))`），那么就可以用矩阵快速幂去做了。实际上，取 min 和取 max 都满足结合律。

还有一些边界情况：`k=0` 时即 floyd 算法，`k=1` 时最多用 1 次魔法，可以枚举哪条边使用魔法。

```C++
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 110, M = 2510;

int n, m, K;
LL d[N][N], f[N][N];

struct Edge
{
    int a, b, c;
}edge[M];

void mul(LL c[][N], LL a[][N], LL b[][N])
{
    static LL t[N][N];
    memset(t, 0x3f, sizeof t);
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ )
            for (int k = 1; k <= n; k ++ )
                t[i][j] = min(t[i][j], a[i][k] + b[k][j]);
    memcpy(c, t, sizeof t);
}

LL qmi()
{
    while (K)
    {
        if (K & 1) mul(d, d, f);
        mul(f, f, f);
        K >>= 1;
    }
    return d[1][n];
}

int main()
{
    scanf("%d%d%d", &n, &m, &K);
    memset(d, 0x3f, sizeof d);
    for (int i = 1; i <= n; i ++ ) d[i][i] = 0;
    for (int i = 0; i < m; i ++ )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        d[a][b] = c;
        edge[i] = {a, b, c};
    }
    // floyd
    for (int k = 1; k <= n; k ++ )
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= n; j ++ )
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);

    memcpy(f, d, sizeof f);
    // 使用 1 次
    for (int k = 0; k < m; k ++ )
    {
        int a = edge[k].a, b = edge[k].b, c = edge[k].c;
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= n; j ++ )
                f[i][j] = min(f[i][j], d[i][a] - c + d[b][j]);
    }

    printf("%lld\n", qmi());
    return 0;
}
```



