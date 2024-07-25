##### Trie模板

```c++
const int N = 35010;
class Trie {
public:
  int son[N][26], cnt[N], idx;

  Trie() {
    memset(son, 0, sizeof son);
    memset(cnt, 0, sizeof cnt);
    idx = 0;
  }
  
  void insert(string word) {
    int p = 0;
    for (int i = 0; i < word.size(); ++i) {
      int u = word[i] - 'a';
      if (!son[p][u]) son[p][u] = ++idx;
      p = son[p][u];
    }
    cnt[p]++;
  }
  
  bool search(string word) {
    int p = 0;
    for (int i = 0; i < word.size(); ++i) {
      int u = word[i] - 'a';
      if (!son[p][u]) return false;
      p = son[p][u];
    }
    return cnt[p];
  }
  
  bool startsWith(string prefix) {
    int p = 0;
    for (int i = 0; i < prefix.size(); ++i) {
      int u = prefix[i] - 'a';
      if (!son[p][u]) return false;
      p = son[p][u];
    }
    return true;
  }
};
```

---

LC2416

```c++
const int N = 1000010;

int tr[N][26], cnt[N], idx;

class Solution {
public:
    void insert(string& word) {
      int p = 0;
      for (auto& c: word) {
        int u = c - 'a';
        if (!tr[p][u]) {
          tr[p][u] = ++idx;
          memset(tr[idx], 0, sizeof tr[idx]); // 全局数组清空
          cnt[idx] = 0;
        }
        p = tr[p][u];
        cnt[p]++;
      }
    }

    int query(string& word) {
      int p = 0, res = 0;
      for (auto& c: word) {
        int u = c - 'a';
        p = tr[p][u];
        res += cnt[p];
      }
      return res;
    }

    vector<int> sumPrefixScores(vector<string>& words) {
      idx = 0;
      memset(tr[0], 0, sizeof tr[0]); // 防 tle

      for (auto& word: words) insert(word);

      vector<int> res;
      for (auto& word: words) res.push_back(query(word));

      return res;
    }
};
```

---

##### 用 trie 来为每个插入串编号，同时 O(1) 查询 s[i...j]

[转换字符的最小成本](https://leetcode.cn/problems/minimum-cost-to-convert-string-ii/description/)

注意 insert 的实现，以及最后枚举时怎么用 a b 实时添加字符

```c++
int tr[N][26], cnt[N];

class Solution {
public:
    long long minimumCost(string source, string target, vector<string>& original, vector<string>& changed, vector<int>& cost) {
        LL w[210][210];
        for (int i = 0; i < 210; i++)
            for (int j = 0; j < 210; j++)
                w[i][j] = 1e18;
        
        int p = 1;
        memset(tr[0], -1, sizeof tr[0]);
        cnt[0] = -1;
        
        auto insert = [&](string &s)
        {
            int cur = 0;
            for (char c: s)
            {
                int u = c - 'a';
                if (tr[cur][u] == -1)
                {
                    tr[cur][u] = p;
                    memset(tr[p], -1, sizeof tr[p]);
                    cnt[p] = -1;
                    p ++;
                }
                cur = tr[cur][u];
            }
            return cur;
        };
        
        int m = 0;
        for (int i = 0; i < original.size(); i++)
        {
            int s = insert(original[i]), t = insert(changed[i]);
            if (cnt[s] == -1) cnt[s] = m ++;
            if (cnt[t] == -1) cnt[t] = m ++;
            w[cnt[s]][cnt[t]] = min(w[cnt[s]][cnt[t]], (LL)cost[i]);
        }
        for (int k = 0; k < m; k++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++)
                    w[i][j] = min(w[i][j], w[i][k] + w[k][j]);
        
        int n = source.size();
        LL f[n + 1];
        f[n] = 0;
        
        for (int i = n - 1; i >= 0; i--)
        {
            f[i] = 1e18;
            if (source[i] == target[i])
                f[i] = f[i + 1];
            for (int j = i, a = 0, b = 0; j < n; j++)
            {
                a = tr[a][source[j] - 'a'];
                b = tr[b][target[j] - 'a'];
                if (a == -1 || b == -1) break;
                if (cnt[a] != -1 && cnt[b] != -1)
                    f[i] = min(f[i], f[j + 1] + w[cnt[a]][cnt[b]]);
            }
        }
        
        if (f[0] >= 1e18) return -1;
        return f[0];
    }
};
```

---

##### 如何统计一个字符串是否同时属于一个串的前后缀？

题目：[3045. 统计前后缀下标对 II](https://leetcode.cn/problems/count-prefix-and-suffix-pairs-ii/)

转换为长为 m 的前后缀必须相等，这一点可以用 Z 函数解决

只用字典树能不能解决？

技巧：把 s 看作 pair 列表：$[(s[0], s[n-1]), (s[1],s[n-2]), ...,(s[n-1],s[0])]$ 

只要这个列表是 t 的 pair 列表的前缀，那么 s 就是 t 的前后缀

代码上，用结构体+指针实现对于 pair 列表的映射

```c++
struct Node {
    unordered_map<int, Node*> son;
    int cnt = 0;
};

class Solution {
public:
    long long countPrefixSuffixPairs(vector<string> &words) {
        long long ans = 0;
        Node *root = new Node();
        for (string &s: words) {
            int n = s.length();
            auto cur = root;
            for (int i = 0; i < n; i++) {
                int p = (int) (s[i] - 'a') << 5 | (s[n - 1 - i] - 'a');
                if (cur->son[p] == nullptr) {
                    cur->son[p] = new Node();
                }
                cur = cur->son[p];
                ans += cur->cnt;
            }
            cur->cnt++;
        }
        return ans;
    }
};
```

---

### 请在所有长度不超过 M 的**连续**子数组中，找出子数组异或和的最大值

模板，开`son[]`数组的时候，范围多开31倍

```c++
void insert(int x, int v)
{
    int p = 0;
    for (int i = 30; i >= 0; i--)
    {
        int u = x >> i & 1;
        if (!tr[p][u]) tr[p][u] = ++idx;
        p = tr[p][u];
        cnt[p] += v; // +1 表示插入 -1 表示删除
    }
}

int query(int x)
{
    int res = 0, p = 0;
    for (int i = 30; i >= 0; i--)
    {
        int u = x >> i & 1;
        if (cnt[tr[p][!u]]) p = tr[p][!u], res = res * 2 + 1;
        else p = tr[p][u], res *= 2;
    }
    return res;
}

int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++)
    {
        int x;
        scanf("%d", &x);
        s[i] = s[i - 1] ^ x;
    }
    
    insert(s[0], 1); // 计算 s[L - 1] 会用到 s[0]
    int res = 0; // 空数组异或和为 0 
    
    for (int i = 1; i <= n; i++)
    {
        if (i - m - 1 >= 0) insert(s[i - m - 1], -1); // 滑动窗口删除
        res = max(res, query(s[i]));
        insert(s[i], 1);
    }
    
    printf("%d\n", res);
    return 0;
}
```

### 最长公共后缀查询

>   给你两个字符串数组 `wordsContainer` 和 `wordsQuery` 。
>
>   对于每个 `wordsQuery[i]` ，你需要从 `wordsContainer` 中找到一个与 `wordsQuery[i]` 有 **最长公共后缀** 的字符串。如果 `wordsContainer` 中有两个或者更多字符串有最长公共后缀，那么答案为长度 **最短** 的。如果有超过两个字符串有 **相同** 最短长度，那么答案为它们在 `wordsContainer` 中出现 **更早** 的一个。
>
>   请你返回一个整数数组 `ans` ，其中 `ans[i]`是 `wordsContainer`中与 `wordsQuery[i]` 有 **最长公共后缀** 字符串的下标。

**Tag：可以在字典树的节点维护更多东西...**

在字典树的节点额外记录到达此节点的最小串长度和下标，查询时跑到哪个点就更新哪个点为的 idx 为 res[i] 即可

```c++
class Solution {
public:
    struct Node {
        unordered_map<char, Node*> son;
        int mn, idx;
    };
    Node *root = new Node();
    void insert(string &s, int idx) {
        int n = s.size();
        auto cur = root;
        for (int i = n - 1; i >= 0; i--) {
            if (cur->son[s[i]] == nullptr) {
                cur->son[s[i]] = new Node();
            }
            cur = cur->son[s[i]];
            if (cur->mn == 0 || cur->mn > n) {
                cur->mn = n, cur->idx = idx;
            }
        }
    }
    vector<int> stringIndices(vector<string>& w, vector<string>& q) {
        int n = w.size(), m = q.size();
        int mn = 0;
        for (int i = 0; i < n; i++) {
            insert(w[i], i);
            if (w[i].size() < w[mn].size()) {
                mn = i;
            }
        }
        vector<int> res(m);
        for (int i = 0; i < m; i++) {
            int sz = q[i].size();
            auto cur = root;
            bool ok = 0;
            for (int j = sz - 1; j >= 0; j--) {
                if (cur->son[q[i][j]] != nullptr) {
                    cur = cur->son[q[i][j]];
                    res[i] = cur->idx;
                    ok = 1;
                } else {
                    break;
                }
            }
            if (!ok) {
                res[i] = mn;
            }
        }
        return res;
    }
};
```



