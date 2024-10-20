模板

```c++
struct TopK {
    int K;
    // st1 保存前 K 小值 st2 保存其他值
    multiset<LL> st1, st2;
    // sum 维护前 st1 中元素和
    LL sum;
    
    TopK(int K): K(K), sum(0) {}
    
    // 调整 st1 和 st2 的大小，保证调整后 st1 保存前 K 小值
    void adjust()
    {
        while (st1.size() < K && st2.size() > 0)
        {
            LL t = *(st2.begin());
            st1.insert(t);
            sum += t;
            st2.erase(st2.begin());
        }
        while (st1.size() > K)
        {
            LL t = *prev(st1.end());
            st2.insert(t);
            st1.erase(prev(st1.end()));
            sum -= t;
        }
    }
    
    // 插入
    void add(LL x)
    {
        if (st2.size() && x >= *(st2.begin())) st2.insert(x);
        else st1.insert(x), sum += x;
        adjust();
    }
    
    // 删除
    void del(LL x)
    {
        auto it = st1.find(x);
        if (it != st1.end()) st1.erase(it), sum -= x;
        else st2.erase(st2.find(x));
        adjust();
    }
};
```

---

例题参见 [100178. 将数组分成最小总代价的子数组 II](https://leetcode.cn/problems/divide-an-array-into-subarrays-with-minimum-cost-ii/)

对顶堆应用：[求窗口内出现次数前 k 大的元素和](https://leetcode.cn/problems/find-x-sum-of-all-k-long-subarrays-ii/description/)

```c++
using PII = pair<int, int>;
using LL = long long;
struct TopK {
    int K;
    // st1 保存前 K 小值 st2 保存其他值
    multiset<PII> st1, st2;
    // sum 维护前 st1 中元素和
    LL sum;
    
    TopK(int K): K(K), sum(0) {}
    
    // 调整 st1 和 st2 的大小，保证调整后 st1 保存前 K 小值
    void adjust()
    {
        while (st1.size() < K && st2.size() > 0)
        {
            PII t = *(st2.begin());
            st1.insert(t);
            sum += 1LL * t.first * t.second;
            st2.erase(st2.begin());
        }
        while (st1.size() > K)
        {
            PII t = *prev(st1.end());
            st2.insert(t);
            st1.erase(prev(st1.end()));
            sum -= 1LL * t.first * t.second;
        }
    }
    
    // 插入
    void add(PII x)
    {
        if (st2.size() && x >= *(st2.begin())) st2.insert(x);
        else st1.insert(x), sum += 1LL * x.first * x.second;
        adjust();
    }
    
    // 删除
    void del(PII x)
    {
        auto it = st1.find(x);
        if (it != st1.end()) st1.erase(it), sum -= 1LL * x.first * x.second;
        else st2.erase(st2.find(x));
        adjust();
    }
};

class Solution {
public:
    vector<long long> findXSum(vector<int>& nums, int k, int x) {
        int n = nums.size();
        vector<LL> res;
        unordered_map<int, int> cnt;
        TopK topk(x);
        for (int i = 0; i < k; i++) {
            cnt[nums[i]] ++;
        }
        // 因为模板维护的是前 x 小的元素，所以这里元素全部取反
        for (auto &[x, c]: cnt) {
            topk.add({-c, -x});
        }
        for (int i = 0; ; i++) {
            res.push_back(topk.sum);
            if (i + k == n) {
                break;
            }
            topk.del({-cnt[nums[i]], -nums[i]});
            cnt[nums[i]] --;
            if (cnt[nums[i]] > 0) {
                topk.add({-cnt[nums[i]], -nums[i]});
            }
            if (cnt[nums[i + k]] > 0) {
                topk.del({-cnt[nums[i + k]], -nums[i + k]});
            }
            cnt[nums[i + k]] ++;
            topk.add({-cnt[nums[i + k]], -nums[i + k]});
        }
        return res;
    }
};
```