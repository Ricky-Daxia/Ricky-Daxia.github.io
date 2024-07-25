>   给你一个下标从 **0** 开始的二维整数数组 `grid` ，数组大小为 `m x n` 。每个单元格都是两个值之一：
>
>   -   `0` 表示一个 **空** 单元格，
>   -   `1` 表示一个可以移除的 **障碍物** 。
>
>   你可以向上、下、左、右移动，从一个空单元格移动到另一个空单元格。
>
>   现在你需要从左上角 `(0, 0)` 移动到右下角 `(m - 1, n - 1)` ，返回需要移除的障碍物的 **最小** 数目。

```c++
int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, -1, 0, 1};
class Solution {
public:
    // 图的边权只有 0 和 1 使用 0-1bfs 也叫 双端队列bfs
    int minimumObstacles(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        int dis[m][n];
        memset(dis, 0x3f, sizeof dis);
        dis[0][0] = 0;
        deque<pair<int, int>> q;
        q.push_front({0, 0});
        while (!q.empty()) {
            int x = q.front().first, y = q.front().second;
            q.pop_front();
            for (int i = 0; i < 4; i++) {
                int a = x + dx[i], b = y + dy[i];
                if (a >= 0 && a < m && b >= 0 && b < n) {
                    int g = grid[a][b];
                    if (dis[x][y] + g < dis[a][b]) {
                        dis[a][b] = dis[x][y] + g;
                        g == 0 ? q.push_front({a, b}) : q.push_back({a, b});
                        // 边权为 0 就入队头 边权为 1 就入队尾
                    }
                }
            }
        }
        return dis[m - 1][n - 1];
    }
};
```

---

>   现在你将作为玩家参与游戏，按规则将箱子 `'B'` 移动到目标位置 `'T'` ：
>
>   -   玩家用字符 `'S'` 表示，只要他在地板上，就可以在网格中向上、下、左、右四个方向移动。
>   -   地板用字符 `'.'` 表示，意味着可以自由行走。
>   -   墙用字符 `'#'` 表示，意味着障碍物，不能通行。 
>   -   箱子仅有一个，用字符 `'B'` 表示。相应地，网格上有一个目标位置 `'T'`。
>   -   玩家需要站在箱子旁边，然后沿着箱子的方向进行移动，此时箱子会被移动到相邻的地板单元格。记作一次「推动」。
>   -   玩家无法越过箱子。
>
>   返回将箱子推到目标位置的最小 **推动** 次数，如果无法做到，请返回 `-1`。

```c++
struct point {
    int px, py, bbx, bby, d;
}; // 用四元组表示状态
class Solution {
public:
    int dx[4] = {0, 1, 0, -1}, dy[4] = {1, 0, -1, 0};
    bool st[20][20][20][20] = {0};
    int minPushBox(vector<vector<char>>& grid) {
        int m = grid.size(), n = grid[0].size();
        int bx, by, tx, ty, sx, sy;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                if (grid[i][j] == 'S') sx = i, sy = j;
                else if (grid[i][j] == 'B') bx = i, by = j;
                else if (grid[i][j] == 'T') tx = i, ty = j;
        
        deque<point> q;
        q.push_back(point{sx, sy, bx, by, 0});
        st[sx][sy][bx][by] = true;
        
        auto check = [&](int x, int y) 
        {
            return x >= 0 && x < m && y >= 0 && y < n && grid[x][y] != '#';
        };

        while (q.size())
        {
            auto t = q.front();
            q.pop_front();
            int px = t.px, py = t.py, bbx = t.bbx, bby = t.bby, d = t.d;
            if (bbx == tx && bby == ty) return d;

            for (int i = 0; i < 4; i++)
            {
                int npx = px + dx[i], npy = py + dy[i];
                if (!check(npx, npy)) continue;

                if (npx == bbx && npy == bby)
                {
                    int nbx = bbx + dx[i], nby = bby + dy[i];
                    if (!check(nbx, nby) || st[npx][npy][nbx][nby]) continue;
                    st[npx][npy][nbx][nby] = true;
                    q.push_back(point{npx, npy, nbx, nby, d + 1}); // 箱子动了
                }
                else if (!st[npx][npy][bbx][bby])
                {
                    q.push_front(point{npx, npy, bbx, bby, d}); // 箱子未动
                    st[npx][npy][bbx][bby] = true;
                }
            }
        }
        return -1;
    }
};
```

---

