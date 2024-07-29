import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3';

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "钱力古的小博客",
  description: "个人博客分享",
  head: [["link", { rel: "icon", href: "https://cdn.jsdelivr.net/gh/Ricky-Daxia/Hei_Xiu/202407132356265.png" }]],
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config

    logo: 'https://cdn.jsdelivr.net/gh/Ricky-Daxia/Hei_Xiu/202407132356265.png',

    lang: 'zh-CN',

    nav: [
      { 
        text: '算法',
        items: [
          { text: '算法题记一', link: '/algo/note1' },
          { text: '算法题记二', link: '/algo/note2' },
          { text: '动态规划题记', link: '/algo/dp1' },
          { text: '经典套路', link: '/algo/tricks' } 
        ]
      },
      { 
        text: '数学', 
        items: [
          { text: 'Y-Combinator', link: '/math/ycombinator' }
        ] 
      },
      {
        text: '找工',
        items: [
          { text: '笔试题记录', link: 'job/笔试题记录' },
          { text: '软件体系结构', link: 'job/软件体系结构' }
        ]
      },
      {
        text: '模板',
        items: [
          { text: '01-BFS', link: '/template/01BFS' },
          { text: '单调栈', link: '/template/单调栈' },
          { text: '动态区间求并', link: '/template/动态区间求并' },
          { text: '滑动窗口前K小数', link: '/template/滑动窗口前K小数' },
          { text: '矩阵快速幂优化floyd', link: '/template/矩阵快速幂优化floyd' },
          { text: '马拉车与扩展kmp', link: '/template/马拉车与扩展kmp' },
          { text: '普通莫队算法', link: '/template/普通莫队算法' },
          { text: '射线法判断点是否在多边形内', link: '/template/射线法判断点是否在多边形内' },
          { text: '实用函数', link: '/template/实用函数' },
          { text: '树上倍增或lca', link: '/template/树上倍增或lca' },
          { text: '树上启发式合并', link: '/template/树上启发式合并' },
          { text: '树状数组', link: '/template/树状数组' },
          { text: '数位DP', link: '/template/数位DP' },
          { text: '图论', link: '/template/图论' },
          { text: '线段树', link: '/template/线段树' },
          { text: '整体二分', link: '/template/整体二分' },
          { text: '子集', link: '/template/子集' },
          { text: '组合数', link: '/template/组合数' },
          { text: '最大流最小费模板', link: '/template/最大流最小费模板' },
          { text: 'AC自动机', link: '/template/AC自动机' },
          { text: 'dfs序或重链剖分', link: '/template/dfs序或重链剖分' },
          { text: 'go 代码', link: '/template/go代码' },
          { text: 'LIS问题', link: '/template/LIS问题' },
          { text: 'meet-in-the-middle', link: '/template/meet-in-the-middle' },
          { text: 'RMQ的st表实现', link: '/template/RMQ的st表实现' },
          { text: 'Tarjan', link: '/template/Tarjan' },
          { text: 'Trie', link: '/template/Trie' }
        ]
      }
    ],

    // sidebar: [
    //   {
    //     text: '算法',
    //     items: [
    //       { text: '算法题记一', link: '/note1' },
    //       { text: '算法题记二', link: '/note2' },
    //       { text: '动态规划笔记', link: '/dp1' },
    //       { text: '经典套路', link: 'tricks'}
    //     ]
    //   },
    //   {
    //     text: '数学',
    //     items: [
    //       { text: 'Y-Combinator', link: '/ycombinator' }
    //     ]
    //   }
    // ],

    // 右侧文章索引级别
    outline: [2, 6],
    // 右侧索引展示文本
    outlineTitle: "文章目录",
    // git提交时间展示文本
    lastUpdated: {
      text: 'Updated at',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short'
      }
    },
    // md 中使用外部链接时展示额外的图标
    externalLinkIcon: true,
    // 移动端切换主题展示文本
    darkModeSwitchLabel: "切换主题",
    // 移动端展示弹出sidebar展示文本
    sidebarMenuLabel: "菜单",
    // 移动端切换语言展示文本
    langMenuLabel: "切换语言",
    // 回到顶部展示文本
    returnToTopLabel: "回到顶部",
    sidebar: false, // 关闭侧边栏
    aside: "left", // 设置右侧侧边栏在左侧显示

    socialLinks: [
      { icon: 'github', link: 'https://github.com/Ricky-Daxia' },
      {
        icon: {
          svg: '<svg t="1721443125728" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2655" width="200" height="200"><path d="M1023.52385 511.966c0 163.698 0 327.396 0.477 491.094 0 17.052-3.888 20.94-20.94 20.872q-491.094-0.819-982.188 0c-17.051 0-20.94-3.82-20.871-20.872Q0.81985 511.966 0.00085 20.872C-0.06615 3.821 3.82085-0.068 20.87285 0.001q491.094 0.818 982.188 0c17.052 0 21.008 3.82 20.94 20.871-0.477 163.698-0.477 327.396-0.477 491.094z" fill="#FEFEFE" p-id="2656"></path><path d="M454.87785 321.736c-61.387 61.386-124.82 121.068-184.774 184.16-44.404 46.722-40.925 113.906 5.593 161.924 55.18 56.68 111.996 111.724 168.063 167.45 21.212 21.963 20.94 47.063 5.388 70.39-14.392 21.485-35.058 34.65-63.092 22.577a132.664 132.664 0 0 1-35.672-25.578c-51.77-52.315-104.767-103.47-155.99-156.4-89.898-92.83-90.921-232.451-0.478-324.668C320.98585 291.724 449.96585 163.903 578.40085 35.537c28.307-28.306 63.843-30.148 87.033-6.002s21.008 55.657-6.411 84.509q-43.176 45.426-86.692 90.443c-30.693 47.609-71.823 84.645-117.453 117.249z" fill="#070706" p-id="2657"></path><path d="M677.02885 641.015H493.41485c-40.925 0-70.186-24.76-69.504-58.113 0.614-32.262 27.966-55.998 68.208-56.135q187.025-0.886 374.05 0c39.833 0 62.342 22.031 62.683 56.476 0 35.74-23.055 57.294-64.934 57.704-62.137 0.545-124.547 0.068-186.889 0.068z" fill="#B4B2B1" p-id="2658"></path><path d="M386.05585 928.1c60.569-7.366 79.053-37.241 57.704-92.967 63.842 33.49 110.837 26.056 162.47-25.577 26.465-26.465 52.52-53.339 79.598-79.19s57.431-26.464 81.44-2.728 23.532 54.566-1.978 81.44c-34.104 35.195-67.457 70.595-103.47 103.266-76.939 69.777-199.917 75.915-275.764 15.756z" fill="#EAA240" p-id="2659"></path><path d="M454.87785 321.736A1295.942 1295.942 0 0 1 572.33085 204.487c89.693 27.829 142.349 101.152 202.372 164.926 19.575 20.871 11.663 53.747-10.777 72.777a53.338 53.338 0 0 1-73.869-2.183 821.627 821.627 0 0 1-74.414-74.278c-44.13-52.315-96.855-66.502-160.765-43.993z" fill="#EAA340" p-id="2660"></path></svg>'
        },
        link: 'https://leetcode.cn/u/ricky-daxia/'
      },
      { 
        icon: {
          svg: '<svg t="1721444208571" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4510" width="200" height="200"><path d="M880.64 204.8h-737.28A61.44 61.44 0 0 0 81.92 265.0112v493.9776a61.44 61.44 0 0 0 18.0224 43.4176 59.8016 59.8016 0 0 0 41.7792 16.7936h737.28a61.44 61.44 0 0 0 61.44-61.44v-491.52A61.44 61.44 0 0 0 880.64 204.8z m0 573.44h-737.28a20.8896 20.8896 0 0 1-20.48-20.48V341.1968l358.8096 206.848a58.9824 58.9824 0 0 0 61.44 0L901.12 341.1968v417.792a20.48 20.48 0 0 1-20.48 19.2512zM901.12 294.0928l-378.88 218.7264a20.48 20.48 0 0 1-20.48 0L122.88 294.0928v-29.0816A20.48 20.48 0 0 1 143.36 245.76h737.28a20.48 20.48 0 0 1 20.48 20.48v26.624z" p-id="4511"></path></svg>' 
        }, 
        link: 'mailto:1915754435@qq.com' 
      },
      {
        icon: {
          svg: '<svg t="1721444355937" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="5497" width="200" height="200"><path d="M272 134.4c16-6.4 35.2-6.4 51.2 0 12.8 6.4 22.4 12.8 32 22.4l121.6 105.6h86.4l121.6-105.6c9.6-9.6 19.2-16 32-22.4 16-6.4 35.2-3.2 51.2 6.4s25.6 25.6 28.8 44.8c0 16-3.2 28.8-12.8 41.6l-25.6 25.6c-6.4 6.4-9.6 9.6-16 12.8h76.8c35.2 0 67.2 16 89.6 38.4 25.6 22.4 38.4 54.4 41.6 89.6v348.8c0 16 0 35.2-3.2 51.2-6.4 35.2-28.8 67.2-60.8 83.2-22.4 12.8-44.8 19.2-70.4 19.2H256c-19.2 0-35.2 0-54.4-3.2-35.2-9.6-64-28.8-83.2-60.8-12.8-19.2-22.4-44.8-22.4-70.4V419.2v-51.2c12.8-57.6 60.8-102.4 121.6-108.8h80c-12.8-6.4-22.4-19.2-35.2-28.8-12.8-12.8-22.4-32-19.2-48 0-19.2 12.8-38.4 28.8-48m-12.8 233.6c-22.4 3.2-41.6 22.4-48 44.8v307.2c0 28.8 16 51.2 41.6 60.8 9.6 3.2 16 3.2 25.6 3.2h492.8c25.6 0 48-12.8 57.6-35.2 6.4-12.8 6.4-25.6 6.4-38.4v-265.6-28.8c-6.4-19.2-19.2-35.2-38.4-41.6-12.8-3.2-25.6-6.4-38.4-6.4H259.2z" fill="#20B0E3" p-id="5498"></path><path d="M358.4 464c16 0 28.8 3.2 41.6 12.8 12.8 12.8 22.4 28.8 22.4 44.8v60.8c0 12.8-3.2 28.8-12.8 38.4-12.8 16-32 22.4-51.2 22.4s-38.4-12.8-48-28.8c-6.4-12.8-9.6-25.6-6.4-41.6V512c3.2-25.6 25.6-48 51.2-51.2l3.2 3.2z m313.6 0c16 0 32 3.2 44.8 16 12.8 9.6 19.2 25.6 19.2 41.6v60.8c0 12.8-3.2 28.8-9.6 38.4-12.8 16-28.8 25.6-51.2 25.6-19.2 0-38.4-9.6-51.2-25.6-6.4-12.8-9.6-28.8-9.6-41.6v-60.8c6.4-28.8 25.6-51.2 57.6-54.4z" fill="#20B0E3" p-id="5499"></path></svg>'
        },
        link: 'https://space.bilibili.com/109098553'
      }
    ],

    footer: {
      message: "If you have any advice, please leave a message:)",
      copyright: "Copyright@ 2024 Ricky Daxia"
    },
  
    // 搜索功能
    search: {
      // 使用本地搜索
      provider: "local",
      options: {
        // 配置搜索组件展示文本
        translations: {
          button: {
            buttonText: "搜索文档",
          },
          modal: {
            displayDetails: "显示详情",
            noResultsText: "未找到相关结果",
            resetButtonTitle: "清除",
            footer: {
              closeText: "关闭",
              selectText: "选择",
              navigateText: "切换",
            },
          },
        },
      },
    },
  },

  markdown: {
    config: (md) => {
      md.use(mathjax3);
    },
  }
})
