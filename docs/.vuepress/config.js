module.exports = {
    title: 'XingYu\'s blog',
    description: '我的个人网站',
    head: [ // 注入到当前页面的 HTML <head> 中的标签
        ['link', {
            rel: 'icon',
            href: '/xy.png'
        }], // 增加一个自定义的 favicon(网页标签的图标)
    ],
    base: '/vuepressBlog/', // 这是部署到github相关的配置
    markdown: {
        lineNumbers: false // 代码块显示行号
    },
    themeConfig: {
        logo: '/xy.png', // 左上角logo
        nav: [ // 导航栏配置
            {
                text: 'Web Developer',
                link: '/web/',
                items: [
                    { text: 'HTML', link: '/web/html' },
                    { text: 'CSS', link: '/web/css' },
                    { text: 'JS', link: '/web/js' },
                    { text: 'VUE', link: '/web/vue' }
                ]
            },
            {
                text: 'Various',
                link: '/various/'
            },
            {
                text: 'Essay',
                link: '/essay/'
            },
            {
                text: 'GitHub',
                link: 'https://baidu.com'
            }
            
        ],
        // sidebar: 'auto', // 侧边栏配置
        sidebarDepth: 2, // 侧边栏显示2级
        sidebar: {
            '/various/': [
                {
                    title: '生活测试',
                    collapsable: true,
                    children: [
                        { title: '生活测试01', path: '/various/01' },
                        { title: '生活测试02', path: '/various/02' },
                        { title: '生活测试03', path: '/various/03' },
                    ]
                }
            ],
            '/vue/': [
                {
                    title: '1',
                    collapsable: false,
                    children: [
                        { title: '1.1', path: '/vue/english/english01' },
                        { title: '1.2', path: '/vue/english/english02' },
                        { title: '1.3', path: '/vue/english/english03' },
                    ]
                }
            ],
            '/web/': [
                {
                    title: '1',
                    collapsable: false,
                    children: [
                        { title: '1.1', path: '/web/math/math01' },
                        { title: '1.2', path: '/web/math/math02' },
                        { title: '1.3', path: '/web/math/math03' },
                    ]
                }
            ],
        },
    }
};
