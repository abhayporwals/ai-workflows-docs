import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'AI Workflows & Tools',
  tagline: 'Modern practices, tools, and guides for building AI systems',
  favicon: 'img/favicon.ico',
  url: 'https://your-ai-workflows-docs.example.com',
  baseUrl: '/',
  organizationName: 'abhayporwals',
  projectName: 'ai-workflows-docs',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.ts'),
          editUrl: 'https://github.com/abhayporwals/ai-workflows-docs/tree/main/',
          versions: {
            current: {
              label: '2.5',
              path: '2.5',
            },
          },
        },
        blog: {
          showReadingTime: true,
          editUrl: 'https://github.com/abhayporwals/ai-workflows-docs/tree/main/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      } satisfies Preset.Options,
    ],
  ],
  plugins: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        language: ['en'],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
        searchBarPosition: 'right',
      },
    ],
  ],
  themeConfig: {
    image: 'img/ai-workflows-social-card.jpg',
    navbar: {
      title: 'AI Workflows & Tools',
      logo: {
        alt: 'AI Workflows Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'mainSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          type: 'docsVersionDropdown',
          position: 'right',
        },
        {
          to: '/blog',
          label: 'Blog',
          position: 'left',
        },

        {
          type: 'dropdown',
          label: 'Community',
          position: 'right',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/abhayporwals/ai-workflows-docs',
            },
            {
              label: 'Discussions',
              href: 'https://github.com/abhayporwals/ai-workflows-docs/discussions',
            },
            {
              label: 'Issues',
              href: 'https://github.com/abhayporwals/ai-workflows-docs/issues',
            },
          ],
        },
        {
          type: 'dropdown',
          label: 'Resources',
          position: 'right',
          items: [
            {
              label: 'Releases',
              href: 'https://github.com/abhayporwals/ai-workflows-docs/releases',
            },
            {
              label: 'Contributing',
              to: '/docs/contributing',
            },
            {
              label: 'Changelog',
              to: '/docs/changelog',
            },
          ],
        },
        {
          type: 'search',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/tutorials/getting-started',
            },
            {
              label: 'Concepts',
              to: '/docs/concepts/overview',
            },
            {
              label: 'Tutorials',
              to: '/docs/tutorials/getting-started',
            },
            {
              label: 'Tools',
              to: '/docs/tools/overview',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/abhayporwals/ai-workflows-docs',
            },
            {
              label: 'Discussions',
              href: 'https://github.com/abhayporwals/ai-workflows-docs/discussions',
            },
            {
              label: 'Contributing',
              to: '/docs/contributing',
            },
            {
              label: 'Releases',
              href: 'https://github.com/abhayporwals/ai-workflows-docs/releases',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'Changelog',
              to: '/docs/changelog',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} AI Workflows & Tools. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'yaml', 'json', 'docker'],
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },

  },

} satisfies Config;

export default config;
