import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  site: 'https://aymaneo.github.io',
  integrations: [tailwind()],
  markdown: {
    shikiConfig: {
      theme: 'github-dark-default',
    },
  },
});
