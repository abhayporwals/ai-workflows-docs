import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const apiSidebars: SidebarsConfig = {
  apiSidebar: [
    'overview',
    {
      type: 'category',
      label: 'Core API',
      items: [
        'core/models',
        'core/training',
        'core/inference',
        'core/evaluation',
      ],
    },
    {
      type: 'category',
      label: 'Data Processing',
      items: [
        'data/loading',
        'data/preprocessing',
        'data/augmentation',
        'data/validation',
      ],
    },
    {
      type: 'category',
      label: 'Deployment',
      items: [
        'deployment/serving',
        'deployment/monitoring',
        'deployment/scaling',
        'deployment/security',
      ],
    },
    {
      type: 'category',
      label: 'Utilities',
      items: [
        'utils/helpers',
        'utils/config',
        'utils/logging',
        'utils/testing',
      ],
    },
  ],
};

export default apiSidebars; 