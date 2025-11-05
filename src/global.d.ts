declare module '*.css';

declare module './components/CompareDemo' {
  import type { ComponentType } from 'react';
  const Component: ComponentType<Record<string, unknown>>;
  export default Component;
}

declare module './components/KnnDemo' {
  import type { ComponentType } from 'react';
  const Component: ComponentType<Record<string, unknown>>;
  export default Component;
}

declare module './components/MlpDemo' {
  import type { ComponentType } from 'react';
  const Component: ComponentType<Record<string, unknown>>;
  export default Component;
}

declare module './components/PerceptronDemo' {
  import type { ComponentType } from 'react';
  const Component: ComponentType<Record<string, unknown>>;
  export default Component;
}

declare module './components/Welcome' {
  import type { ComponentType } from 'react';
  const Component: ComponentType<Record<string, unknown>>;
  export default Component;
}

// Global status published by demos for AgentPanel to read
declare global {
  interface Window {
    mlvStatus?: {
      classifier?: string;
      equation?: string | null;
      weights?: number[];
      bias?: number;
      updatedAt?: number;
    };
  }
}
