/// <reference types="vite/client" />
interface ImportMetaEnv {
  readonly VITE_COPILOTKIT_PUBLIC_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
