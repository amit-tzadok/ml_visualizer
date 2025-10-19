import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
// CopilotKit UI styles
import '@copilotkit/react-ui/styles.css'
import { CopilotKit } from '@copilotkit/react-core'
import App from './App'

const COPILOT_PUBLIC_KEY = import.meta.env.VITE_COPILOTKIT_PUBLIC_KEY ?? "ck_pub_be807f406cc5a80e55ea71dbe7035f83";

createRoot(document.getElementById('root') as HTMLElement).render(
  <StrictMode>
    <CopilotKit publicApiKey={COPILOT_PUBLIC_KEY}>
      <App />
    </CopilotKit>
  </StrictMode>,
)
