import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import "./test-env"; // Temporary: test env vars

createRoot(document.getElementById("root")!).render(<App />);
