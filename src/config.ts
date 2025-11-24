// API Configuration
// For local development with both running locally: http://localhost:5000
// For production/preview: Set VITE_API_URL to your deployed backend URL
// If backend isn't accessible, the app will show an error
export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

console.log("API_URL configured as:", API_URL);

