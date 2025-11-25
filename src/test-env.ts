// Temporary test file to verify env vars
console.log("=== ENV TEST ===");
console.log("VITE_API_URL:", import.meta.env.VITE_API_URL);
console.log("All VITE_ vars:", Object.keys(import.meta.env).filter(k => k.startsWith('VITE_')));
console.log("All env vars:", Object.keys(import.meta.env));

