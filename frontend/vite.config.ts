import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { createHtmlPlugin } from "vite-plugin-html";
// https://vitejs.dev/config/
export default defineConfig({
    plugins: [
        react(),
        createHtmlPlugin({
            inject: {
                data: {
                    title: process.env.VITE_TITLE
                }
            }
        })
    ],
    build: {
        outDir: "../src/quartapp/static",
        emptyOutDir: true,
        sourcemap: true,
        rollupOptions: {
            output: {
                manualChunks: id => {
                    if (id.includes("@fluentui/react-icons")) {
                        return "fluentui-icons";
                    } else if (id.includes("@fluentui/react")) {
                        return "fluentui-react";
                    } else if (id.includes("node_modules")) {
                        return "vendor";
                    }
                }
            }
        },
        target: "esnext"
    },
    server: {
        proxy: {
            "/content/": "http://localhost:50505",
            "/chat": "http://localhost:50505"
        }
    }
});
