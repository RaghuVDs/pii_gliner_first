# ==========================================================================
# Dockerfile -- PII Engine SaaS Frontend
# Multi-stage build: Node.js (build) -> Nginx (serve)
# ==========================================================================

# --------------------------------------------------------------------------
# Stage 1: Build the React/Vite application
# --------------------------------------------------------------------------
FROM node:20-alpine AS build

WORKDIR /app

# Copy package files for dependency caching
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci --silent

# Copy source code
COPY . .

# Build the production bundle
RUN npm run build

# --------------------------------------------------------------------------
# Stage 2: Serve with Nginx
# --------------------------------------------------------------------------
FROM nginx:alpine AS production

# Remove default nginx content
RUN rm -rf /usr/share/nginx/html/*

# Copy the built assets from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Custom nginx config for SPA routing
RUN cat > /etc/nginx/conf.d/default.conf << 'NGINX'
server {
    listen 3000;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml application/json application/javascript application/xml+rss application/atom+xml image/svg+xml;

    # Cache static assets aggressively
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # SPA fallback: serve index.html for all non-file routes
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to the backend
    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
NGINX

# Expose the configured port
EXPOSE 3000

# Start nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
