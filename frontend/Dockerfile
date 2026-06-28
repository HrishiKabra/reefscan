# ReefScan frontend — Next.js production server.
FROM node:20-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
# NEXT_PUBLIC_* is baked at build time; the browser (on the host) calls the published port.
ARG NEXT_PUBLIC_REEFSCAN_API=""
ENV NEXT_PUBLIC_REEFSCAN_API=$NEXT_PUBLIC_REEFSCAN_API
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
