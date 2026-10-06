#!/usr/bin/with-contenv bashio
set -e

bashio::log.info "Démarrage de Smart Wardrobe AI"
cd /app
exec python3 -u /app/ai_proxy.py
