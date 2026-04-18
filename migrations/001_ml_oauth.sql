-- Migration: ML OAuth — move credentials from Supabase to Railway Postgres
--
-- Apply once. Idempotent (IF NOT EXISTS + ON CONFLICT DO NOTHING).
--
-- Usage (local):
--   psql $DATABASE_URL -f migrations/001_ml_oauth.sql
--
-- Usage (Railway): paste into the SQL console of your Postgres plugin, OR
-- rely on FastAPI's ml_oauth.ensure_schema() which runs on startup.

-- ──────────────────────────────────────────────────────────────────────────
-- 1. App credentials (singleton)
-- ──────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_app_config (
  id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  client_id TEXT,
  client_secret TEXT,
  redirect_uri TEXT,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO ml_app_config (id) VALUES (1) ON CONFLICT DO NOTHING;

-- ──────────────────────────────────────────────────────────────────────────
-- 2. Per-user tokens
-- ──────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_user_tokens (
  user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  access_token TEXT NOT NULL,
  refresh_token TEXT NOT NULL,
  access_token_expires_at TIMESTAMPTZ NOT NULL,
  ml_user_id BIGINT,
  ml_nickname TEXT,
  ml_site_id TEXT,
  scope TEXT,
  last_refreshed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ml_user_tokens_expires_idx
  ON ml_user_tokens (access_token_expires_at)
  WHERE refresh_token IS NOT NULL;

-- ──────────────────────────────────────────────────────────────────────────
-- 3. Seed app credentials from Supabase (run MANUALLY ONCE)
-- ──────────────────────────────────────────────────────────────────────────
--
-- If you have existing credentials in Supabase `ml_configs` and want to
-- copy them over, run this AFTER confirming you still have access to Supabase:
--
-- From Supabase SQL editor, run:
--   SELECT client_id, client_secret, redirect_uri FROM ml_configs
--    WHERE config_key = 'default';
--
-- Then on Railway Postgres, run (substitute actual values, keep secret quoted):
--
--   UPDATE ml_app_config
--      SET client_id = '2299422146528077',
--          client_secret = 'xJ3Z8h....your-actual-secret-here',
--          redirect_uri = 'https://phenolated-maleah-leonine.ngrok-free.dev/api/auth/ml/callback',
--          updated_at = NOW()
--    WHERE id = 1;
--
-- Existing OAuth tokens (access_token / refresh_token) from Supabase are NOT
-- migrated to ml_user_tokens — those require a per-user mapping, and the
-- cleanest approach is to have the user re-click "Подключить Mercado Livre"
-- (triggers fresh OAuth, stores tokens bound to the authenticated user.id).
