ALTER TABLE market_index_daily
    ADD COLUMN currency VARCHAR(8) NULL DEFAULT 'CNY' AFTER amount,
    ADD COLUMN pct_chg_unit VARCHAR(20) NULL AFTER currency,
    ADD COLUMN volume_unit VARCHAR(20) NULL AFTER pct_chg_unit,
    ADD COLUMN amount_unit VARCHAR(20) NULL AFTER volume_unit;

