ALTER TABLE index_futures_contract_daily
    ADD COLUMN contract_multiplier DECIMAL(12,4) NULL AFTER expiry_date,
    ADD COLUMN raw_turnover_10k_cny DECIMAL(24,4) NULL AFTER amount,
    ADD COLUMN expiry_date_source VARCHAR(50) NULL AFTER raw_turnover_10k_cny,
    ADD COLUMN amount_unit VARCHAR(20) NOT NULL DEFAULT 'CNY' AFTER expiry_date_source;
