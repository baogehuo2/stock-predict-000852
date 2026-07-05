ALTER TABLE etf_fund_daily
    ADD COLUMN exchange VARCHAR(10) NULL AFTER etf_code,
    ADD COLUMN fund_name VARCHAR(100) NULL AFTER exchange,
    ADD COLUMN share_source_url TEXT NULL AFTER share_data_source,
    ADD COLUMN nav_source_url TEXT NULL AFTER nav_data_source,
    ADD COLUMN share_available_time DATETIME NULL AFTER nav_source_url,
    ADD COLUMN nav_available_time DATETIME NULL AFTER share_available_time,
    ADD KEY idx_etf_fund_code (etf_code, trade_date);
