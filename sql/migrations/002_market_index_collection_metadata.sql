ALTER TABLE market_index_daily
    ADD COLUMN source_url TEXT NULL AFTER data_source,
    ADD COLUMN available_time DATETIME NULL AFTER source_url,
    ADD COLUMN crawl_time DATETIME NULL DEFAULT CURRENT_TIMESTAMP AFTER available_time;
