ALTER TABLE stock_universe_snapshot_raw
    ADD COLUMN security_status VARCHAR(20) NOT NULL DEFAULT 'listed' AFTER listing_date,
    ADD COLUMN delisting_date DATE NULL AFTER security_status;

CREATE INDEX idx_stock_universe_lifecycle
    ON stock_universe_snapshot_raw (security_status, delisting_date);
