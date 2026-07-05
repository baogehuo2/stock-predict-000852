CREATE TABLE IF NOT EXISTS stock_universe_snapshot_raw (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    snapshot_id VARCHAR(64) NOT NULL,
    snapshot_date DATE NOT NULL,
    stock_code VARCHAR(20) NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    stock_name VARCHAR(100),
    listing_date DATE,
    data_source VARCHAR(100) NOT NULL,
    source_url TEXT,
    available_time DATETIME NOT NULL,
    crawl_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    raw_hash CHAR(64),
    UNIQUE KEY uk_stock_universe_snapshot (snapshot_id, exchange, stock_code),
    KEY idx_stock_universe_date (snapshot_date, exchange)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS stock_daily_collection_checkpoint (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    stock_code VARCHAR(20) NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    data_source VARCHAR(100) NOT NULL,
    requested_start_date DATE,
    requested_end_date DATE,
    last_success_date DATE,
    fetched_rows BIGINT DEFAULT 0,
    attempt_count INT DEFAULT 0,
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_stock_checkpoint (stock_code, exchange, data_source),
    KEY idx_stock_checkpoint_status (status, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

