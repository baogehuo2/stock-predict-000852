CREATE TABLE IF NOT EXISTS schema_migration (
    version VARCHAR(80) PRIMARY KEY,
    checksum CHAR(64) NOT NULL,
    applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS data_ingestion_log (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    run_id VARCHAR(64) NOT NULL,
    dataset_name VARCHAR(80) NOT NULL,
    data_source VARCHAR(100),
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    requested_start_date DATE,
    requested_end_date DATE,
    fetched_rows INT DEFAULT 0,
    inserted_rows INT DEFAULT 0,
    updated_rows INT DEFAULT 0,
    skipped_rows INT DEFAULT 0,
    error_rows INT DEFAULT 0,
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    code_version VARCHAR(64),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_ingestion_run (run_id, dataset_name),
    KEY idx_ingestion_dataset (dataset_name, start_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS data_quality_daily (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    check_date DATE NOT NULL,
    dataset_name VARCHAR(80) NOT NULL,
    metric_name VARCHAR(80) NOT NULL,
    metric_value DECIMAL(24,8),
    expected_min DECIMAL(24,8),
    expected_max DECIMAL(24,8),
    status VARCHAR(20) NOT NULL,
    details JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_quality_metric (check_date, dataset_name, metric_name),
    KEY idx_quality_status (check_date, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS market_stock_daily_raw (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    trade_date DATE NOT NULL,
    stock_code VARCHAR(20) NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    stock_name VARCHAR(100),
    open DECIMAL(18,4),
    high DECIMAL(18,4),
    low DECIMAL(18,4),
    close DECIMAL(18,4),
    pre_close DECIMAL(18,4),
    pct_chg DECIMAL(12,8),
    volume DECIMAL(24,4),
    amount DECIMAL(24,4),
    turnover_rate DECIMAL(12,8),
    amplitude DECIMAL(12,8),
    currency VARCHAR(8) DEFAULT 'CNY',
    data_source VARCHAR(100) NOT NULL,
    source_url TEXT,
    available_time DATETIME NOT NULL,
    crawl_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    raw_hash CHAR(64),
    UNIQUE KEY uk_stock_daily (trade_date, stock_code),
    KEY idx_stock_daily_code (stock_code, trade_date),
    KEY idx_stock_daily_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS margin_market_daily (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    trade_date DATE NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    financing_balance DECIMAL(24,4),
    financing_buy_amount DECIMAL(24,4),
    securities_lending_balance DECIMAL(24,4),
    securities_lending_sell_volume DECIMAL(24,4),
    margin_balance DECIMAL(24,4),
    currency VARCHAR(8) DEFAULT 'CNY',
    data_source VARCHAR(100) NOT NULL,
    source_url TEXT,
    release_time DATETIME,
    available_time DATETIME NOT NULL,
    crawl_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_margin_market (trade_date, exchange),
    KEY idx_margin_market_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS etf_fund_daily (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    trade_date DATE NOT NULL,
    etf_code VARCHAR(20) NOT NULL,
    fund_share DECIMAL(24,4),
    unit_nav DECIMAL(18,8),
    accumulated_nav DECIMAL(18,8),
    subscription_status VARCHAR(30),
    redemption_status VARCHAR(30),
    currency VARCHAR(8) DEFAULT 'CNY',
    share_data_source VARCHAR(100),
    nav_data_source VARCHAR(100),
    data_source VARCHAR(100) NOT NULL,
    source_url TEXT,
    available_time DATETIME NOT NULL,
    crawl_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_etf_fund_daily (trade_date, etf_code),
    KEY idx_etf_fund_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS index_futures_contract_daily (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    trade_date DATE NOT NULL,
    product_code VARCHAR(10) NOT NULL,
    contract_code VARCHAR(20) NOT NULL,
    expiry_date DATE,
    open DECIMAL(18,4),
    high DECIMAL(18,4),
    low DECIMAL(18,4),
    close DECIMAL(18,4),
    settle DECIMAL(18,4),
    pre_settle DECIMAL(18,4),
    volume DECIMAL(24,4),
    open_interest DECIMAL(24,4),
    amount DECIMAL(24,4),
    data_source VARCHAR(100) NOT NULL,
    source_url TEXT,
    available_time DATETIME NOT NULL,
    crawl_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_future_contract (trade_date, contract_code),
    KEY idx_future_product (product_code, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS index_constituent_snapshot_raw (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    snapshot_id VARCHAR(64) NOT NULL,
    snapshot_date DATE NOT NULL,
    index_code VARCHAR(20) NOT NULL,
    stock_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(100),
    weight DECIMAL(12,8),
    source_effective_date DATE,
    data_source VARCHAR(100) NOT NULL,
    source_url TEXT,
    available_time DATETIME NOT NULL,
    crawl_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    raw_hash CHAR(64),
    UNIQUE KEY uk_constituent_snapshot (snapshot_id, index_code, stock_code),
    KEY idx_constituent_snapshot_date (index_code, snapshot_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS index_constituent_history (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    index_code VARCHAR(20) NOT NULL,
    stock_code VARCHAR(20) NOT NULL,
    effective_date DATE NOT NULL,
    end_date DATE,
    weight DECIMAL(12,8),
    adjustment_type VARCHAR(30),
    announcement_time DATETIME,
    available_time DATETIME NOT NULL,
    data_source VARCHAR(100) NOT NULL,
    source_url TEXT,
    crawl_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_index_member (index_code, stock_code, effective_date),
    KEY idx_index_member_date (index_code, effective_date, end_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

