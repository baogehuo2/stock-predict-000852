ALTER TABLE sentiment_guba_raw
    ADD COLUMN available_time DATETIME NULL AFTER crawl_time,
    ADD COLUMN raw_hash CHAR(64) NULL AFTER available_time,
    ADD COLUMN content_hash CHAR(64) NULL AFTER raw_hash,
    ADD COLUMN author_id_hash CHAR(64) NULL AFTER content_hash;

UPDATE sentiment_guba_raw
SET
    available_time = COALESCE(available_time, publish_time, crawl_time),
    raw_hash = COALESCE(raw_hash, SHA2(CONCAT_WS('|',
        COALESCE(post_id, ''),
        COALESCE(source, ''),
        COALESCE(bar_name, ''),
        COALESCE(CAST(publish_time AS CHAR), ''),
        COALESCE(title, ''),
        COALESCE(content, ''),
        COALESCE(url, '')
    ), 256)),
    content_hash = COALESCE(content_hash, SHA2(CONCAT_WS('|',
        COALESCE(title, ''),
        COALESCE(content, '')
    ), 256)),
    author_id_hash = COALESCE(author_id_hash, CASE
        WHEN author IS NULL OR author = '' THEN NULL
        ELSE SHA2(author, 256)
    END);

CREATE INDEX idx_guba_available_time ON sentiment_guba_raw (available_time);

ALTER TABLE sentiment_guba_comment_raw
    ADD COLUMN available_time DATETIME NULL AFTER crawl_time,
    ADD COLUMN raw_hash CHAR(64) NULL AFTER available_time,
    ADD COLUMN content_hash CHAR(64) NULL AFTER raw_hash,
    ADD COLUMN author_id_hash CHAR(64) NULL AFTER content_hash;

UPDATE sentiment_guba_comment_raw
SET
    available_time = COALESCE(available_time, publish_time, crawl_time),
    raw_hash = COALESCE(raw_hash, SHA2(CONCAT_WS('|',
        COALESCE(comment_id, ''),
        COALESCE(post_id, ''),
        COALESCE(source, ''),
        COALESCE(bar_name, ''),
        COALESCE(CAST(publish_time AS CHAR), ''),
        COALESCE(content, ''),
        COALESCE(url, '')
    ), 256)),
    content_hash = COALESCE(content_hash, SHA2(COALESCE(content, ''), 256)),
    author_id_hash = COALESCE(author_id_hash, CASE
        WHEN author IS NULL OR author = '' THEN NULL
        ELSE SHA2(author, 256)
    END);

CREATE INDEX idx_guba_comment_available_time ON sentiment_guba_comment_raw (available_time);

ALTER TABLE news_raw
    ADD COLUMN language VARCHAR(16) NULL AFTER matched_groups,
    ADD COLUMN country_region VARCHAR(30) NULL AFTER language,
    ADD COLUMN source_type VARCHAR(40) NULL AFTER country_region,
    ADD COLUMN first_seen_time DATETIME NULL AFTER source_type,
    ADD COLUMN last_seen_time DATETIME NULL AFTER first_seen_time,
    ADD COLUMN canonical_url TEXT NULL AFTER last_seen_time,
    ADD COLUMN content_hash CHAR(64) NULL AFTER canonical_url,
    ADD COLUMN is_reprint TINYINT NULL AFTER content_hash,
    ADD COLUMN original_news_id VARCHAR(160) NULL AFTER is_reprint,
    ADD COLUMN event_layer_hint VARCHAR(16) NULL AFTER original_news_id,
    ADD COLUMN publisher_country VARCHAR(30) NULL AFTER event_layer_hint,
    ADD COLUMN document_type VARCHAR(30) NULL AFTER publisher_country,
    ADD COLUMN official_source TINYINT NULL AFTER document_type,
    ADD COLUMN source_priority INT NULL AFTER official_source,
    ADD COLUMN available_time DATETIME NULL AFTER crawl_time,
    ADD COLUMN raw_hash CHAR(64) NULL AFTER available_time;

UPDATE news_raw
SET
    language = COALESCE(language, 'zh'),
    country_region = COALESCE(country_region, 'CN'),
    source_type = COALESCE(source_type, CASE
        WHEN source LIKE '%CCTV%' OR source LIKE '%新闻联播%' THEN 'official_media'
        WHEN source LIKE '%经济日历%' THEN 'economic_calendar'
        ELSE 'financial_media'
    END),
    first_seen_time = COALESCE(first_seen_time, crawl_time, publish_time),
    last_seen_time = COALESCE(last_seen_time, crawl_time, publish_time),
    canonical_url = COALESCE(canonical_url, url),
    content_hash = COALESCE(content_hash, SHA2(CONCAT_WS('|',
        COALESCE(title, ''),
        COALESCE(content, '')
    ), 256)),
    is_reprint = COALESCE(is_reprint, 0),
    event_layer_hint = COALESCE(event_layer_hint, 'unknown'),
    publisher_country = COALESCE(publisher_country, 'CN'),
    document_type = COALESCE(document_type, CASE
        WHEN source LIKE '%经济日历%' THEN 'calendar'
        ELSE 'news'
    END),
    official_source = COALESCE(official_source, CASE
        WHEN source LIKE '%CCTV%' OR source LIKE '%新闻联播%' THEN 1
        ELSE 0
    END),
    source_priority = COALESCE(source_priority, CASE
        WHEN source LIKE '%CCTV%' OR source LIKE '%新闻联播%' THEN 2
        WHEN source LIKE '%经济日历%' THEN 4
        ELSE 3
    END),
    available_time = COALESCE(available_time, publish_time, crawl_time),
    raw_hash = COALESCE(raw_hash, SHA2(CONCAT_WS('|',
        COALESCE(news_id, ''),
        COALESCE(source, ''),
        COALESCE(CAST(publish_time AS CHAR), ''),
        COALESCE(title, ''),
        COALESCE(content, ''),
        COALESCE(url, '')
    ), 256));

CREATE INDEX idx_news_available_time ON news_raw (available_time);
CREATE INDEX idx_news_content_hash ON news_raw (content_hash);

CREATE TABLE IF NOT EXISTS sentiment_social_raw (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    content_id VARCHAR(160) NOT NULL,
    source VARCHAR(30) NOT NULL,
    topic VARCHAR(100),
    author_id_hash CHAR(64),
    publish_time DATETIME NOT NULL,
    trade_date DATE NOT NULL,
    title TEXT,
    content MEDIUMTEXT,
    read_count BIGINT,
    comment_count BIGINT,
    like_count BIGINT,
    share_count BIGINT,
    url TEXT,
    available_time DATETIME NOT NULL,
    crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    raw_hash CHAR(64),
    UNIQUE KEY uk_social_content (source, content_id),
    KEY idx_social_trade_date (trade_date, source),
    KEY idx_social_available_time (available_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
