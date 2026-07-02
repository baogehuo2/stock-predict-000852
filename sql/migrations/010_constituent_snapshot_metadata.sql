ALTER TABLE index_constituent_snapshot_raw
    ADD COLUMN constituent_file_date DATE NULL AFTER source_effective_date,
    ADD COLUMN weight_file_date DATE NULL AFTER constituent_file_date,
    ADD COLUMN exchange VARCHAR(30) NULL AFTER stock_name,
    ADD COLUMN weight_unit VARCHAR(20) NOT NULL DEFAULT 'decimal' AFTER weight,
    ADD KEY idx_constituent_snapshot_id (snapshot_id);
